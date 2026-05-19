import re
import time

import numpy as np
import pandas as pd
import mlflow
from pdmlabs.mango import scheduler, Tuner

from pdmlabs.experiment.experiment import PdMExperiment
from pdmlabs.evaluation.default_evaluators import DefaultRULEvaluator
from pdmlabs.exceptions.exception import IncompatibleMethodException
from pdmlabs.method.supervised_method import SupervisedMethodInterface


class SupervisedRULPdMExperiment(PdMExperiment):
    """Supervised Remaining Useful Life (RUL) prediction experiment.

    This experiment flavor is for RUL regression where:
    - Target is continuous (time to failure, cycles to failure, etc.)
    - Method must implement SupervisedMethodInterface with RUL prediction
    - Evaluation metrics are regression-based (MAE, MSE, etc.) not classification

    Differs from anomaly detection experiments:
    - Predicts continuous RUL rather than binary anomalies
    - Evaluation based on prediction accuracy, not detection timing
    - May use different postprocessing

    Useful for:
    - Predictive maintenance with remaining life estimates
    - RUL-aware planning and maintenance scheduling

    Raises:
        IncompatibleMethodException: If method is not SupervisedMethodInterface.

    Examples:
        >>> experiment = SupervisedRULPdMExperiment(
        ...     experiment_name='rul-demo',
        ...     pipeline=pipeline,
        ...     param_space={'method_fit_intercept': [True, False]},
        ...     optimization_param='MAE'
        ... )
        >>> results = experiment.execute()
    """
    def execute(self) -> dict:
        """Run supervised RUL prediction experiment.

        Trains RUL regression methods and evaluates on test scenarios:

        1. Fits method on labeled historic RUL data (once)
        2. For each target scenario:
           a. Preprocesses target data
           b. Predicts RUL values
           c. Compares against ground truth
        3. Returns best parameters

        Returns:
            dict: Result dictionary with best_params, best_objective, and metrics.

        Raises:
            IncompatibleMethodException: If method is not SupervisedMethodInterface.
        """
        super()._register_experiment()
        conf_dict = {
            'initial_random': self.initial_random,
            'num_iteration': self.num_iteration,
            'constraint': self.constraint_function,
            # 'batch_size': self.batch_size, currently commented out because of using only scheduler.parallel, more info on issue #97 on Mango - alternatives include using only scheduler.parallel or letting the user decide depending on his hardware
        }

        @scheduler.parallel(n_jobs=self.n_jobs)
        def optimization_objective(**params: dict):
            cached_result = self._check_cached_run(params)

            if cached_result is not None:
                return cached_result

            with mlflow.start_run(experiment_id=self.experiment_id) as parent_run:
                result_scores = []
                result_dates = []
                plot_rul_dictionary={}
                result_labels = []
                end_with_failure = []

                if isinstance(self.pipeline.event_preferences['failure'], list):
                    if len(self.pipeline.event_preferences['failure']) == 0:
                        run_to_failure_scenarios = True
                    else:
                        run_to_failure_scenarios = False
                elif self.pipeline.event_preferences['failure'] is None:
                    run_to_failure_scenarios = True
                else:
                    run_to_failure_scenarios = False

                method_params = {re.sub('method_', '', k): v for k, v in params.items() if 'method' in k}
                current_method = self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params)
                if "match_sources" not in self.pipeline.dataset:
                    self.pipeline.dataset["match_sources"]= {source: source for source in self.pipeline.dataset["target_sources"]}
                if not isinstance(current_method, SupervisedMethodInterface):
                    raise IncompatibleMethodException('Expected a supervised method to be provided')
                ### Check if data are compatible
                if "anomaly_labels" not in self.pipeline.dataset:
                    raise ValueError(
                        "The pipeline dataset must contain 'anomaly_labels' for supervised classification experiment.")
                assert len(self.historic_data) == len(self.pipeline.dataset[
                                                         "anomaly_labels"]), "The number of historic data sources and anomaly_labels must match."
                for eni, (hs_data, anomaly_range) in enumerate(
                        zip(self.historic_data, self.pipeline.dataset["anomaly_labels"])):
                    assert len(hs_data) == len(
                        anomaly_range), "The number of historic data sources and anomaly_labels must match."

                for eni, (hs_data, labs) in enumerate(
                        zip(self.target_data, self.pipeline.dataset["target_labels"])):
                    assert len(hs_data) == len(
                        labs), "The number of target data sources and target_labels must match."

                preprocessor_params = {re.sub('preprocessor_', '', k): v for k, v in params.items() if 'preprocessor' in k}
                current_preprocessor = self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params)

                postprocessor_params = {re.sub('postprocessor_', '', k): v for k, v in params.items() if 'postprocessor' in k}
                current_postprocessor = self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences, **postprocessor_params)

                thresholder_params = {re.sub('thresholder_', '', k): v for k, v in params.items() if 'thresholder' in k}
                current_thresholder = self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences,
                                                                **thresholder_params)

                try:
                    fit_time=time.time()
                    new_historic_data = []
                    for current_historic_data, current_historic_source in zip(self.historic_data, self.historic_sources):
                        current_dates = self.pipeline.historic_dates
                        # if the user passed a string take the corresponding column of the historic_data as 'dates' for the evaluation
                        if isinstance(current_dates, str):
                            name=current_dates
                            current_dates = pd.to_datetime(current_historic_data[current_dates])
                            current_dates=[date for date in current_dates]
                            # also drop the corresponding column from the historic_data df
                            current_historic_data = current_historic_data.drop(name, axis=1)
                        # current_historic_data.index = current_dates
                        new_historic_data.append(current_historic_data)

                    from pdmlabs.pipeline.mlflow_pipeline import RULPdMPipeline
                    pdm_pipeline = RULPdMPipeline(
                        preprocessor=current_preprocessor,
                        method=current_method,
                        postprocessor=current_postprocessor,
                        thresholder=current_thresholder
                    )
                    pdm_pipeline.fit(new_historic_data, self.historic_sources, self.event_data, self.pipeline.dataset["anomaly_labels"])
                    
                    fit_time=time.time() - fit_time
                    mlflow.log_metric("fit_time", fit_time)

                    if "is_failure" not in self.pipeline.dataset.keys():
                        self.pipeline.dataset["is_failure"] = [1]*len(self.pipeline.dataset["target_sources"])
                    
                    inference_time = time.time()
                    for current_target_data, current_target_source,current_labels,rtf in zip(self.target_data, self.target_sources,self.pipeline.dataset["target_labels"],self.pipeline.dataset["is_failure"]):
                        # print(i)
                        # i += 1
                        current_dates = self.pipeline.target_dates
                        # if the user passed a string take the corresponding column of the target_data as 'dates' for the evaluation
                        if isinstance(current_dates, str):
                            name=current_dates
                            current_dates = pd.to_datetime(current_target_data[current_dates])
                            current_dates=[date for date in current_dates]
                            # also drop the corresponding column from the target_data df
                            current_target_data = current_target_data.drop(name, axis=1)

                        current_target_data.index = current_dates
                        current_target_source_fitted= self.pipeline.dataset["match_sources"][current_target_source]
                        processed_target_scores = pdm_pipeline.predict_scores_only(
                            target_data=current_target_data,
                            source=current_target_source_fitted,
                            event_data=self.event_data
                        )


                        if self.debug:
                            plot_rul_dictionary[current_target_source]={"scores":processed_target_scores,"labels":current_labels,"thresholds":None,"index":current_dates,"rtf":rtf}

                        # if not run_to_failure_scenarios:
                        #     is_failure, current_scores_splitted, current_dates_splitted, _ = split_into_episodes(processed_target_scores, current_failure_dates, current_dates)
                        # else:
                        #     is_failure = [1]

                        current_scores_splitted = [processed_target_scores]
                        current_dates_splitted = [current_dates]


                        result_scores.extend(current_scores_splitted)
                        result_dates.extend(current_dates_splitted)
                        result_labels.append(current_labels)
                        end_with_failure.append(rtf)

                    inference_time = time.time() - inference_time
                    mlflow.log_metric("inference_time", inference_time)
                except Exception as e:
                    if self.debug:
                        raise e
                    print(e)
                    print("Assing score 0 and continuing to the next experiment.")
                    self._finish_run(parent_run=parent_run, current_steps={
                        'preprocessor': current_preprocessor,
                        'method': current_method,
                        'postprocessor': current_postprocessor,
                        'thresholder': current_thresholder
                    })
                    return 0
                best_metrics_dict = self._run_evaluators(
                    DefaultRULEvaluator(debug=self.debug),
                    result_scores=result_scores,
                    result_dates=result_dates,
                    result_labels=result_labels,
                    plot_dictionary=plot_rul_dictionary,
                    rtfs=end_with_failure
                )

                if "best" in self.extra_metrics:
                    if best_metrics_dict[self.optimization_param] > self.extra_metrics["best"] and self.maximize:
                        self.extra_metrics["best"] = best_metrics_dict[self.optimization_param]
                        self.extra_metrics["th"] = best_metrics_dict["threshold_auc"]
                        self.best_pipeline = pdm_pipeline
                    elif best_metrics_dict[self.optimization_param] < self.extra_metrics["best"] and not self.maximize:
                        self.extra_metrics["best"] = best_metrics_dict[self.optimization_param]
                        self.extra_metrics["th"] = best_metrics_dict["threshold_auc"]
                        self.best_pipeline = pdm_pipeline
                else:
                    self.extra_metrics["best"] = best_metrics_dict[self.optimization_param]
                    self.extra_metrics["th"] = best_metrics_dict["threshold_auc"]
                    self.best_pipeline = pdm_pipeline

                self._plot_RUL(plot_rul_dictionary)

                self._finish_run(parent_run=parent_run, current_steps={
                    'preprocessor': current_preprocessor,
                    'method': current_method,
                    'postprocessor': current_postprocessor,
                    'thresholder': current_thresholder
                })

            return best_metrics_dict[self.optimization_param]

        tuner = Tuner(self.param_space, optimization_objective, conf_dict=conf_dict)
        if self.maximize:
            results=tuner.maximize()
        else:
            results = tuner.minimize()
        dict_ro_return = {}
        dict_ro_return['best_params'] = results['best_params']
        dict_ro_return["best_objective"] = results["best_objective"]
        dict_ro_return["th"] = self.extra_metrics["th"]
        
        if hasattr(self, 'best_pipeline'):
            try:
                # TODO: use a flag parameter to decide whether to log the best pipeline or not, as it can be time consuming and take a lot of space in the MLflow tracking server
                with mlflow.start_run(experiment_id=self.experiment_id, run_name="Best_Pipeline_Model"):
                    mlflow.pyfunc.log_model(artifact_path="best_pdm_pipeline", python_model=self.best_pipeline)
            except Exception as e:
                print(f"Warning: Failed to log MLflow pipeline model: {e}")
                
        return self._finish_experiment(dict_ro_return)