import re
import time

import numpy as np
import pandas as pd
import mlflow
from pdmlabs.mango import scheduler, Tuner

from pdmlabs.experiment.experiment import PdMExperiment
from pdmlabs.evaluation.default_evaluators import DefaultADEvaluator
from pdmlabs.evaluation.evaluation import AUCPR_new as pdm_evaluate, breakIntoEpisodes as split_into_episodes
from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.exceptions.exception import IncompatibleMethodException
from pdmlabs.method.supervised_method import SupervisedMethodInterface


class SupervisedPdMExperiment(PdMExperiment):
    """Supervised anomaly detection with labeled anomaly windows.

    This experiment flavor is designed for scenarios where you have:
    - Training data with explicit anomaly labels (ranges or boolean arrays)
    - A supervised method that can learn from these labels
    - Target data to evaluate on

    It implements a train-once, test-many approach:
    1. Fits method, preprocessor, postprocessor on ALL historic labeled data (once)
    2. Then applies to each target scenario independently

    This differs from semi-supervised (which fits per-scenario) and ensures
    consistent model training across all test scenarios.

    Attributes:
        pipeline (PdMPipeline): Must have 'anomaly_labels' key in dataset.
            Lists of arrays matching historic_data in length and dimensionality.
        param_space (dict): Hyperparameter search space.

    Raises:
        ValueError: If dataset lacks 'anomaly_labels' key.
        ValueError: If anomaly_labels length does not match historic_data.
        IncompatibleMethodException: If method does not implement SupervisedMethodInterface.

    Examples:
        >>> dataset = {
        ...     'historic_data': [df_train],
        ...     'target_data': [df_test],
        ...     'anomaly_labels': [label_array],  # 1D array of same length as df_train
        ...     ...
        ... }
        >>> from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
        >>> experiment = SupervisedPdMExperiment(
        ...     experiment_name='supervised-demo',
        ...     pipeline=pipeline,
        ...     param_space={'method_nu': [0.05, 0.1, 0.2]},
        ...     num_iteration=20
        ... )
        >>> results = experiment.execute()
    """
    def execute(self) -> dict:
        """Run supervised experiment with labeled anomaly training data.

        Trains a supervised method once on all labeled historic data, then evaluates
        on each target scenario:

        1. Preprocesses all historic data (fit once)
        2. Fits method on all labeled historic data (single consolidated training)
        3. Fits postprocessor on labeled data
        4. For each target:
           a. Preprocesses using the fitted preprocessor
           b. Applies fitted method to get anomaly scores
           c. Postprocesses scores
           d. Thresholds to get binary predictions
           e. Evaluates against ground truth
        5. Returns best parameters found

        This approach ensures the model is trained consistently across all test scenarios,
        unlike semi-supervised where the model adapts per-scenario.

        Returns:
            dict: Result dictionary with:
                - 'best_params': Best parameter combination found
                - 'best_objective': Best metric value
                - 'th': Best decision threshold

        Raises:
            ValueError: If anomaly_labels dimension mismatches data.
            IncompatibleMethodException: If method is not SupervisedMethodInterface.

        Examples:
            >>> results = experiment.execute()
            >>> print(f"Best threshold: {results['th']:.3f}")
            Best threshold: 0.645
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
                result_thresholds = []
                results_isfailure =[]
                plot_dictionary={}


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

                preprocessor_params = {re.sub('preprocessor_', '', k): v for k, v in params.items() if 'preprocessor' in k}
                current_preprocessor = self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params)

                postprocessor_params = {re.sub('postprocessor_', '', k): v for k, v in params.items() if 'postprocessor' in k}
                current_postprocessor = self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences, **postprocessor_params)

                thresholder_params = {re.sub('thresholder_', '', k): v for k, v in params.items() if 'thresholder' in k}
                current_thresholder = self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences, **thresholder_params)
                try:
                    fit_time_start=time.time()
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
                        current_historic_data.index = current_dates
                        new_historic_data.append(current_historic_data)

                    from pdmlabs.pipeline.mlflow_pipeline import SupervisedPdMPipeline
                    pdm_pipeline = SupervisedPdMPipeline(
                        preprocessor=current_preprocessor,
                        method=current_method,
                        postprocessor=current_postprocessor,
                        thresholder=current_thresholder
                    )
                    pdm_pipeline.fit(new_historic_data, self.historic_sources, self.event_data, self.pipeline.dataset["anomaly_labels"])

                    fit_time=time.time() - fit_time_start
                    mlflow.log_metric("fit_time", fit_time)

                    inference_time_start = time.time()
                    # i = 0
                    for current_target_data, current_target_source in zip(self.target_data, self.target_sources):
                        # print(i)
                        # i += 1
                        current_failure_dates = self.pipeline.extract_failure_dates_for_source(current_target_source)

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
                        inference_results = pdm_pipeline.predict({
                            'target_data': current_target_data,
                            'source': current_target_source_fitted,
                            'event_data': self.event_data
                        })
                        processed_target_scores = inference_results['scores']
                        current_thresholds = inference_results['dynamic_thresholds']

                        if self.debug:
                            plot_dictionary[current_target_source]={"scores":processed_target_scores,"failures":current_failure_dates,"thresholds":current_thresholds,"index":current_dates}

                        if not run_to_failure_scenarios:
                            is_failure, current_scores_splitted, current_dates_splitted, current_thresholds_splitted = split_into_episodes(processed_target_scores, current_failure_dates, current_thresholds, current_dates)
                        else:
                            is_failure = [1]
                            current_scores_splitted = [processed_target_scores]
                            current_dates_splitted = [current_dates]
                            current_thresholds_splitted = [current_thresholds]


                        result_thresholds.extend(current_thresholds_splitted)
                        result_scores.extend(current_scores_splitted)
                        results_isfailure.extend(is_failure)
                        result_dates.extend(current_dates_splitted)
                    
                    inference_time = time.time() - inference_time_start
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
                    DefaultADEvaluator(debug=self.debug),
                    result_scores=result_scores,
                    result_dates=result_dates,
                    results_isfailure=results_isfailure,
                    plot_dictionary=plot_dictionary
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
                self._plot_scores(plot_dictionary, best_metrics_dict)

                self._finish_run(parent_run=parent_run, current_steps={
                    'preprocessor': current_preprocessor,
                    'method': current_method,
                    'postprocessor': current_postprocessor,
                    'thresholder': current_thresholder
                })

            return best_metrics_dict[self.optimization_param]

        tuner = Tuner(self.param_space, optimization_objective, conf_dict=conf_dict)
        if self.maximize:
            results = tuner.maximize()
        else:
            results = tuner.minimize()
        dict_ro_return = {}
        dict_ro_return['best_params'] = results['best_params']
        dict_ro_return["best_objective"] = results["best_objective"]
        dict_ro_return["th"] = self.extra_metrics["th"]
        
        if hasattr(self, 'best_pipeline'):
            self.best_pipeline.set_global_threshold(self.extra_metrics["th"])
            try:
                # TODO: use a flag parameter to decide whether to log the best pipeline or not, as it can be time consuming and take a lot of space in the MLflow tracking server
                with mlflow.start_run(experiment_id=self.experiment_id, run_name="Best_Pipeline_Model"):
                    mlflow.pyfunc.log_model(artifact_path="best_pdm_pipeline", python_model=self.best_pipeline)
            except Exception as e:
                print(f"Warning: Failed to log MLflow pipeline model: {e}")
                
        return self._finish_experiment(dict_ro_return)