import pandas as pd
import mlflow
from pdmlabs.mango import scheduler

from pdmlabs.experiment.experiment import PdMExperiment
from pdmlabs.pipeline.pipeline import split_step_params
from pdmlabs.evaluation.default_evaluators import DefaultADEvaluator
from pdmlabs.evaluation.evaluation import AUCPR_new as pdm_evaluate, breakIntoEpisodes as split_into_episodes
from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
from pdmlabs.exceptions.exception import IncompatibleMethodException


class UnsupervisedPdMExperiment(PdMExperiment):
    """Unsupervised anomaly detection without any labeled data.

    This experiment flavor is designed for scenarios where:
    - You have no labeled anomaly data
    - You want the method to learn patterns from the entire dataset
    - Each target scenario is evaluated independently

    The approach fits the method on a per-scenario basis (similar to semi-supervised)
    but without leveraging any labels. The method must determine anomalies based on
    statistical or distributional properties alone.

    Suitable for:
    - Early-stage PdM where no failure labels exist
    - Discovering unknown failure modes
    - Baseline comparisons

    Raises:
        IncompatibleMethodException: If method does not implement UnsupervisedMethodInterface.

    Examples:
        >>> from pdmlabs.method.isolation_forest import IsolationForest
        >>> experiment = UnsupervisedPdMExperiment(
        ...     experiment_name='unsupervised-demo',
        ...     pipeline=pipeline,
        ...     param_space={'method_contamination': [0.01, 0.05, 0.1]},
        ...     num_iteration=20
        ... )
        >>> results = experiment.execute()
    """
    def execute(self) -> dict:
        """Run unsupervised experiment without labeled training data.

        For each parameter combination:
        1. For each target scenario (segmented by reset/failure events):
           a. Fits method using only the entire scenario data (unsupervised)
           b. Predicts anomaly scores
           c. Applies postprocessor and thresholder
        2. Evaluates using PdM metrics to find best parameters

        Returns:
            dict: Result dictionary with best_params, best_objective, and threshold.

        Raises:
            IncompatibleMethodException: If method is not UnsupervisedMethodInterface.
        """
        super()._register_experiment()

        trial_sink = self._new_trial_sink()

        def optimization_objective(**params: dict):
            cached_result = self._check_cached_run(params)

            if cached_result is not None:
                cached_score, cached_extras = cached_result
                trial_sink.record(cached_score, None, params=params, **cached_extras)
                return cached_score

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

                step_params = split_step_params(params)
                method_params = step_params['method']
                current_method = self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params)

                print(method_params)
                
                if not isinstance(current_method, UnsupervisedMethodInterface):
                    raise IncompatibleMethodException('Expected an unsupervised method to be provided')

                preprocessor_params = step_params['preprocessor']
                current_preprocessor = self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params)
                current_raw_preprocessor = self._build_raw_preprocessor(step_params['raw_preprocessor'])

                postprocessor_params = step_params['postprocessor']
                current_postprocessor = self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences, **postprocessor_params)

                thresholder_params = step_params['thresholder']
                current_thresholder = self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences, **thresholder_params)
                try:
                    from pdmlabs.pipeline.mlflow_pipeline import UnsupervisedPdMPipeline
                    pdm_pipeline = UnsupervisedPdMPipeline(
                        raw_preprocessor=current_raw_preprocessor,
                        preprocessor=current_preprocessor,
                        method=current_method,
                        postprocessor=current_postprocessor,
                        thresholder=current_thresholder
                    )
                    pdm_pipeline.fit()

                    for current_target_data, current_target_source in zip(self.target_data, self.target_sources):
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

                        inference_results = pdm_pipeline.predict({
                            'target_data': current_target_data,
                            'source': current_target_source,
                            'event_data': self.event_data
                        })
                        processed_target_scores = inference_results['scores']
                        current_thresholds = inference_results['dynamic_thresholds']

                        # processed_target_scores.extend(processed_target_scores_until_reset)
                        # current_thresholds.extend(current_thresholds_until_reset)

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

                except Exception as e:
                    if self.debug:
                        raise e
                    print(e)
                    print("Assing score 0 and continuing to the next experiment.")
                    self._finish_run(parent_run=parent_run, current_steps={
                        'raw_preprocessor': current_raw_preprocessor,
                        'preprocessor': current_preprocessor,
                        'method': current_method,
                        'postprocessor': current_postprocessor,
                        'thresholder': current_thresholder
                    }, params=params)
                    return 0

                best_metrics_dict = self._run_evaluators(
                    DefaultADEvaluator(debug=self.debug),
                    result_scores=result_scores,
                    result_dates=result_dates,
                    results_isfailure=results_isfailure,
                    plot_dictionary=plot_dictionary,
                    thresholder=current_thresholder
                )

                # Ship this trial back to the parent process. Assigning to self
                # here would be lost: every backend with n_jobs > 1 runs this
                # objective in a worker process. Must stay ahead of _finish_run(),
                # which calls method.destruct() and can delete on-disk state the
                # pipeline depends on.
                trial_sink.record(
                    best_metrics_dict[self.optimization_param],
                    pdm_pipeline,
                    th=best_metrics_dict["threshold_auc"],
                    params=params,
                )

                self._plot_scores(plot_dictionary, best_metrics_dict)

                self._finish_run(parent_run=parent_run, current_steps={
                    'raw_preprocessor': current_raw_preprocessor,
                    'preprocessor': current_preprocessor,
                    'method': current_method,
                    'postprocessor': current_postprocessor,
                    'thresholder': current_thresholder
                }, params=params)

            return best_metrics_dict[self.optimization_param]

        try:
            results = self._run_optimizer(self.param_space, optimization_objective, maximize=self.maximize)
            self._collect_best_trial(results, trial_sink)
        finally:
            trial_sink.cleanup()

        dict_ro_return = {}
        dict_ro_return['best_params'] = results['best_params']
        dict_ro_return["best_objective"] = results["best_objective"]
        dict_ro_return["th"] = self.extra_metrics["th"]
        dict_ro_return["best_pipeline_objective"] = self.extra_metrics["best_pipeline_objective"]
        dict_ro_return["best_pipeline_params"] = self.extra_metrics["best_params_used"]

        if self.log_best_pipeline and self.best_pipeline is not None:
            if self.extra_metrics["best_pipeline_th"] is not None:
                self.best_pipeline.set_global_threshold(self.extra_metrics["best_pipeline_th"])
            try:
                with mlflow.start_run(experiment_id=self.experiment_id, run_name="Best_Pipeline_Model"):
                    mlflow.pyfunc.log_model(artifact_path="best_pdm_pipeline", python_model=self.best_pipeline)
            except Exception as e:
                print(f"Warning: Failed to log MLflow pipeline model: {e}")
                
        return self._finish_experiment(dict_ro_return)