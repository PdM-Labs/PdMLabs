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
from pdmlabs.exceptions.exception import IncompatibleMethodException, ShortScenarioLengthException


class IncrementalSemiSupervisedPdMExperiment(PdMExperiment):
    """Incremental semi-supervised anomaly detection with model retraining.

    This experiment flavor implements online/incremental learning scenarios:
    1. Processes data incrementally over time (not all at once)
    2. Optionally retrains method and preprocessor on newly observed data
    3. Maintains model adaptation as new patterns emerge

    Different from semi-supervised:
    - Processes scenarios in temporal segments (not monolithic)
    - Can retrain models as new data arrives
    - Simulates streaming/online scenarios while using batch methods

    Useful for:
    - Detecting concept drift without full retraining
    - Systems where data arrives in batches
    - Simulating online learning behavior
    - Gradual performance monitoring

    Attributes:
        refit_new_method_object (bool): If True, creates fresh method for increments.
            If False, updates same method. Default True.
        refit_new_preprocessor_object (bool): If True, creates fresh preprocessor.
            If False, updates same preprocessor. Default True.

    Raises:
        IncompatibleMethodException: If method does not implement SemiSupervisedMethodInterface.
        ShortScenarioLengthException: If scenario is too short for incremental processing.

    Examples:
        >>> experiment = IncrementalSemiSupervisedPdMExperiment(
        ...     experiment_name='incremental-demo',
        ...     pipeline=pipeline,
        ...     param_space={'method_alpha': [0.1, 1.0]},
        ...     refit_new_method_object=True,
        ...     num_iteration=20
        ... )
        >>> results = experiment.execute()
    """
    def __init__(self, refit_new_method_object: bool = True, refit_new_preprocessor_object: bool = True, *args, **kwargs):
        """Initialize incremental experiment with refit strategy.

        Args:
            refit_new_method_object (bool): Create new method for each increment. Defaults to True.
            refit_new_preprocessor_object (bool): Create new preprocessor for each increment. Defaults to True.
            *args: Positional arguments passed to PdMExperiment.__init__().
            **kwargs: Keyword arguments passed to PdMExperiment.__init__().

        See PdMExperiment.__init__() for full parameter documentation.
        """
        super().__init__(*args, **kwargs)
        self.refit_new_method_object = refit_new_method_object
        self.refit_new_preprocessor_object = refit_new_preprocessor_object

    
    def execute(self) -> dict:
        """Run incremental semi-supervised experiment with model retraining.

        Processes data incrementally, optionally retraining models:

        1. For each parameter combination:
           a. For each target scenario's temporal segments:
              - Fits method (optionally new instance)
              - Predicts anomaly scores
              - Applies postprocessor and thresholder
              - Evaluates current segment
           b. Aggregates across segments
        2. Returns best parameters found

        Returns:
            dict: Result dictionary with best_params, best_objective, and threshold.

        Raises:
            IncompatibleMethodException: If method is not SemiSupervisedMethodInterface.
            ShortScenarioLengthException: If scenario is too short to process incrementally.
        """
        super()._register_experiment()
        conf_dict = {
            'initial_random': self.initial_random,
            'num_iteration': self.num_iteration,
            'constraint': self.constraint_function
            # 'batch_size': self.batch_size, currently commented out because of using only scheduler.parallel, more info on issue #97 on Mango - alternatives include using only scheduler.parallel or letting the user decide depending on his hardware
        }

        @scheduler.parallel(n_jobs=self.n_jobs)
        def optimization_objective(**params: dict):
            # cached_result = self._check_cached_run(params)
            #
            # if cached_result is not None:
            #     return cached_result

            with mlflow.start_run(experiment_id=self.experiment_id) as parent_run:
                result_scores = []
                result_dates = []
                result_thresholds = []
                results_isfailure = []
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

                initial_incremental_window_length = params['initial_incremental_window_length']
                incremental_window_length = params['incremental_window_length']
                incremental_slide = params['incremental_slide']

                print('initial_incremental_window_length', initial_incremental_window_length)
                print('incremental_window_length', incremental_window_length)
                print('incremental_slide', incremental_slide)

                mlflow.log_param('initial_incremental_window_length', initial_incremental_window_length)
                mlflow.log_param('incremental_window_length', incremental_window_length)
                mlflow.log_param('incremental_slide', incremental_slide)

                method_params = {re.sub('method_', '', k): v for k, v in params.items() if 'method' in k}
                print(method_params)
                current_method = self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params)

                if not isinstance(current_method, SemiSupervisedMethodInterface):
                    raise IncompatibleMethodException('Expected a semi-supervised method to be provided')

                preprocessor_params = {re.sub('preprocessor_', '', k): v for k, v in params.items() if 'preprocessor' in k}
                current_preprocessor = self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params)

                postprocessor_params = {re.sub('postprocessor_', '', k): v for k, v in params.items() if 'postprocessor' in k}
                print(postprocessor_params)
                current_postprocessor = self.pipeline.postprocessor(event_preferences=self.pipeline.event_preferences, **postprocessor_params)

                thresholder_params = {re.sub('thresholder_', '', k): v for k, v in params.items() if 'thresholder' in k}
                current_thresholder = self.pipeline.thresholder(event_preferences=self.pipeline.event_preferences, **thresholder_params)
                try:
                    total_fit_time=0
                    total_inference_time=0
                    for current_target_data, current_target_source in zip(self.target_data, self.target_sources):
                        current_failure_dates = self.pipeline.extract_failure_dates_for_source(current_target_source)
                        current_reset_dates = self.pipeline.extract_reset_dates_for_source(current_target_source)

                        current_dates = self.pipeline.target_dates
                        # if the user passed a string take the corresponding column of the target_data as 'dates' for the evaluation
                        if isinstance(current_dates, str):
                            name=current_dates
                            current_dates = pd.to_datetime(current_target_data[current_dates])
                            current_dates=[date for date in current_dates]
                            # also drop the corresponding column from the target_data df
                            current_target_data = current_target_data.drop(name, axis=1)

                        current_target_data.index = current_dates

                        if len(current_reset_dates) != 0:
                            if current_reset_dates[-1] < current_target_data.index[-1]:
                                current_reset_dates.append(current_target_data.index[-1])
                        else:
                            current_reset_dates.append(current_target_data.index[-1])

                        if current_target_data.shape[0] < initial_incremental_window_length + incremental_slide:
                            processed_target_scores = [0 for i in range(current_target_data.shape[0])]
                            # TODO warning
                            #raise ShortScenarioLengthException(f'Scenario with length {current_target_data.shape[0]} is not enough for initial incremental window length: {initial_incremental_window_length} and slide: {incremental_slide}')
                        else:
                            current_window_buffer = pd.DataFrame([], columns=current_target_data.columns)
                            executed_initial_fit = False
                            executed_latest_fit = False

                            current_slide_buffer = pd.DataFrame([], columns=current_target_data.columns)
                            executed_initial_prediction = False
                            current_target_scores = []

                            data_points_left = current_target_data.shape[0]
                            for current_data_point_index, current_data_point in current_target_data.iterrows():
                                data_points_left -= 1
                                if not executed_initial_fit and current_window_buffer.shape[0] < initial_incremental_window_length:
                                    # we still collect the initial window data points
                                    current_window_buffer.loc[current_data_point_index] = current_data_point.tolist()

                                    if current_window_buffer.shape[0] == initial_incremental_window_length:
                                        # we collected all the initial window data points so fit the method for the first time
                                        if self.refit_new_method_object:
                                            current_method.destruct()

                                            current_method = self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params)

                                        if self.refit_new_preprocessor_object:
                                            current_preprocessor = self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params)

                                        start_fit=time.time()
                                        current_preprocessor.fit([current_window_buffer], [current_target_source], self.event_data)
                                        total_fit_time+=time.time()-start_fit

                                        current_window_buffer_preprocessed = current_preprocessor.transform(current_window_buffer, current_target_source, self.event_data)

                                        start_fit=time.time()
                                        current_method.fit([current_window_buffer_preprocessed], [current_target_source], self.event_data)
                                        total_fit_time+=time.time()-start_fit
                                        executed_initial_fit = True

                                    continue

                                if not executed_initial_prediction and current_slide_buffer.shape[0] < incremental_slide:
                                    # we still collect the initial slide data points
                                    current_slide_buffer.loc[current_data_point_index] = current_data_point.tolist()
                                    # also add this data point to the window buffer in order to use it in future fits
                                    current_window_buffer.loc[current_data_point_index] = current_data_point.tolist()

                                    if current_slide_buffer.shape[0] == incremental_slide or data_points_left == 0:
                                        current_slide_buffer = current_preprocessor.transform(current_slide_buffer, current_target_source, self.event_data)
                                        # we collected all the initial slide data points so predict scores for the first time
                                        # also add a prefix of 0s with size initial_incremental_window_length for the initial window points in order for the lists to have the same length for the evaluation
                                        start_inference=time.time()
                                        current_target_scores.extend([0 for i in range(initial_incremental_window_length)] + current_method.predict(current_slide_buffer, current_target_source, self.event_data))
                                        total_inference_time+=time.time()-start_inference
                                        executed_initial_prediction = True
                                        current_slide_buffer.drop(current_slide_buffer.index, inplace=True)

                                    continue

                                if not executed_latest_fit:
                                    if current_window_buffer.shape[0] > incremental_window_length:
                                        # keep the last incremental_window_length data points from the window buffer
                                        current_window_buffer = current_window_buffer[-incremental_window_length:]

                                    if self.refit_new_method_object:
                                        current_method.destruct()

                                        current_method = self.pipeline.method(event_preferences=self.pipeline.event_preferences, **method_params)

                                    if self.refit_new_preprocessor_object:
                                        current_preprocessor = self.pipeline.preprocessor(event_preferences=self.pipeline.event_preferences, **preprocessor_params)
                                    
                                    start_fit=time.time()
                                    current_preprocessor.fit([current_window_buffer], [current_target_source], self.event_data)
                                    total_fit_time+=time.time()-start_fit

                                    current_window_buffer_preprocessed = current_preprocessor.transform(current_window_buffer, current_target_source, self.event_data)

                                    start_fit=time.time()
                                    current_method.fit([current_window_buffer_preprocessed], [current_target_source], self.event_data)
                                    total_fit_time+=time.time()-start_fit

                                    executed_latest_fit = True
                                    # add the current point to the slide in order to predict its score afterwards
                                    current_slide_buffer.loc[current_data_point_index] = current_data_point.tolist()
                                    # also add this data point to the window buffer in order to use it in future fits
                                    current_window_buffer.loc[current_data_point_index] = current_data_point.tolist()

                                    if data_points_left != 0:
                                        continue
                                    else:
                                        start_inference=time.time()
                                        current_target_scores.extend(current_method.predict(current_slide_buffer, current_target_source, self.event_data))
                                        total_inference_time+=time.time()-start_inference
                                        continue

                                if current_slide_buffer.shape[0] < incremental_slide:
                                    # we still collect the next slide data points
                                    current_slide_buffer.loc[current_data_point_index] = current_data_point.tolist()
                                    # also add this data point to the window buffer in order to use it in future fits
                                    current_window_buffer.loc[current_data_point_index] = current_data_point.tolist()

                                    if current_slide_buffer.shape[0] == incremental_slide or data_points_left == 0:
                                        start_inference=time.time()
                                        current_slide_buffer = current_preprocessor.transform(current_slide_buffer, current_target_source, self.event_data)

                                        current_target_scores.extend(current_method.predict(current_slide_buffer, current_target_source, self.event_data))
                                        total_inference_time+=time.time()-start_inference
                                        current_slide_buffer.drop(current_slide_buffer.index, inplace=True)
                                        executed_latest_fit = False

                                    if data_points_left != 0:
                                        continue


                            processed_target_scores = current_postprocessor.transform(current_target_scores, current_target_source, self.event_data)

                        current_thresholds = current_thresholder.infer_threshold(processed_target_scores, current_target_source, self.event_data, current_dates)

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
                    
                    mlflow.log_metric("inference_time", total_inference_time)
                    mlflow.log_metric("fit_time", total_fit_time)

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

                from pdmlabs.pipeline.mlflow_pipeline import SemiSupervisedPdMPipeline
                pdm_pipeline = SemiSupervisedPdMPipeline(
                    preprocessor=current_preprocessor,
                    method=current_method,
                    postprocessor=current_postprocessor,
                    thresholder=current_thresholder
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
