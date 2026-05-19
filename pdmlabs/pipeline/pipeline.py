"""PdMPipeline: Core data processing and component orchestration for predictive maintenance.

This module defines the PdMPipeline class, which orchestrates the complete anomaly detection
pipeline for predictive maintenance experiments. A pipeline combines:

- Dataset (with features, events, preferences, evaluation parameters)
- Processing steps: preprocessor → method → postprocessor → thresholder
- Event mapping logic (failures, resets, source relationships)

The pipeline is the contract between raw data and experiment execution. It encapsulates:
- The transformation steps applied to features
- How anomalies are detected (via the method)
- How anomaly scores are refined (postprocessor)
- How scores are converted to binary predictions (thresholder)
- Event semantics (what constitutes a failure, reset, or normal boundary)

Architecture:

    Raw Data (DataFrame)
         ↓
    [Preprocessor] - Normalization, feature engineering, windowing
         ↓
    [Method] - Anomaly detection (returns anomaly scores)
         ↓
    [Postprocessor] - Score refinement (smoothing, aggregation, fusion)
         ↓
    [Thresholder] - Binary decision boundary (0/1 predictions)
         ↓
    Predictions (binary labels)

Key Concepts:

    Events: Timestamps with type, description, and source that define:
      - Failures: when equipment actually fails
      - Resets: when systems restart/reinitialize episodes
      - Sources: which subsystem/sensor an event applies to

    Event Preferences: Rules for how events propagate to target sources:
      - Direct (source → target_source)
      - Broadcast (* applies to all targets)
      - Same-source (= only to same source)

    Caching: Event extraction is cached on first call for performance.
"""

from typing import TypedDict, List

import pandas as pd

from pdmlabs.preprocessing.record_level.default import DefaultPreProcessor
from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
from pdmlabs.postprocessing.default import DefaultPostProcessor
from pdmlabs.thresholding.constant import ConstantThresholder
from pdmlabs.thresholding.thresholder import ThresholderInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences
from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdmlabs.method.method import MethodInterface
from pdmlabs.postprocessing.post_processor import PostProcessorInterface
from pdmlabs.utils.utils import expand_event_preferences


class PdMPipelineSteps(TypedDict):
    """Type definition for the four-stage anomaly detection pipeline.

    Attributes:
        preprocessor (RecordLevelPreProcessorInterface): Transforms raw features
            (e.g., normalization, feature engineering, windowing). Fitted on training
            data, applied to test data.
        method (MethodInterface): Anomaly detection model that returns anomaly scores.
            Can be supervised, semi-supervised, or unsupervised depending on availability
            of labels.
        postprocessor (PostProcessorInterface): Refines anomaly scores (e.g., smoothing,
            source fusion, aggregation). Applied after method predictions.
        thresholder (ThresholderInterface): Converts anomaly scores to binary predictions
            using a decision threshold. Can be fixed, adaptive, or learned.
    """
    preprocessor:  RecordLevelPreProcessorInterface
    method : MethodInterface
    postprocessor : PostProcessorInterface
    thresholder : ThresholderInterface


class PdMPipeline():
    """Orchestrates the complete predictive maintenance anomaly detection pipeline.

    A PdMPipeline combines a dataset with four processing steps (preprocessor, method,
    postprocessor, thresholder) to form a complete anomaly detection system. It also
    manages event mappings (failures, resets) and evaluation parameters.

    The pipeline serves as the contract between raw data and experiment execution:
    - Defines what transformations apply to features
    - Specifies which anomaly detection method to use
    - Caches event extraction results for performance
    - Provides utilities to query failure/reset dates by device source

    Attributes:
        dataset (dict): Complete dataset specification with keys:
            - 'event_data': DataFrame with columns [date, type, description, source]
            - 'event_preferences': Dict mapping 'failure' and 'reset' to event rules
            - 'dates': Target dates/indices (for temporal alignment)
            - 'predictive_horizon': Lead time for detection [int or list]
            - 'slide': Sliding window size for VUS metrics
            - 'lead': Detection lead time threshold
            - 'beta': Weighting parameter for metrics
            - 'historic_data': Training data files/DataFrames
            - 'target_data': Test data files/DataFrames
            - 'historic_sources': Source labels for training data
            - 'target_sources': Source labels for test data
        steps (PdMPipelineSteps): Dictionary with preprocessor, method, postprocessor, thresholder.
        auc_resolution (int): Resolution for threshold sweep in evaluation (e.g., 100).
        experiment_type: Default method class if not provided in steps.
        event_preferences (dict): Parsed event rules for failures/resets.
        event_data (pd.DataFrame): Parsed event log with timestamps and sources.
        preprocessor, method, postprocessor, thresholder: Pipeline component instances.

    Examples:
        >>> from pdmlabs.method.isolation_forest import IsolationForest
        >>> from pdmlabs.preprocessing.no_preprocessor import NoPreprocessor
        >>> from pdmlabs.postprocessing.no_postprocessor import NoPostprocessor
        >>> from pdmlabs.thresholding.static_threshold import StaticThreshold
        >>>
        >>> steps = {
        ...     'preprocessor': NoPreprocessor(event_preferences={...}),
        ...     'method': IsolationForest,
        ...     'postprocessor': NoPostprocessor(event_preferences={...}),
        ...     'thresholder': StaticThreshold(threshold_value=0.5, event_preferences={...})
        ... }
        >>> pipeline = PdMPipeline(
        ...     steps=steps,
        ...     dataset=my_dataset,
        ...     auc_resolution=100
        ... )
        >>> failure_dates = pipeline.extract_failure_dates_for_source('bearing_1')
        >>> reset_dates = pipeline.extract_reset_dates_for_source('bearing_1')
    """
    def __init__(self,
                steps: PdMPipelineSteps,
                dataset: dict,
                auc_resolution : int,
                experiment_type=SemiSupervisedMethodInterface,
    ):
        """Initialize a PdM pipeline with steps, dataset, and evaluation parameters.

        Args:
            steps (PdMPipelineSteps): Dictionary containing preprocessor, method,
                postprocessor, and thresholder instances/classes.
            dataset (dict): Complete dataset specification (see class docstring for required keys).
            auc_resolution (int): Number of thresholds to sample when evaluating metric curves
                (e.g., 100 means test thresholds from 0 to 1 in 0.01 increments).
            experiment_type: Default method interface type if method not in steps.
                Defaults to SemiSupervisedMethodInterface.

        Raises:
            KeyError: If required dataset keys are missing ('event_data', 'event_preferences', etc.).
            TypeError: If steps components don't implement required interfaces.
        """
        self.dataset = dataset
        self.steps = steps
        self.event_data = dataset['event_data']
        self.event_data['date']=pd.to_datetime( self.event_data['date'])
        self.event_preferences = dataset['event_preferences']
        self.target_dates = dataset['dates']
        self.historic_dates = dataset['dates']
        self.predictive_horizon = dataset['predictive_horizon']
        self.slide = dataset['slide']
        self.lead = dataset['lead']
        self.beta = dataset['beta']
        self.auc_resolution = auc_resolution

        self.preprocessor = steps.get('preprocessor', DefaultPreProcessor(event_preferences=self.event_preferences))
        self.method = steps.get('method', experiment_type)
        self.postprocessor = steps.get('postprocessor', DefaultPostProcessor(event_preferences=self.event_preferences))

        self.thresholder = steps.get('thresholder', ConstantThresholder(threshold_value=0.5, event_preferences=self.event_preferences))


    def get_steps(self) -> PdMPipelineSteps:
        """Return the pipeline steps (preprocessor, method, postprocessor, thresholder).

        Returns:
            PdMPipelineSteps: Dictionary containing all four pipeline components.

        Examples:
            >>> pipeline = PdMPipeline(...)
            >>> steps = pipeline.get_steps()
            >>> method = steps['method']
        """
        return self.steps


    def get_step_by_name(self, step_name: str):
        """Get a specific pipeline step by name.

        Args:
            step_name (str): Name of the step to retrieve. Must be one of:
                'preprocessor', 'method', 'postprocessor', 'thresholder'.

        Returns:
            RecordLevelPreProcessorInterface | MethodInterface | PostProcessorInterface | ThresholderInterface:
                The requested pipeline component.

        Raises:
            KeyError: If step_name is not found in the pipeline.

        Examples:
            >>> pipeline = PdMPipeline(...)
            >>> method = pipeline.get_step_by_name('method')
            >>> preprocessor = pipeline.get_step_by_name('preprocessor')
        """
        return self.steps[step_name]

    def extract_failure_dates_for_source(self, source: str) -> list[pd.Timestamp]:
        """Extract all failure timestamps for a specific source (device/subsystem).

        Queries the event log to find all events matching the failure preferences
        that should apply to the given source. Uses caching for performance on
        repeated calls.

        Event preferences define rules like:
        - Direct: specific source fires failure event
        - Broadcast (*): all sources affected when event fires
        - Group: multiple source IDs affected by same event

        Args:
            source (str): Source/device identifier (e.g., 'bearing_1', 'motor_A').

        Returns:
            list[pd.Timestamp]: Sorted list of unique failure timestamps for this source.
                Empty list if no failures defined or source not in preferences.

        Caching:
            Failure dates are cached on first call via expanded_event_preferences.
            Subsequent calls are O(1) lookups. Cache persists across calls to
            extract_reset_dates_for_source.

        Examples:
            >>> pipeline = PdMPipeline(...)
            >>> failure_times = pipeline.extract_failure_dates_for_source('bearing_1')
            >>> print(f"Failures at: {failure_times}")
            Failures at: [Timestamp('2024-01-10 14:30:00'), Timestamp('2024-01-15 09:15:00')]

            >>> # For broadcast events (* applies to all sources):
            >>> failures_motor = pipeline.extract_failure_dates_for_source('motor_A')
            >>> failures_bearing = pipeline.extract_failure_dates_for_source('bearing_1')
            >>> # Both may include times from * events
        """
        result = []
        try:
            expanded_event_preferences = self.expanded_event_preferences
            source_event_dict = self.source_event_dict
            get_affected_failure = self.get_affected_failure
            get_affected_reset = self.get_affected_reset
        except Exception as e:
            self.expanded_event_preferences = expand_event_preferences(event_data=self.event_data,
                                                                       event_preferences=self.event_preferences)
            expanded_event_preferences = self.expanded_event_preferences
            self.source_event_dict = {}
            get_affected_failure = {}
            get_affected_reset = {}
            for row in self.event_data.itertuples():
                if row.source not in self.source_event_dict:
                    self.source_event_dict[row.source] = []
                    get_affected_failure[row.source] = []
                    get_affected_reset[row.source] = []
                self.source_event_dict[row.source].append(row)
            for key in self.source_event_dict:
                self.source_event_dict[key] = pd.DataFrame(self.source_event_dict[key])

            get_affected_failure = self.find_affected_sources(expanded_event_preferences['failure'],
                                                              get_affected_failure)
            get_affected_reset = self.find_affected_sources(expanded_event_preferences['reset'], get_affected_reset)

            self.get_affected_failure = get_affected_failure
            self.get_affected_reset = get_affected_reset
            source_event_dict = self.source_event_dict

        for current_pref_ in get_affected_failure[source]:
            for row_index, row in source_event_dict[current_pref_[0]].iterrows():
                if row['type'] == current_pref_[2] and row['description'] == current_pref_[1]:
                    result.append(row['date'])

        # for current_preference in expanded_event_preferences['failure']:
        #     matched_rows = source_event_dict[current_preference.source].loc[(self.event_data['type'] == current_preference.type) & (self.event_data['description'] == current_preference.description)]
        #     for row_index, row in matched_rows.iterrows():
        #         if current_preference.target_sources == '=' and str(row.source) == str(source):
        #             result.append(row['date'])
        #         elif source in current_preference.target_sources:
        #             result.append(row['date'])
        #         elif current_preference.target_sources == '*':
        #             result.append(row['date'])
        # old_res=self.old_extract_failure_dates_for_source(source)
        # if len(old_res)!=len(result):
        #     raise Exception("mismatch in failure date extraction")
        return sorted(list(set(result)))

    def old_extract_failure_dates_for_source(self, source) -> list[pd.Timestamp]:
        result = []
        expanded_event_preferences = expand_event_preferences(event_data=self.event_data,
                                                              event_preferences=self.event_preferences)
        for current_preference in expanded_event_preferences['failure']:
            matched_rows = self.event_data.loc[(self.event_data['type'] == current_preference.type) & (
                    self.event_data['source'] == current_preference.source) & (self.event_data[
                                                                                   'description'] == current_preference.description)]
            for row_index, row in matched_rows.iterrows():
                if current_preference.target_sources == '=' and str(row.source) == str(source):
                    result.append(row['date'])
                elif source in current_preference.target_sources:
                    result.append(row['date'])
                elif current_preference.target_sources == '*':
                    result.append(row['date'])
        return sorted(list(set(result)))

    def find_affected_sources(self, given_expanded_preferences, get_affected) -> dict[str, List[List[str]]]:
        """Map event preferences to affected sources (internal caching utility).

        Processes expanded event preferences and populates a dictionary indicating
        which sources are affected by each event. Handles three preference types:
        1. Direct (=): event affects only its source of origin
        2. Broadcast (*): event affects all known sources
        3. List: event affects specific named target sources

        This method is called once per pipeline to build a cache used by
        extract_failure_dates_for_source() and extract_reset_dates_for_source().

        Args:
            given_expanded_preferences (list): List of expanded preference objects
                (typically all 'failure' or 'reset' preferences).
            get_affected (dict): Dictionary being populated with structure:
                {source_name: [[event_source, event_description, event_type], ...]}

        Returns:
            dict[str, List[List[str]]]: Same dict with preferences added:
                Maps each source to list of [source, description, type] tuples
                indicating which events apply to that source.

        Examples:
            >>> preferences = [...expanded preferences...]
            >>> affected = {}  # Will be populated
            >>> result = pipeline.find_affected_sources(preferences, affected)
            >>> print(result['bearing_1'])  # Events affecting this source
            [['source_A', 'overheat', 'failure'], ['*', 'shutdown', 'failure']]
        """
        for current_preference in given_expanded_preferences:
            if current_preference.target_sources == '=':
                get_affected[current_preference.source].append(
                    [current_preference.source, current_preference.description, current_preference.type])
            elif current_preference.target_sources == '*':
                for source_key in self.source_event_dict.keys():
                    get_affected[source_key].append(
                        [current_preference.source, current_preference.description, current_preference.type])
            else:
                for target_source in current_preference.target_sources:
                    get_affected[target_source].append(
                        [current_preference.source, current_preference.description, current_preference.type])
        return get_affected

    def extract_reset_dates_for_source(self, source) -> list[pd.Timestamp]:
        """Extract all reset timestamps for a specific source (device/subsystem).

        Queries the event log to find all events matching the reset preferences
        that should apply to the given source. Reset events typically mark:
        - Equipment restarts / reinitializations
        - Episode boundaries (e.g., new mission, device replacement)
        - Scenario transitions for evaluation

        Similar to extract_failure_dates_for_source, but queries 'reset' preferences
        from event_preferences.event_data. Uses the same caching mechanism.

        Args:
            source (str): Source/device identifier (e.g., 'bearing_1', 'motor_A').

        Returns:
            list[pd.Timestamp]: Sorted list of unique reset timestamps for this source.
                Empty list if no resets defined or source not in preferences.

        Caching:
            Resets are cached alongside failures on first call. Repeated calls are O(1).

        Examples:
            >>> pipeline = PdMPipeline(...)
            >>> reset_times = pipeline.extract_reset_dates_for_source('bearing_1')
            >>> print(f"Resets at: {reset_times}")
            Resets at: [Timestamp('2024-01-05 08:00:00'), Timestamp('2024-01-20 16:30:00')]

            >>> # Use to segment data into episodes:
            >>> failures = pipeline.extract_failure_dates_for_source('bearing_1')
            >>> resets = pipeline.extract_reset_dates_for_source('bearing_1')
            >>> # Resets define episode boundaries, failures are ground truth within each episode
        """
        result = []
        try:
            expanded_event_preferences = self.expanded_event_preferences
            source_event_dict = self.source_event_dict
            get_affected_failure = self.get_affected_failure
            get_affected_reset = self.get_affected_reset
        except Exception as e:
            self.expanded_event_preferences = expand_event_preferences(event_data=self.event_data,
                                                                       event_preferences=self.event_preferences)
            expanded_event_preferences = self.expanded_event_preferences
            self.source_event_dict = {}
            get_affected_failure = {}
            get_affected_reset = {}
            for row in self.event_data.itertuples():
                if row.source not in self.source_event_dict:
                    self.source_event_dict[row.source] = []
                    get_affected_failure[row.source] = []
                    get_affected_reset[row.source] = []
                self.source_event_dict[row.source].append(row)
            for key in self.source_event_dict:
                self.source_event_dict[key] = pd.DataFrame(self.source_event_dict[key])

            get_affected_failure = self.find_affected_sources(expanded_event_preferences['failure'],
                                                              get_affected_failure)
            get_affected_reset = self.find_affected_sources(expanded_event_preferences['reset'], get_affected_reset)

            self.get_affected_failure = get_affected_failure
            self.get_affected_reset = get_affected_reset
            source_event_dict = self.source_event_dict

        for current_pref_ in get_affected_reset[source]:
            for row_index, row in source_event_dict[current_pref_[0]].iterrows():
                if row['type'] == current_pref_[2] and row['description'] == current_pref_[1]:
                    result.append(row['date'])

        # for current_preference in expanded_event_preferences['reset']:
        #     matched_rows = source_event_dict[current_preference.source].loc[(self.event_data['type'] == current_preference.type) & (self.event_data['description'] == current_preference.description)]
        #     for row_index, row in matched_rows.iterrows():
        #         if current_preference.target_sources == '=' and str(row.source) == str(source):
        #             result.append(row['date'])
        #         elif source in current_preference.target_sources:
        #             result.append(row['date'])
        #         elif current_preference.target_sources == '*':
        #             result.append(row['date'])

        return sorted(list(set(result)))


    def get_steps_as_str(self):
        """Generate a string representation of the pipeline steps for display/logging.

        Creates a human-readable identifier of the pipeline configuration, useful for:
        - MLflow experiment naming and logging
        - Cache keys in optimization
        - Result comparison and tracking

        The format is:
            preprocessor_<name>_method_<name>_postprocessor_<name>_thresholder_<name>

        Returns:
            str: Concatenated names of all four pipeline components.

        Examples:
            >>> pipeline = PdMPipeline(...)
            >>> config_str = pipeline.get_steps_as_str()
            >>> print(config_str)
            preprocessor_StandardScaler_method_IsolationForest_postprocessor_Smoother_thresholder_StaticThreshold

            >>> # Use in MLflow experiment names:
            >>> exp_name = f"pdm-{pipeline.get_steps_as_str()}"
        """
        return f'preprocessor_{self.steps["preprocessor"]}_method_{self.steps["method"]}_postprocessor_{self.steps["postprocessor"]}_thresholder_{self.steps["thresholder"]}'