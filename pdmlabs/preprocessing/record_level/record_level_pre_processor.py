"""Interface definition for record-level (feature-level) preprocessing components.

A preprocessor transforms raw features before anomaly detection. Common operations:
- Normalization/scaling (MinMaxScaler, StandardScaler)
- Feature engineering/aggregation
- Windowing (sliding window transformations)
- Feature selection
- Imputation/handling missing values

Preprocessing can be:
- Stateless: transform() produces same output regardless of fit (e.g., FeatureSelector)
- Stateful: fit() learns statistics from training data, transform() applies them

All preprocessors implement RecordLevelPreProcessorInterface to ensure consistency.
"""

import abc

import pandas as pd

from pdmlabs.pdm_evaluation_types.types import EventPreferences


class RecordLevelPreProcessorInterface(abc.ABC):
    """Abstract base class for record-level (feature) preprocessing.

    Record-level preprocessing operates on features/columns of individual records
    (as opposed to raw_level preprocessing which might handle multi-file combinations).

    Subclasses must implement:
    - fit(): Learn statistics from training data
    - transform(): Apply learned transformation to test data
    - transform_one(): Transform a single sample (for online scenarios)
    - get_params(): Return hyperparameters as dict

    Attributes:
        event_preferences (EventPreferences): Event configuration dict with
            'failure', 'reset', etc. for context-aware preprocessing if needed.

    Examples:
        >>> class MyScaler(RecordLevelPreProcessorInterface):
        ...     def fit(self, historic_data, historic_sources, event_data, anomaly_ranges=None):
        ...         # Learn scaling factors from historic_data
        ...         pass
        ...     def transform(self, target_data, source, event_data):
        ...         # Apply scaling to target_data
        ...         return target_data
        ...     def transform_one(self, new_sample, source, is_event):
        ...         # Transform single row
        ...         return new_sample
        ...     def get_params(self):
        ...         # Return {param_name: value}
        ...         return {}
        ...     def __str__(self):
        ...         return 'MyScaler'
    """
    def __init__(self, event_preferences: EventPreferences):
        """Initialize preprocessor with event preferences.

        Args:
            event_preferences (EventPreferences): Dict with 'failure', 'reset', etc.
                Events that may trigger special preprocessing behavior (e.g., reset
                normalization baseline at reset events).
        """
        self.event_preferences = event_preferences


    @abc.abstractmethod
    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame,anomaly_ranges=None) -> None:
        """Learn preprocessing statistics from training data.

        Called once per experiment fold on all historic (training) data.
        Subclasses use this to compute statistics (e.g., min/max, mean/std)
        that are later applied in transform().

        Args:
            historic_data (list[pd.DataFrame]): List of training DataFrames,
                one per source.
            historic_sources (list[str]): List of source identifiers corresponding
                to historic_data (e.g., ['bearing_1', 'bearing_2']).
            event_data (pd.DataFrame): Complete event log with columns
                ['date', 'type', 'description', 'source']. May be used to
                segment preprocessing (e.g., per-episode statistics).
            anomaly_ranges (optional): Pre-computed anomaly ranges (rarely used).
        """
        pass
        

    @abc.abstractmethod
    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Apply learned preprocessing to test data.

        Called after fit() to transform test/target data using statistics
        learned during fit. Must preserve index and alignment with original data.

        Args:
            target_data (pd.DataFrame): Test data to transform.
            source (str): Source identifier (e.g., 'bearing_1'). Use to select
                per-source statistics if learned separately.
            event_data (pd.DataFrame): Complete event log (same as in fit()).

        Returns:
            pd.DataFrame: Transformed data with same shape and index as target_data.

        Examples:
            >>> preprocessor = MinMaxScaler(event_preferences={...})
            >>> preprocessor.fit([train_df], ['bearing_1'], events_df)
            >>> test_df_scaled = preprocessor.transform(test_df, 'bearing_1', events_df)
        """
        pass


    @abc.abstractmethod
    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Transform a single record (for online/streaming scenarios).

        Applied to individual rows as they arrive (not in batch). Used by
        streaming and online experiments to preprocess samples incrementally.

        Args:
            new_sample (pd.Series): Single row with column names as index.
            source (str): Source identifier for this sample.
            is_event (bool): Whether this sample corresponds to an event timestamp
                (may trigger special handling, e.g., reset normalization).

        Returns:
            pd.Series: Transformed sample with same index as input.

        Note:
            For stateful preprocessors, ensures consistency with batch transform().
            For streaming, typically applies pre-learned statistics (from fit()).
        """
        pass


    @abc.abstractmethod
    def get_params(self):
        """Return hyperparameters and configuration as a dictionary.

        Used for logging to MLflow and for reproducibility. Should return
        a flat dict with string keys and JSON-serializable values.

        Returns:
            dict: Hyperparameters, e.g., {'scale': 'minmax', 'feature_count': 10}.

        Examples:
            >>> scaler = MinMaxScaler(event_preferences={...})
            >>> scaler.fit(...)
            >>> print(scaler.get_params())
            {}
        """
        pass


    @abc.abstractmethod
    def __str__(self) -> str:
        """Return a short name/identifier for this preprocessor.

        Used in pipeline naming, logging, and display. Should be a simple
        string like 'MinMaxScaler', 'FeatureSelector', etc.

        Returns:
            str: Preprocessor name.

        Examples:
            >>> print(MinMaxScaler(event_preferences={...}))
            MinMaxScaler
        """
        pass