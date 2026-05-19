"""Feature selection preprocessor for dimensionality reduction.

FeatureSelector filters a DataFrame to keep only selected columns/features.
Useful for:
- Removing noisy/irrelevant features
- Dimensionality reduction
- Domain-specific feature sets
"""

import pandas as pd

from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class FeatureSelector(RecordLevelPreProcessorInterface):
    """Select a subset of features for anomaly detection.

    This preprocessor is stateless and deterministic: given a fixed list of
    feature names, it returns only those columns from input data. Useful for:
    - Domain knowledge: select physically meaningful sensors
    - Dimensionality reduction: drop highly correlated features
    - Noise filtering: remove sensors with too much noise
    - Feature importance: keep top-K features from prior analysis

    Attributes:
        selected_features (list[str]): Column names to keep from input DataFrames.

    Examples:
        >>> from pdmlabs.preprocessing.record_level.feature_selector import FeatureSelector
        >>> import pandas as pd
        >>>
        >>> # Data with 5 features, but we only want 3
        >>> df_train = pd.DataFrame({
        ...     'vibration': [1, 2, 3],
        ...     'temp': [50, 60, 70],
        ...     'pressure': [100, 102, 104],
        ...     'noise1': [0.1, 0.2, 0.3],
        ...     'noise2': [0.05, 0.07, 0.09]
        ... })
        >>> df_test = pd.DataFrame({...})  # Similar structure
        >>>
        >>> selector = FeatureSelector(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     selected_features=['vibration', 'temp', 'pressure']
        ... )
        >>> selector.fit([df_train], ['bearing_1'], events_df)
        >>> df_test_selected = selector.transform(df_test, 'bearing_1', events_df)
        >>> # df_test_selected now has only 3 columns: vibration, temp, pressure
    """
    def __init__(self, event_preferences: EventPreferences, selected_features: list[str]):
        """Initialize FeatureSelector.

        Args:
            event_preferences (EventPreferences): Event configuration dict.
            selected_features (list[str]): Column names to keep. If empty list,
                transform() returns data unchanged (select all).
        """
        super().__init__(event_preferences=event_preferences)
        self.selected_features = selected_features


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame,anomaly_ranges=None) -> None:
        """Fit feature selector (no-op, just placeholder).

        Feature selection is stateless, so fit() does nothing. The selected
        features are fixed at initialization.

        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Select subset of features from target data.

        Args:
            target_data (pd.DataFrame): Test data to filter.
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).

        Returns:
            pd.DataFrame: Subset of target_data with only selected_features columns.
                If selected_features is empty, returns input unchanged (fallback).

        Raises:
            KeyError: If any selected feature not in target_data.columns.

        Examples:
            >>> df_test_selected = selector.transform(df_test, 'bearing_1', events_df)
            >>> print(df_test_selected.columns.tolist())
            ['vibration', 'temp', 'pressure']
            >>> print(df_test_selected.shape)
            (100, 3)  # 100 rows, 3 columns
        """
        if len(self.selected_features)==0:
            return target_data
        return target_data[self.selected_features]


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Select features from a single sample.

        Args:
            new_sample (pd.Series): Single row (Series with feature names as index).
            source (str): Source identifier (unused).
            is_event (bool): Event flag (unused).

        Returns:
            pd.Series: Subset of new_sample with only selected_features.

        Examples:
            >>> new_row = pd.Series({'vibration': 1.5, 'temp': 65, 'pressure': 103, 'noise1': 0.15, 'noise2': 0.06})
            >>> selected = selector.transform_one(new_row, 'bearing_1', False)
            >>> print(selected.tolist())
            [1.5, 65, 103]
        """
        return new_sample[self.selected_features]
    

    def get_params(self):
        """Return hyperparameters.

        Returns:
            dict: {'features': list of selected feature names}

        Examples:
            >>> print(selector.get_params())
            {'features': ['vibration', 'temp', 'pressure']}
        """
        return {
            'features': self.selected_features
        }
    

    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'Feature_Selector'
        """
        return 'Feature_Selector'