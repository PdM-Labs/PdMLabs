"""Min-Max scaling preprocessor that fits on test data (data leakage variant).

WARNING: MinMaxScalerCheat is for TESTING ONLY. It intentionally violates the
train/test separation principle by fitting the scaler on the test data during
transform(). This is a form of data leakage that provides unrealistically
optimistic results.

This implementation is primarily for:
- Baseline/upper-bound performance (best possible scenario)
- Debugging and understanding preprocessing impact
- Sanity checking (what performance could we achieve if we knew test data?)

DO NOT USE IN PRODUCTION or for realistic performance estimates.

Compare against MinMaxScaler (correct) to see preprocessing impact.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SKLearnMinMaxScaler

from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class MinMaxScalerCheat(RecordLevelPreProcessorInterface):
    """Min-Max scaling that fits on test data (DATA LEAKAGE - FOR TESTING ONLY).

    This preprocessor fits the scaler on test data during transform(), which is
    cheating and provides unrealistically good results. It's useful for:
    - Measuring best-case performance with perfect normalization
    - Debugging preprocessing pipeline
    - Academic/research comparison to show preprocessing limits

    WARNING: This violates proper machine learning practice. Use only for
    experimental analysis, not for model evaluation.

    Attributes:
        scaler_per_source (dict): Maps source identifier to MinMaxScaler
            fitted on TEST data (cheating).

    Examples:
        >>> # Bad: This is how NOT to do preprocessing
        >>> scaler_cheat = MinMaxScalerCheat(event_preferences={'failure': [], 'reset': []})
        >>> scaler_cheat.fit([df_train], ['bearing_1'], events_df)  # Does nothing
        >>> df_test_cheated = scaler_cheat.transform(df_test, 'bearing_1', events_df)
        >>> # Results will be unrealistically good because scaler was fit on df_test!
    """
    def __init__(self, event_preferences: EventPreferences):
        """Initialize MinMaxScalerCheat.

        Args:
            event_preferences (EventPreferences): Event configuration dict.
        """
        super().__init__(event_preferences=event_preferences)
        self.scaler_per_source = {}


    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """No-op fit (does nothing).

        The scaler is fitted on test data during transform() instead, which is
        why this is cheating.

        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on target data, then scale it (DATA LEAKAGE).

        WARNING: This method violates train/test separation by fitting on the
        test data. Results are unrealistically optimistic.

        Args:
            target_data (pd.DataFrame): Test data (used to fit AND transform).
            source (str): Source identifier.
            event_data (pd.DataFrame): Event log (unused).

        Returns:
            pd.DataFrame: Scaled test data using scaler fitted on that same
                test data (cheating).

        Examples:
            >>> df_test_cheated = scaler_cheat.transform(df_test, 'bearing_1', events_df)
            >>> # Results will have suspiciously perfect scaling
        """
        self.scaler_per_source[source] = SKLearnMinMaxScaler().fit(target_data)

        if source in self.scaler_per_source:
            return pd.DataFrame(self.scaler_per_source[source].transform(target_data), columns=target_data.columns, index=target_data.index)
        
        return target_data 


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Scale a single sample using fitted scaler.

        Args:
            new_sample (pd.Series): Single row to scale.
            source (str): Source identifier.
            is_event (bool): Event flag (unused).

        Returns:
            pd.Series: Scaled sample.

        Note:
            transform_one() implementation has potential issues with the
            deprecated append() method and may not work as intended.
        """
        return self.scaler_per_source[source].transform_one(pd.DataFrame([], columns=new_sample.index).append(new_sample, ignore_index=True)).iloc[0]
    

    def get_params(self):
        """Return hyperparameters (none for this preprocessor).

        Returns:
            dict: Empty dict {} (no hyperparameters).
        """
        return {}
    

    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'MinMaxScaler' (misleading - should probably be 'MinMaxScalerCheat')
        """
        return 'MinMaxScaler'