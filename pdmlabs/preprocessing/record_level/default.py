"""Identity/passthrough preprocessor that applies no transformations.

DefaultPreProcessor is useful for:
- Baseline comparisons (no preprocessing)
- When raw features are already in good format
- Testing if preprocessing helps or hurts performance
"""

import pandas as pd

from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface


class DefaultPreProcessor(RecordLevelPreProcessorInterface):
    """No-op preprocessor that returns data unchanged.

    This is an identity transformation: fit() does nothing, transform() returns
    input data as-is. Useful for experiments that compare preprocessing vs. no
    preprocessing, or for pipelines where feature engineering happens elsewhere.

    Examples:
        >>> from pdmlabs.preprocessing.record_level.default import DefaultPreProcessor
        >>> preprocessor = DefaultPreProcessor(event_preferences={'failure': [], 'reset': []})
        >>> preprocessor.fit([df_train], ['bearing_1'], events_df)
        >>> df_test_transformed = preprocessor.transform(df_test, 'bearing_1', events_df)
        >>> df_test_transformed.equals(df_test)  # Always True
        True
    """
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """No-op fit (does nothing).

        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Return input unchanged.

        Args:
            target_data (pd.DataFrame): Data to transform.
            source (str): Source identifier (ignored).
            event_data (pd.DataFrame): Event log (ignored).

        Returns:
            pd.DataFrame: Same as target_data (identity transformation).
        """
        return target_data


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Return single sample unchanged.

        Args:
            new_sample (pd.Series): Sample to transform.
            source (str): Source identifier (ignored).
            is_event (bool): Event flag (ignored).

        Returns:
            pd.Series: Same as new_sample.
        """
        return new_sample
    

    def get_params(self):
        """Return empty parameter dict (no hyperparameters).

        Returns:
            dict: Empty dict {}.
        """
        return {}
    
    
    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'Default'
        """
        return 'Default'