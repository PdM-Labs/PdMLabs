"""Time-series aggregation preprocessor for temporal downsampling.

MeanAggregator performs time-window downsampling by computing the average
(mean) value within fixed time periods. Useful for:
- Reducing sampling frequency / data volume
- Smoothing high-frequency noise
- Aligning data to a fixed temporal grid
"""

import pandas as pd

from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class MeanAggregator(RecordLevelPreProcessorInterface):
    """Downsample time-series data by computing mean over fixed time windows.

    This preprocessor aggregates time-indexed data into larger time buckets,
    computing the mean value for each feature in each bucket. It reduces data
    volume and can smooth high-frequency noise.

    Attributes:
        period (str): Pandas resampling frequency string (e.g. '10T' = 10 minutes,
            '1H' = 1 hour, '1D' = 1 day). See pandas resample documentation for
            all valid frequencies.

    Examples:
        >>> from pdmlabs.preprocessing.record_level.aggregator import MeanAggregator
        >>> import pandas as pd
        >>>
        >>> # Minute-level data with 3 sensors
        >>> times = pd.date_range('2024-01-01', periods=600, freq='1T')
        >>> df_train = pd.DataFrame({
        ...     'vibration': range(600),
        ...     'temperature': [20 + i * 0.01 for i in range(600)],
        ...     'pressure': [100 + i * 0.005 for i in range(600)]
        ... }, index=times)
        >>>
        >>> aggregator = MeanAggregator(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     period='10T'  # Aggregate to 10-minute intervals
        ... )
        >>> aggregator.fit([df_train], ['sensor_array_1'], events_df)
        >>> df_test_agg = aggregator.transform(df_test, 'sensor_array_1', events_df)
        >>> # df_test_agg now has one row per 10-minute window with averaged values
    """

    def __init__(self, event_preferences: EventPreferences, period: str = '10T'):
        """Initialize MeanAggregator.

        Args:
            event_preferences (EventPreferences): Event configuration dict.
            period (str, optional): Pandas resampling frequency. Defaults to '10T'
                (10 minutes). Examples:
                - '5T': 5 minutes
                - '1H': 1 hour
                - '6H': 6 hours
                - '1D': 1 day
        """
        super().__init__(event_preferences=event_preferences)
        self.period = period

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """Fit aggregator (no-op, just placeholder).

        Mean aggregation is stateless, so fit() does nothing. The aggregation
        period is fixed at initialization.

        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate time-series data by computing mean over time windows.

        Args:
            target_data (pd.DataFrame): Time-indexed DataFrame to aggregate.
                Must have a DatetimeIndex.
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).

        Returns:
            pd.DataFrame: Aggregated data with one row per time period. Any
                rows that become all-NaN after aggregation are dropped.

        Examples:
            >>> # Original: 600 rows (1 minute intervals)
            >>> df_test_agg = aggregator.transform(df_test, 'sensor_array_1', events_df)
            >>> # Result: ~60 rows (1 row per 10-minute window)
            >>> print(df_test_agg.shape)
            (61, 3)
        """
        df = target_data.resample(self.period).mean()
        df.dropna(inplace=True)
        return df

    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Aggregation is not supported for single samples.

        Args:
            new_sample (pd.Series): Single row (unused).
            source (str): Source identifier (unused).
            is_event (bool): Event flag (unused).

        Returns:
            None: Single-sample aggregation is not supported.

        Raises:
            NotImplementedError: Implicitly (returns None).
        """
        pass

    def get_params(self):
        """Return hyperparameters.

        Returns:
            dict: {'period': resampling frequency string}

        Examples:
            >>> print(aggregator.get_params())
            {'period': '10T'}
        """
        return {
            'period': self.period
        }

    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'MeanAggregator'
        """
        return 'MeanAggregator'