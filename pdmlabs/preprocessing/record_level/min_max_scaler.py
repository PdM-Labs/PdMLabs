"""Min-Max scaling preprocessor for normalizing features to [0, 1] range.

Min-Max scaling transforms features to a fixed range [0, 1] using:
    x_scaled = (x - x_min) / (x_max - x_min)

Useful when:
- Features have different units/scales
- Anomaly detectors (neural nets, distance-based methods) are sensitive to scale
- Want to ensure all features contribute equally

Keeps source-specific scalers (one per device/source) to handle variations
in sensor ranges across different equipment instances.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SKLearnMinMaxScaler

from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class MinMaxScaler(RecordLevelPreProcessorInterface):
    """Scale features to [0, 1] using min-max normalization (per source).

    This preprocessor maintains separate scalers for each source (device/subsystem),
    allowing for different value ranges across equipment. For example, 'bearing_1'
    might have vibration in range [0, 100] while 'bearing_2' has [0, 50].

    Attributes:
        scaler_per_source (dict): Maps source identifier to fitted sklearn MinMaxScaler.
            Populated during fit(), used in transform().

    Examples:
        >>> from pdmlabs.preprocessing.record_level.min_max_scaler import MinMaxScaler
        >>> import pandas as pd
        >>>
        >>> # Training data
        >>> df_train = pd.DataFrame({'vibration': [10, 20, 30], 'temp': [50, 60, 70]})
        >>> df_test = pd.DataFrame({'vibration': [15, 25], 'temp': [55, 65]})
        >>>
        >>> scaler = MinMaxScaler(event_preferences={'failure': [], 'reset': []})
        >>> scaler.fit([df_train], ['bearing_1'], events_df)
        >>> df_test_scaled = scaler.transform(df_test, 'bearing_1', events_df)
        >>> # df_test_scaled now has values in [0, 1]
    """
    def __init__(self, event_preferences: EventPreferences):
        """Initialize MinMaxScaler.

        Args:
            event_preferences (EventPreferences): Event configuration dict.
        """
        super().__init__(event_preferences=event_preferences)
        self.scaler_per_source = {}


    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame,anomaly_ranges=None) -> None:
        """Fit scalers for each source using training data.

        Computes min/max values for each source's features from the training data.
        Creates source-specific SKLearnMinMaxScaler instances.

        Args:
            historic_data (list[pd.DataFrame]): Training DataFrames, one per source.
            historic_sources (list[str]): Source identifiers (e.g., ['bearing_1', 'bearing_2']).
            event_data (pd.DataFrame): Event log (unused by this preprocessor).
            anomaly_ranges: Unused.
        """
        for data, source in zip(historic_data, historic_sources):
            self.scaler_per_source[source] = SKLearnMinMaxScaler().fit(data)
        

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to transform test data to [0, 1].

        Args:
            target_data (pd.DataFrame): Test data to scale.
            source (str): Source identifier (used to select the appropriate scaler).
            event_data (pd.DataFrame): Event log (unused).

        Returns:
            pd.DataFrame: Scaled data in [0, 1] range. Returns original data if
                source not found in scaler_per_source (fallback for generalization).

        Examples:
            >>> df_test_scaled = scaler.transform(df_test, 'bearing_1', events_df)
            >>> print(df_test_scaled.min().min())  # Near 0
            >>> print(df_test_scaled.max().max())  # Near 1
        """
        if source in self.scaler_per_source:
            return pd.DataFrame(self.scaler_per_source[source].transform(target_data), columns=target_data.columns, index=target_data.index)
        
        return target_data 


    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Scale a single sample using fitted scaler.

        Args:
            new_sample (pd.Series): Single row to scale.
            source (str): Source identifier.
            is_event (bool): Whether this is an event row (unused).

        Returns:
            pd.Series: Scaled sample.
        """
        return self.scaler_per_source[source].transform_one(pd.DataFrame([], columns=new_sample.index).append(new_sample, ignore_index=True)).iloc[0]
    

    def get_params(self):
        """Return hyperparameters (none for this preprocessor).

        Returns:
            dict: Empty dict {} (no hyperparameters to configure).
        """
        return {}
    

    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'MinMaxScaler'
        """
        return 'MinMaxScaler'