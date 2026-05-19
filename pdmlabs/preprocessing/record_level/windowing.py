"""Sliding window preprocessing for transforming univariate time series into multivariate windows.

Windowing converts a 1D time series into sequences/windows, useful for:
- Neural network inputs (LSTM, CNN expect sequences)
- Learning temporal patterns
- Converting univariate → multivariate representations

Example: Time series [1, 2, 3, 4, 5] with window_size=3 becomes:
    [[1, 2, 3],
     [2, 3, 4],
     [3, 4, 5]]

Each new row is a sliding window of past values (horizon = window_size).
"""

from pdmlabs.utils import utils
import pandas as pd
from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences



class Windowing(RecordLevelPreProcessorInterface):
    """Convert univariate time series into sliding windows for sequence-based methods.

    This preprocessor transforms a single column (univariate) into a multivariate
    representation using sliding windows. Useful for neural network-based anomaly
    detectors (LSTM, CNN, etc.) that expect sequence inputs.

    Attributes:
        col_pos (int): Column position (0-based index) of the univariate series
            to window. Other columns are kept as-is.
        slidingWindow (int or None): Window size (number of past timesteps to include).
            If None, automatically determines optimal window size.
            If < 2, auto-determines via utils.find_length().

    Examples:
        >>> from pdmlabs.preprocessing.record_level.windowing import Windowing
        >>> import pandas as pd
        >>>
        >>> # Univariate data (single sensor)
        >>> df_train = pd.DataFrame({'sensor': [1.0, 1.1, 1.2, 1.3]})
        >>> df_test = pd.DataFrame({'sensor': [1.4, 1.5, 1.6]})
        >>>
        >>> windower = Windowing(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     slidingWindow=3,
        ...     col_pos=0
        ... )
        >>> windower.fit([df_train], ['bearing_1'], events_df)
        >>> df_test_windowed = windower.transform(df_test, 'bearing_1', events_df)
        >>> # df_test_windowed now has columns s_0, s_1, s_2 (windows of size 3)
    """
    def __init__(self, event_preferences: EventPreferences,slidingWindow=None,col_pos=0):
        """Initialize Windowing preprocessor.

        Args:
            event_preferences (EventPreferences): Event configuration dict.
            slidingWindow (int or None): Desired window size. If None or < 2,
                automatically determines optimal size. Defaults to None.
            col_pos (int): Column index of univariate series to window (0-based).
                Defaults to 0 (first column).
        """
        super().__init__(event_preferences=event_preferences)

        self.col_pos=col_pos
        self.slidingWindow=slidingWindow

    def _sequencing_Univariate_data(self, df):
        """Convert univariate time series column into sliding windows.

        Internal method that performs the windowing transformation.

        Args:
            df (pd.DataFrame): Input DataFrame with at least (col_pos + 1) columns.

        Returns:
            pd.DataFrame: Transformed DataFrame with:
                - New columns s_0, s_1, ..., s_{window_size-1} for windowed values
                - Other original columns preserved
                - Rows padded with first window to match original length
        """
        data = df[df.columns[self.col_pos]].values
        if self.slidingWindow is None:
            return df
        elif self.slidingWindow < 2:
            slidingWindow = utils.find_length(data)
        else:
            slidingWindow = self.slidingWindow

        X_data = utils.Window(window=slidingWindow).convert(data).to_numpy()

        new_df = {}
        for col in range(X_data.shape[1]):
            new_df[f"s_{col}"] = X_data[:, col]
        for col in df.columns:
            if col != df.columns[self.col_pos]:
                new_df[col] = df[col].values

        new_df = pd.DataFrame(new_df)

        row = new_df.iloc[0]
        repeat_times = df.shape[0] - new_df.shape[0]
        repeated_rows = pd.concat([row.to_frame().transpose()] * repeat_times, ignore_index=True)
        new_df = pd.concat([repeated_rows, new_df], ignore_index=True)
        new_df.index = df.index
        return new_df



    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame,anomaly_ranges=None) -> None:
        """Fit windowing (no-op, just placeholder).

        Windowing is stateless, so fit() does nothing.

        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Apply sliding window transformation to univariate series.

        Args:
            target_data (pd.DataFrame): Test data to window.
            source (str): Source identifier (unused by this preprocessor).
            event_data (pd.DataFrame): Event log (unused).

        Returns:
            pd.DataFrame: Windowed data with columns s_0, s_1, ..., plus other columns.

        Examples:
            >>> df_windowed = windower.transform(df_test, 'bearing_1', events_df)
            >>> print(df_windowed.columns)
            Index(['s_0', 's_1', 's_2', ...], dtype='object')
        """
        return self._sequencing_Univariate_data(target_data)

    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Transform a single sample (not fully implemented for streaming).

        Args:
            new_sample (pd.Series): Single row.
            source (str): Source identifier.
            is_event (bool): Event flag.

        Returns:
            pd.Series or None: Placeholder (streaming windowing not yet implemented).
        """
        pass

    def get_params(self):
        """Return hyperparameters.

        Returns:
            dict: {'col_pos': column index, 'slidingWindow': window size or None}
        """
        return {"col_pos":self.col_pos,
                 "slidingWindow":self.slidingWindow}

    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'Windowing'
        """
        return 'Windowing'