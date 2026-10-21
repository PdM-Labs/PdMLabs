"""Default (pass-through) raw-level preprocessor.

It leaves the data unchanged. Because a raw-level step must turn each snapshot into
exactly one row, this is only possible when every snapshot already is a single row (e.g.
snapshots that were featurized beforehand). Snapshots with several rows raise an error
asking for a feature extractor.

It is used automatically when a snapshot dataset is run without a raw_preprocessor.
"""

import pandas as pd

from pdmlabs.preprocessing.raw_level.raw_level_pre_processor import RawLevelPreProcessorInterface


class DefaultRawPreProcessor(RawLevelPreProcessorInterface):
    """Pass single-row snapshots through unchanged.

    Examples:
        >>> import pandas as pd
        >>> DefaultRawPreProcessor().extract(pd.DataFrame({'ch0': [0.4], 'ch1': [1.2]}))
        {'ch0': 0.4, 'ch1': 1.2}
    """

    def extract(self, snapshot: pd.DataFrame) -> dict:
        """Return the numeric values of a single-row snapshot unchanged.

        Args:
            snapshot (pd.DataFrame): A snapshot with exactly one row.

        Returns:
            dict: Column name -> value.

        Raises:
            ValueError: If the snapshot does not have exactly one row.
        """
        if snapshot.shape[0] != 1:
            raise ValueError(
                f'DefaultRawPreProcessor needs 1-row snapshots; got {snapshot.shape[0]} rows. '
                'Pass e.g. raw_preprocessor=StatisticalRawPreProcessor to turn each snapshot '
                'into one row of features.'
            )
        row = self.numeric_channels(snapshot).iloc[0]
        return {column: float(value) for column, value in row.items()}

    def get_params(self) -> dict:
        """Return hyperparameters (none for this preprocessor).

        Returns:
            dict: Empty dict {}
        """
        return {}

    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'Default'
        """
        return 'Default'
