"""Statistical snapshot features: peak-to-peak, RMS and FFT bands per channel.

For every numeric channel ``c`` of a snapshot it computes:

- ``p2p_<c>``: peak-to-peak amplitude, ``max(x) - min(x)``
- ``rms_<c>``: root mean square, ``sqrt(mean(x ** 2))``
- ``fft_<c>_<i>`` for ``i`` in ``0 .. n_bins - 1``: spectral features (see below)

Columns are ordered by kind: every p2p column, then every rms column, then every fft column.

FFT features
------------
The single-sided amplitude spectrum ``2 / N * |fft(x)[0 : N // 2]|`` is split into
``n_bins`` frequency bands of (near) equal width over [0, Nyquist], and ``fft_<c>_<i>``
is the largest amplitude in band ``i``. The bands are fixed by the snapshot length
only, so no information is shared between snapshots.
"""

import numbers

import numpy as np
import pandas as pd
from scipy.fft import fft

from pdmlabs.preprocessing.raw_level.raw_level_pre_processor import RawLevelPreProcessorInterface


class StatisticalRawPreProcessor(RawLevelPreProcessorInterface):
    """Peak-to-peak, RMS and FFT-band features per channel.

    Attributes:
        n_bins (int): Number of FFT bands per channel.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> snapshot = pd.DataFrame({0: np.sin(np.linspace(0, 20 * np.pi, 1024))})
        >>> features = StatisticalRawPreProcessor(n_bins=20).extract(snapshot)
        >>> round(features['p2p_0'], 2), round(features['rms_0'], 2)
        (2.0, 0.71)
    """

    def __init__(self, n_bins: int = 20):
        """Initialize the feature extractor.

        Args:
            n_bins (int): Number of FFT bands per channel, a positive integer (a whole-number
                float such as 10.0 is accepted). Defaults to 20.

        Raises:
            ValueError: If ``n_bins`` is not a positive integer.
        """

        if (not isinstance(n_bins, numbers.Real) or isinstance(n_bins, bool)
                or not float(n_bins).is_integer() or n_bins < 1):
            raise ValueError(f'n_bins must be a positive integer, got {n_bins!r}.')
        self.n_bins = int(n_bins)

    def extract(self, snapshot: pd.DataFrame) -> dict:
        """Compute p2p, rms and fft features for every numeric channel of the snapshot.

        Args:
            snapshot (pd.DataFrame): One snapshot, one column per channel.

        Returns:
            dict: ``p2p_<c>``, ``rms_<c>`` and ``fft_<c>_<i>`` features.

        Raises:
            ValueError: If the snapshot has fewer than ``2 * n_bins`` samples: the spectrum of
                ``N`` samples has ``N // 2`` frequencies, so some bands would be empty and their
                features always 0.
        """
        if len(snapshot) < 2 * self.n_bins:
            raise ValueError(
                f'n_bins={self.n_bins} needs snapshots of at least {2 * self.n_bins} samples (every FFT '
                f'band needs a frequency), got {len(snapshot)}. Use a smaller n_bins, or '
                'DefaultRawPreProcessor for single-row snapshots.'
            )
        channels = self.numeric_channels(snapshot)
        signals = {column: channels[column].to_numpy(dtype=float) for column in channels.columns}

        p2p = {f'p2p_{column}': float(np.ptp(x)) for column, x in signals.items()}
        rms = {f'rms_{column}': float(np.sqrt(np.mean(x ** 2))) for column, x in signals.items()}
        spectral = {}
        for column, x in signals.items():
            for index, value in enumerate(self._band_maxima(x)):
                spectral[f'fft_{column}_{index}'] = float(value)

        return {**p2p, **rms, **spectral}

    def _amplitude_spectrum(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        return 2.0 / n * np.abs(fft(x)[0:n // 2])

    def _band_maxima(self, x: np.ndarray) -> list:
        bands = np.array_split(self._amplitude_spectrum(x), self.n_bins)
        return [band.max() if band.size else 0.0 for band in bands]

    def get_params(self) -> dict:
        """Return hyperparameters.

        Returns:
            dict: {'n_bins': int}
        """
        return {'n_bins': self.n_bins}

    def __str__(self) -> str:
        """Return preprocessor name.

        Returns:
            str: 'Statistical'
        """
        return 'Statistical'
