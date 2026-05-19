"""Identity/passthrough post-processor (no-op transformation).

DefaultPostProcessor returns scores unchanged. Useful for:
- Baseline comparisons (post-processing disabled)
- Testing if post-processing helps or hurts performance
- Experiments where thresholding happens elsewhere
"""

import pandas as pd

from pdmlabs.postprocessing.post_processor import PostProcessorInterface


class DefaultPostProcessor(PostProcessorInterface):
    """No-op post-processor that returns scores unchanged.

    This is an identity transformation: fit() does nothing, transform() and
    transform_one() return input scores as-is. Useful for experiments that
    compare post-processing vs. no post-processing.

    Examples:
        >>> from pdmlabs.postprocessing.default import DefaultPostProcessor
        >>> processor = DefaultPostProcessor(event_preferences={'failure': [], 'reset': []})\n        >>> scores = [0.1, 0.5, 0.9, 0.3]
        >>> processor.fit([df_train], ['bearing_1'], events_df)
        >>> transformed = processor.transform(scores, 'bearing_1', events_df)
        >>> transformed == scores  # Always True
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

    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        """Return scores unchanged (identity transformation).
        
        Args:
            scores (list[float]): Anomaly scores to process.
            source (str): Source identifier (ignored).
            event_data (pd.DataFrame): Event log (ignored).
        
        Returns:
            list[float]: Same as input scores.
        """
        return scores

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Return single score unchanged.
        
        Args:
            score_point (float): Single anomaly score.
            source (str): Source identifier (ignored).
            is_event (bool): Event flag (ignored).
        
        Returns:
            float: Same as input score.
        """
        return score_point
    

    def get_params(self):
        """Return empty parameter dict (no hyperparameters).
        
        Returns:
            dict: Empty dict {}.
        """
        return {}


    def __str__(self) -> str:
        """Return post-processor name.
        
        Returns:
            str: 'Default'
        """
        return 'Default'