"""Self-tuning score normalization post-processor (adaptive z-score).

SelfTuningPostProcessor normalizes anomaly scores using an adaptive z-score
transformation based on a sliding window of historical scores:
    z = (score - mean) / std_dev

Uses initial window_length scores to estimate mean/std, then applies
normalization to all scores. This adapts the scale to the actual score
distribution.

Useful when:
- Anomaly score ranges vary across different datasets/models
- Want to normalize to a standard normal-like distribution
- Thresholding at 0 or fixed values (e.g., threshold=2.0 for 2-sigma)
"""

import statistics

import numpy as np
import pandas as pd

from pdmlabs.postprocessing.post_processor import PostProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class SelfTuningPostProcessor(PostProcessorInterface):
    """Normalize scores using adaptive z-score (mean and std from window).
    
    Computes mean and standard deviation from the first window_length scores,
    then normalizes all scores: (score - mean) / std. Handles edge case where
    std=0 by returning only (score - mean).
    
    Attributes:
        window_length (int): Number of initial scores to use for computing
            mean and std.
        scores_buffer_per_source (dict): Maintains recent scores per source
            for online/streaming mode.
    
    Examples:
        >>> from pdmlabs.postprocessing.self_tuning import SelfTuningPostProcessor
        >>> processor = SelfTuningPostProcessor(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     window_length=10
        ... )
        >>> processor.fit([df_train], ['bearing_1'], events_df)
        >>> 
        >>> scores = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 3.0]
        >>> normalized = processor.transform(scores, 'bearing_1', events_df)
        >>> # First 10 scores used to compute mean/std, then all normalized
    """
    def __init__(self, event_preferences: EventPreferences, window_length: int):
        """Initialize SelfTuningPostProcessor.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
            window_length (int): Number of initial scores to use for computing
                mean and standard deviation. If 0, returns scores unchanged.
        """
        super().__init__(event_preferences=event_preferences)
        self.window_length = window_length
        self.scores_buffer_per_source = {}

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """No-op fit (normalization is computed from score window).
        
        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass

    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        """Normalize scores using z-score from initial window.
        
        Computes mean and std from first window_length scores (removing duplicates).
        Then normalizes all scores: (score - mean) / std. If std=0, returns
        (score - mean) instead.
        
        Args:
            scores (list[float]): Anomaly scores to normalize.
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).
        
        Returns:
            list[float]: Normalized scores (same length as input).
        
        Examples:
            >>> scores = [1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 3.0]  # Mean ~1.2, some outliers
            >>> normalized = processor.transform(scores, 'bearing_1', events_df)
            >>> # Normalized so mean of first 5 = 0, std = 1
        """
        if self.window_length == 0:
            return scores

        # Extract first window_length scores
        scores_for_calculating_metrics_init = scores[:self.window_length]
        
        # Remove consecutive duplicates for more reliable std computation
        scores_for_calculating_metrics = []
        for sc in scores_for_calculating_metrics_init:
            if len(scores_for_calculating_metrics) == 0:
                scores_for_calculating_metrics.append(sc)
            elif sc == scores_for_calculating_metrics[-1]:
                continue  # Skip duplicate
            else:
                scores_for_calculating_metrics.append(sc)
        
        # Compute mean and std, with fallback for edge cases
        if len(scores_for_calculating_metrics) > 1:
            mean, std = statistics.mean(scores_for_calculating_metrics), np.std(scores_for_calculating_metrics)
        else:
            mean = statistics.mean(scores_for_calculating_metrics)
            std = 0
            
        # Normalize all scores
        if std == 0.0:
            return [sc - mean for sc in scores]

        return list(map(lambda score: (score - mean) / std, scores))
    

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Normalize single score using buffered window (online mode).
        
        Maintains a buffer of the first window_length scores. Once buffer is full,
        computes mean/std from buffer and normalizes the incoming score.
        
        Args:
            score_point (float): Single anomaly score to normalize.
            source (str): Source identifier (used to maintain separate buffers).
            is_event (bool): Event flag (unused).
        
        Returns:
            float: If buffer < window_length: returns score unchanged.
                   Otherwise: returns normalized score using buffered mean/std.
        """
        if self.window_length == 0:
            return score_point

        if source not in self.scores_buffer_per_source:
            self.scores_buffer_per_source[source] = []

        self.scores_buffer_per_source[source].append(score_point)

        if len(self.scores_buffer_per_source[source]) < self.window_length:
            return score_point
        else:
            # Keep only first window_length scores
            self.scores_buffer_per_source[source] = self.scores_buffer_per_source[source][:self.window_length]
            assert len(self.scores_buffer_per_source[source]) == self.window_length

            mean, std = statistics.mean(self.scores_buffer_per_source[source]), statistics.stdev(self.scores_buffer_per_source[source])
            return (score_point - mean) / std
    

    def get_params(self):
        """Return hyperparameters.
        
        Returns:
            dict: {'window_length': number of scores to use for computing mean/std}
        """
        return {
            'window_length': self.window_length
        }


    def __str__(self) -> str:
        """Return post-processor name.
        
        Returns:
            str: 'Self_Tuning'
        """
        return 'Self_Tuning'