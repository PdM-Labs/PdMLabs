"""Moving average smoothing post-processor for score noise reduction.

MovingAveragePostProcessor applies a rolling window average to anomaly scores,
smoothening sharp spikes and reducing high-frequency noise. The first window_length
scores are returned unchanged to avoid NaN values.

Useful when:
- Scores are noisy/volatile (want stable detections)
- Transient spikes need filtering
- Want to smooth before thresholding

Note: Creates temporal dependence - each score depends on previous scores.
"""

import pandas as pd

from pdmlabs.postprocessing.post_processor import PostProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class MovingAveragePostProcessor(PostProcessorInterface):
    """Smooth anomaly scores using rolling window mean.
    
    This post-processor reduces score variance by averaging values within a
    fixed-size sliding window. First window_length scores are unchanged (to
    avoid NaN), then each score is replaced by the mean of its window.
    
    Attributes:
        window_length (int): Number of scores in rolling window.
        scores_buffer_per_source (dict): Maintains recent scores per source
            for online/streaming mode.
    
    Examples:
        >>> from pdmlabs.postprocessing.moving_average import MovingAveragePostProcessor
        >>> 
        >>> processor = MovingAveragePostProcessor(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     window_length=5
        ... )
        >>> processor.fit([df_train], ['bearing_1'], events_df)
        >>> 
        >>> scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5]
        >>> smoothed = processor.transform(scores, 'bearing_1', events_df)
        >>> # First 5 scores unchanged, then rolling mean applied
    """
    def __init__(self, event_preferences: EventPreferences, window_length: int):
        """Initialize MovingAveragePostProcessor.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
            window_length (int): Size of rolling window (number of scores to average).
                If <= 0 or >= len(scores), returns scores unchanged.
        """
        super().__init__(event_preferences=event_preferences)
        self.window_length = window_length
        self.scores_buffer_per_source = {}

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """No-op fit (moving average is stateless).
        
        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass

    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        """Apply rolling window average to smooth scores.
        
        Args:
            scores (list[float]): Anomaly scores to smooth.
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).
        
        Returns:
            list[float]: Smoothed scores (same length as input). First
                window_length scores are unchanged; remaining are rolling means.
        
        Examples:
            >>> processor = MovingAveragePostProcessor(event_preferences={...}, window_length=3)
            >>> scores = [1, 2, 10, 3, 4, 5]  # Has spike at position 2
            >>> smoothed = processor.transform(scores, 'sensor_1', events_df)
            >>> # Result: [1, 2, 10, 5, 4, 4]  (first 3 unchanged, then rolling means)
        """
        if self.window_length <= 0 or self.window_length >= len(scores):
            return scores
        # Use first self.window_length scores to avoid NaN values
        result = scores[:self.window_length] + pd.Series(scores).rolling(window=self.window_length).mean().tolist()[self.window_length:]
        assert len(result) == len(scores)
        return result
    

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Apply moving average to single score (online mode).
        
        Maintains a buffer of recent scores per source. Returns the score
        unchanged until window_length scores are buffered, then returns the
        mean of the last window_length scores.
        
        Args:
            score_point (float): Single anomaly score to process.
            source (str): Source identifier (used to maintain separate buffers).
            is_event (bool): Event flag (unused).
        
        Returns:
            float: If buffer < window_length: returns score unchanged.
                   Otherwise: returns mean of last window_length scores.
        """
        if source not in self.scores_buffer_per_source:
            self.scores_buffer_per_source[source] = []
        
        self.scores_buffer_per_source[source].append(score_point)

        if len(self.scores_buffer_per_source[source]) < self.window_length:
            return score_point
        
        # Keep only last window_length scores
        self.scores_buffer_per_source[source] = self.scores_buffer_per_source[source][-self.window_length:]
        assert len(self.scores_buffer_per_source[source]) == self.window_length

        return sum(self.scores_buffer_per_source[source]) / len(self.scores_buffer_per_source[source])
    

    def get_params(self):
        """Return hyperparameters.
        
        Returns:
            dict: {'window_length': window size in scores}
        """
        return {
            'window_length': self.window_length
        }


    def __str__(self) -> str:
        """Return post-processor name.
        
        Returns:
            str: 'Moving_Average'
        """
        return 'Moving_Average'