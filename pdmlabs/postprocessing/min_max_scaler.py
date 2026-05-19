"""Min-Max scaling post-processor for anomaly score normalization.

MinMaxPostProcessor normalizes scores to [0, 1] range. Fits scaler on the
test data itself (within each transform call), so each batch gets scaled
relative to its own min/max values. This is useful for:
- Normalizing scores to probability-like [0, 1] range
- Comparing scores across different models/sources
- Computing normalized confidence scores

Note: Fits on data each time, so different batches may have different scales.
For consistent scaling across batches, use a pre-fitted scaler.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler as SKLearnMinMaxScaler

from pdmlabs.postprocessing.post_processor import PostProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class MinMaxPostProcessor(PostProcessorInterface):
    """Normalize anomaly scores to [0, 1] using min-max scaling.
    
    This post-processor scales scores so that min score -> 0 and max score -> 1.
    Each call to transform() fits a new scaler on that batch.
    
    Attributes:
        scores_buffer_per_source (dict): Maintains recent scores per source
            for online/streaming mode.
    
    Examples:
        >>> from pdmlabs.postprocessing.min_max_scaler import MinMaxPostProcessor
        >>> processor = MinMaxPostProcessor(event_preferences={'failure': [], 'reset': []})
        >>> processor.fit([df_train], ['bearing_1'], events_df)  # No-op
        >>> 
        >>> scores = [0.5, 1.0, 2.0, 1.5]  # Range [0.5, 2.0]
        >>> normalized = processor.transform(scores, 'bearing_1', events_df)
        >>> # Result: [0.0, 0.333..., 1.0, 0.833...]  (scaled to [0, 1])
    """
    def __init__(self, event_preferences: EventPreferences):
        """Initialize MinMaxPostProcessor.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
        """
        super().__init__(event_preferences=event_preferences)
        self.scores_buffer_per_source = {}

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """No-op fit (scaler is fitted per transform call).
        
        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass

    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        """Scale scores to [0, 1] range based on min/max of this batch.
        
        Fits a new scaler on the provided scores, then transforms them.
        Note: Different batches will have independent scalings.
        
        Args:
            scores (list[float]): Anomaly scores to normalize.
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).
        
        Returns:
            list[float]: Normalized scores in [0, 1] range (same length as input).
        
        Examples:
            >>> scores = [0.5, 1.0, 2.0, 1.5]
            >>> normalized = processor.transform(scores, 'bearing_1', events_df)
            >>> print(normalized)  # [0.0, 0.333..., 1.0, 0.833...]
        """
        scaler = SKLearnMinMaxScaler()
        scaler.fit(np.array(scores).reshape(-1, 1))
        return scaler.transform(np.array(scores).reshape(-1, 1)).ravel().tolist()
    

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Scale single score using accumulated buffer.
        
        Maintains a buffer of recent scores per source. Fits scaler to buffer,
        then normalizes the new score.
        
        Args:
            score_point (float): Single anomaly score to normalize.
            source (str): Source identifier (used to maintain separate buffers).
            is_event (bool): Event flag (unused).
        
        Returns:
            float: Normalized score (scaled relative to buffer min/max).
        
        Note:
            May crash or behave unexpectedly if buffer contains only one
            unique value (range becomes 0).
        """
        if source not in self.scores_buffer_per_source:
            self.scores_buffer_per_source[source] = []
        
        self.scores_buffer_per_source[source].append(score_point)

        scaler = SKLearnMinMaxScaler()
        scaler.fit(np.array(self.scores_buffer_per_source[source]).reshape(-1, 1))

        return scaler.transform(np.array([score_point]).reshape(-1, 1)).ravel().tolist()[0]
    

    def get_params(self):
        """Return hyperparameters (none for this post-processor).
        
        Returns:
            dict: Empty dict {}.
        """
        return {}


    def __str__(self) -> str:
        """Return post-processor name.
        
        Returns:
            str: 'MinMaxScaler'
        """
        return 'MinMaxScaler'