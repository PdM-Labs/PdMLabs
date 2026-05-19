"""Abstract base class interface for post-processors.

Post-processors transform anomaly detection scores to improve decision quality.
They operate on MODEL OUTPUT (anomaly scores), not raw sensor data:

- Smoothing: Reduce score variance (moving average, self-tuning normalization)
- Thresholding: Convert scores to binary anomaly labels or adaptive thresholds
- Normalization: Scale scores to [0, 1] for consistent interpretation
- Filtering: Remove noise or refine predictions across time windows

Typical pipeline: RAW DATA -> (Preprocessing) -> SENSOR FEATURES -> (Detection Model) -> 
ANOMALY SCORES -> (PostProcessing) -> FINAL ANOMALY LABELS/CONFIDENCE
"""

import abc

import pandas as pd

from pdmlabs.pdm_evaluation_types.types import EventPreferences


class PostProcessorInterface(abc.ABC):
    """Abstract base class for anomaly score post-processors.
    
    Post-processors operate on model outputs (anomaly scores) to improve:
    - Score quality (smoothing, normalization)
    - Interpretability (thresholding to binary labels)
    - Robustness (adaptive thresholds based on history)
    
    Each post-processor must implement fit/transform in two modes:
    - Batch mode: transform() processes many scores at once
    - Online/streaming mode: transform_one() processes one score at a time
    
    Attributes:
        event_preferences (EventPreferences): Event configuration dict
    """
    def __init__(self, event_preferences: EventPreferences):
        """Initialize post-processor.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
        """
        self.event_preferences = event_preferences

    @abc.abstractmethod
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """Fit post-processor on training data (anomaly scores or raw data).
        
        Some post-processors are stateless and fit() does nothing. Others compute
        statistics from train data to calibrate thresholds or normalization.
        
        Args:
            historic_data (list[pd.DataFrame]): Training data, one per source.
            historic_sources (list[str]): Source identifiers.
            event_data (pd.DataFrame): Event log with failure/reset events.
            anomaly_ranges: Optional data structure marking normal/anomalous regions.
        """
        pass

    @abc.abstractmethod
    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        """Transform batch of anomaly scores (offline/batch mode).
        
        Args:
            scores (list[float]): Anomaly scores to post-process.
            source (str): Source identifier (e.g., 'bearing_1').
            event_data (pd.DataFrame): Event log (unused by most post-processors).
        
        Returns:
            list[float]: Post-processed scores (same length as input).
        """
        pass


    @abc.abstractmethod
    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Transform single anomaly score (online/streaming mode).
        
        Used when processing one score at a time (e.g., real-time anomaly detection).
        Maintains internal buffer for context-aware transformations (moving average, etc).
        
        Args:
            score_point (float): Single anomaly score to post-process.
            source (str): Source identifier (used to maintain per-source state).
            is_event (bool): Whether this score is from an event sample.
        
        Returns:
            float: Post-processed score.
        """
        pass


    @abc.abstractmethod
    def get_params(self):
        """Return hyperparameters.
        
        Returns:
            dict: Hyperparameter names and values (e.g., {'window_length': 5}).
        """
        pass
    

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return post-processor name.
        
        Returns:
            str: Human-readable name (e.g., 'Moving_Average').
        """
        pass