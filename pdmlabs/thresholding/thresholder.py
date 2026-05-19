"""Abstract base interface for thresholders.

Thresholders convert anomaly scores to threshold values for decision-making.
This differs from post-processors which convert scores to binary labels directly.

Thresholders:
- Accept anomaly scores (float values)
- Return threshold value(s) - the boundary between normal and anomalous
- Support both batch and online modes
- Can be adaptive (threshold varies per sample) or static (fixed threshold)

Use cases:
- Fixed threshold: Simple baseline (threshold=0.5, any score > threshold = anomaly)
- Adaptive threshold: Adjusts per time period or based on local statistics
- Survival analysis: Converts survival probabilities to RUL (Remaining Useful Life)
- Context-aware: Different thresholds for different sources/times

Typical pipeline: anomaly_scores -> thresholder -> threshold -> binary_labels
Or directly with post-processors: anomaly_scores -> post_processor -> binary_labels
"""

import abc

import pandas as pd

from pdmlabs.pdm_evaluation_types.types import EventPreferences


class ThresholderInterface(abc.ABC):
    """Abstract base class for threshold determination methods.
    
    Thresholders determine the boundary value(s) between normal and anomalous
    scores. This enables converting continuous anomaly scores to binary decisions.
    
    Two usage patterns:
    1. Single threshold: Apply same threshold to all scores
    2. Adaptive threshold: Different threshold per sample/time
    
    Attributes:
        event_preferences (EventPreferences): Event configuration dict.
    """
    def __init__(self, event_preferences: EventPreferences):
        """Initialize thresholder.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
        """
        self.event_preferences = event_preferences

    @abc.abstractmethod
    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """Fit thresholder on training data (optional for some thresholders).
        
        Some thresholders are stateless (e.g., constant threshold), others learn
        thresholds from training data or anomaly labels.
        
        Args:
            historic_data (list): Training data DataFrames (one per source).
            historic_sources (list[str]): Source identifiers.
            event_data (pd.DataFrame): Event log with 'date', 'type', etc.
            anomaly_ranges (list, optional): Labels marking anomalous time periods.
                Used by supervised thresholders to learn optimal threshold.
        """
        pass

    @abc.abstractmethod
    def infer_threshold(self, scores: list[float], source: str, event_data: pd.DataFrame, scores_dates: list[pd.Timestamp]) -> list[float]:
        """Determine threshold value(s) for batch of scores (offline mode).
        
        Returns threshold value for each score. Can be:
        - Single value repeated: [0.5, 0.5, 0.5, ...]  (static threshold)
        - Varying values: [0.4, 0.45, 0.5, 0.55, ...]  (adaptive threshold)
        
        Args:
            scores (list[float]): Anomaly scores to threshold.
            source (str): Source identifier for source-specific thresholds.
            event_data (pd.DataFrame): Event log for context-aware thresholds.
            scores_dates (list[pd.Timestamp]): Timestamps of scores (enables
                time-based adaptive thresholds).
        
        Returns:
            list[float]: Threshold value(s). Same length as scores.
                Compare: anomaly_detected = (score > threshold)
        """
        pass


    @abc.abstractmethod
    def infer_threshold_one(self, score: float, source: str, event_data: pd.DataFrame) -> float:
        """Determine threshold for single score (online/streaming mode).
        
        Args:
            score (float): Single anomaly score.
            source (str): Source identifier.
            event_data (pd.DataFrame): Event log (unused by most thresholders).
        
        Returns:
            float: Threshold value for this score.
        """
        pass


    @abc.abstractmethod
    def get_params(self):
        """Return thresholder hyperparameters.
        
        Returns:
            dict: Configuration parameters (e.g., {'threshold': 0.5}).
        """
        pass
    
    
    @abc.abstractmethod
    def __str__(self) -> str:
        """Return thresholder name.
        
        Returns:
            str: Human-readable name (e.g., 'ConstantThresholder').
        """
        pass