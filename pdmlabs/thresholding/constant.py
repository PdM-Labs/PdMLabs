"""Constant (fixed) thresholder for simple decision boundaries.

ConstantThresholder returns the same threshold value for all scores.
Simple but effective for many use cases where a fixed decision boundary works.

Useful when:
- Anomaly detection model outputs are well-calibrated (0-1)
- Domain knowledge specifies a fixed decision boundary
- Need a fast baseline thresholder
- Want control over False Positive vs False Negative trade-off

Example: If detector outputs [0.1, 0.3, 0.7, 0.9] with threshold=0.5,
anomaly labels would be [0, 0, 1, 1] (score > threshold = anomaly).
"""

import pandas as pd

from pdmlabs.thresholding.thresholder import ThresholderInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class ConstantThresholder(ThresholderInterface):
    """Apply a constant threshold to all anomaly scores.
    
    Simplest thresholder: returns same fixed threshold value for every score.
    Used when a single decision boundary works across all conditions.
    
    Attributes:
        threshold_value (float): Fixed threshold. Any score > threshold_value
            is classified as anomalous. Default 0.5 (typical for normalized scores).
    
    Examples:
        >>> from pdmlabs.thresholding.constant import ConstantThresholder
        >>> thresholder = ConstantThresholder(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     threshold_value=0.5
        ... )
        >>> thresholder.fit([df_train], ['bearing_1'], events_df)  # No-op
        >>> 
        >>> # Batch mode
        >>> scores = [0.1, 0.3, 0.7, 0.9, 0.5]
        >>> thresholds = thresholder.infer_threshold(scores, 'bearing_1', events_df, dates)
        >>> print(thresholds)  # [0.5, 0.5, 0.5, 0.5, 0.5]
        >>> 
        >>> # Online mode
        >>> threshold = thresholder.infer_threshold_one(0.7, 'bearing_1', events_df)
        >>> print(threshold)  # 0.5
        >>> is_anomaly = 0.7 > threshold  # True
    """
    def __init__(self, event_preferences: EventPreferences, threshold_value: float = 0.5):
        """Initialize ConstantThresholder.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
            threshold_value (float, optional): Fixed threshold value. Any score
                greater than this is anomalous. Defaults to 0.5.
                Typical range depends on anomaly detector output:
                - Normalized scores [0,1]: threshold around 0.3-0.7
                - Reconstruction error: threshold depends on feature scale
                - Distance-based: threshold tuned empirically
        """
        super().__init__(event_preferences=event_preferences)
        self.threshold_value = threshold_value

    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """No-op fit (thresholder is stateless).
        
        Constant thresholder doesn't learn from data. Threshold is fixed at init.
        
        Args:
            historic_data (list): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass

    def infer_threshold(self, scores: list[float], source: str, event_data: pd.DataFrame, scores_dates: list[pd.Timestamp]) -> list[float]:
        """Return constant threshold for each score.
        
        Args:
            scores (list[float]): Anomaly scores (unused).
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).
            scores_dates (list[pd.Timestamp]): Score timestamps (unused).
        
        Returns:
            list[float]: List of constant threshold values repeated for each score.
        
        Examples:
            >>> scores = [0.1, 0.3, 0.7, 0.9]
            >>> thresholds = thresholder.infer_threshold(scores, 'bearing_1', events_df, dates)
            >>> print(thresholds)  # [0.5, 0.5, 0.5, 0.5]
        """
        return [self.threshold_value for i in range(len(scores))]
    

    def infer_threshold_one(self, score: float, source: str, event_data: pd.DataFrame) -> float:
        """Return constant threshold for single score.
        
        Args:
            score (float): Single anomaly score (unused).
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).
        
        Returns:
            float: The constant threshold_value.
        """
        return self.threshold_value


    def get_params(self):
        """Return thresholder parameters.
        
        Returns:
            dict: {'threshold_value': the fixed threshold}
        """
        return {
            'threshold_value': self.threshold_value
        }
    

    def __str__(self) -> str:
        """Return thresholder name.
        
        Returns:
            str: 'ConstantThresholder'
        """
        return 'ConstantThresholder'