"""Moving 2-threshold adaptive thresholding post-processor.

Moving2Thresholder converts anomaly scores to binary labels (0/1) using an
adaptive threshold based on the distribution of historical scores. The threshold
is recalculated for each new score based on recent history:

    threshold = mean(non-anomalies) + factor * std(non-anomalies)

Optionally can exclude previously detected anomalies from threshold statistics.
Implements two-pass thresholding for more robust estimation.

Useful when:
- Baseline (normal operation) scores shift over time
- Want adaptive thresholds that adjust to data changes
- Need binary anomaly labels from continuous scores
"""

import statistics

import numpy as np
import pandas as pd
from operator import itemgetter
import datetime
from tqdm import tqdm
from pdmlabs.postprocessing.post_processor import PostProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class Moving2Thresholder(PostProcessorInterface):
    """Threshold anomaly scores adaptively using moving mean/std approach.
    
    Converts continuous anomaly scores to binary labels (0=normal, 1=anomaly)
    using dynamic thresholds calculated from recent score history.
    Optionally excludes previously flagged anomalies from statistics.
    
    Attributes:
        factor (float): Multiplier for standard deviation in threshold calculation.
            Higher values = higher threshold = fewer anomalies detected.
        history_window (int): Number of historical scores to consider (None=all).
        exclude (bool): If True, exclude previously detected anomalies from
            threshold statistics (more robust to anomaly clusters).
        anomaly_scores_dict (dict): Maintains history of scores per source.
    
    Examples:
        >>> from pdmlabs.postprocessing.Moving2T import Moving2Thresholder
        >>> processor = Moving2Thresholder(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     factor=3.0,
        ...     history_window=100,
        ...     exclude=True
        ... )
        >>> processor.fit([df_train], ['bearing_1'], events_df)
        >>> 
        >>> scores = [0.5, 0.6, 0.55, 1.2, 0.7, 2.5, 0.8]  # Has spike at 2.5
        >>> binary_labels = processor.transform(scores, 'bearing_1', events_df)
        >>> # Result: [0, 0, 0, 0, 0, 1, 0]  (only 2.5 crosses threshold)
    """
    def __init__(self, event_preferences: EventPreferences, factor: float = 3, history_window=None, exclude=False):
        """Initialize Moving2Thresholder.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
            factor (float, optional): Std multiplier for threshold. Defaults to 3.
                Larger values = higher thresholds = fewer detections.
                factor=1: 1-sigma threshold
                factor=2: 2-sigma threshold
                factor=3: 3-sigma threshold (unlikely anomalies)
            history_window (int, optional): Number of historical scores to consider.
                None = use all history. Defaults to None.
            exclude (bool, optional): If True, exclude previously detected anomalies
                from threshold statistics (more robust). Defaults to False.
        """
        super().__init__(event_preferences=event_preferences)
        self.factor = factor
        self.history_window = history_window
        self.exclude = exclude
        self.anomaly_scores_dict = {}

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """No-op fit (thresholds computed on-the-fly during transform).
        
        Args:
            historic_data (list[pd.DataFrame]): Ignored.
            historic_sources (list[str]): Ignored.
            event_data (pd.DataFrame): Ignored.
            anomaly_ranges: Ignored.
        """
        pass

    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame) -> list[float]:
        """Convert scores to binary anomaly labels with adaptive thresholds.
        
        Processes scores sequentially, computing threshold for each based on
        history of all previous scores. Returns 1 if score > threshold, 0 otherwise.
        
        Args:
            scores (list[float]): Anomaly scores to threshold.
            source (str): Source identifier (used to maintain separate histories).
            event_data (pd.DataFrame): Event log (unused).
        
        Returns:
            list[float]: Binary labels (0 or 1) indicating anomaly/normal.
        
        Examples:
            >>> scores = [0.5, 0.6, 0.55, 1.2, 0.7, 2.5, 0.8]
            >>> labels = processor.transform(scores, 'bearing_1', events_df)
            >>> # Returns [0, 0, 0, 0, 0, 1, 0]  (threshold rises as history grows)
        """
        self.anomaly_scores_dict[source] = []
        new_scores = []
        for qi in range(len(scores)):
            sc = scores[qi]
            self.anomaly_scores_dict[source].append(sc)
            if self.exclude:
                succed, th = Moving2Texclude(self.anomaly_scores_dict[source], new_scores, factor=self.factor,
                                      hscaleCount=self.history_window)
            else:
                succed, th = Moving2T(self.anomaly_scores_dict[source], factor=self.factor, hscaleCount=self.history_window)

            if sc > th:
                new_scores.append(1)
            else:
                new_scores.append(0)
        return new_scores

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Threshold single score using adaptive threshold (online mode).
        
        Args:
            score_point (float): Single anomaly score to threshold.
            source (str): Source identifier (used to maintain separate histories).
            is_event (bool): Event flag (unused).
        
        Returns:
            float: 1 if score > threshold, 0 otherwise.
        """
        if source in self.anomaly_scores_dict.keys():
            self.anomaly_scores_dict[source].append(score_point)
        else:
            self.anomaly_scores_dict[source] = [score_point]
        succed, th = Moving2T(self.anomaly_scores_dict[source], factor=self.factor, hscaleCount=self.history_window)

        if score_point > th:
            return 1
        else:
            return 0

    def get_params(self):
        """Return hyperparameters.
        
        Returns:
            dict: {'factor': std multiplier, 'history_window': window size,
                   'exclude': whether to exclude anomalies from stats}
        """
        return {
            'factor': self.factor,
            'history_window': self.history_window,
            'exclude': self.exclude,
        }

    def __str__(self) -> str:
        """Return post-processor name.
        
        Returns:
            str: 'Moving2T'
        """
        return 'Moving2T'

def Moving2Texclude(MAerror, anomalies, factor, hscaleCount=1000):
    """Exclude previously detected anomalies before applying Moving2T threshold.
    
    This helper removes scores that were already flagged as anomalies from the
    statistics calculation, making thresholds more robust to clustered anomalies.
    
    Args:
        MAerror (list[float]): All anomaly scores so far.
        anomalies (list[int/bool]): Binary indicators (0/False=normal, 1/True=anomaly)
            for the first len(anomalies) scores.
        factor (float): Std multiplier for threshold calculation.
        hscaleCount (int, optional): History window size. Defaults to 1000 (full history).
    
    Returns:
        tuple: (is_anomaly_bool, threshold_value)
            - First element indicates if last score is anomaly
            - Second element is the calculated threshold
    
    Examples:
        >>> scores = [0.5, 0.6, 2.0, 0.55, 3.0]  # Indices 2, 4 are anomalies
        >>> anomalies = [0, 0, 1, 0, 1]  # Marking detected anomalies
        >>> is_anom, thresh = Moving2Texclude(scores, anomalies[:-1], factor=2)
        >>> # Calculates threshold using only normal scores (0.5, 0.6, 0.55)
    """
    withoutAnomalies = [error for error, isanomaly in zip(MAerror[:len(anomalies)], anomalies) if isanomaly == False or isanomaly == 0]
    withoutAnomalies.extend(MAerror[len(anomalies):])
    return Moving2T(withoutAnomalies, factor, hscaleCount=hscaleCount)


def Moving2T(MAerror, factor, hscaleCount=1000):
    """Calculate adaptive threshold using two-pass statistical method.
    
    Two-pass approach for robust threshold estimation:
    - Pass 1: Compute mean + factor*std from all scores, flag scores > threshold
    - Pass 2: Compute mean + factor*std from scores < Pass 1 threshold (non-outliers)
    - Return Pass 2 threshold and check if last score exceeds it
    
    This two-pass method makes thresholds resistant to outlier inflation.
    
    Args:
        MAerror (list[float]): All anomaly scores so far.
        factor (float): Std multiplier for threshold calculation.
            factor=1: 1-sigma threshold
            factor=2: 2-sigma threshold
            factor=3: 3-sigma threshold (very unlikely anomalies)
        hscaleCount (int, optional): Number of recent scores to consider.
            None uses all history. Defaults to 1000.
    
    Returns:
        tuple: (is_anomaly_bool, threshold_value)
            - First element indicates if last score exceeds threshold AND we achieved
              stable threshold (not all-NaN or degenerate)
            - Second element is the calculated threshold
    
    Edge cases handled:
        - If only 1 unique score: threshold = that score
        - If pass 2 has no values: return last score as threshold
        - If all scores are outliers: threshold determined from remaining scores
    
    Examples:
        >>> scores = [0.5, 0.6, 0.55, 0.7, 0.8, 2.5, 3.0]
        >>> is_anom, thresh = Moving2T(scores, factor=2.0, hscaleCount=None)
        >>> # Pass 1 threshold based on all scores ~1.2
        >>> # Pass 2 threshold based on normal scores ~0.95
        >>> # Returns whether 3.0 > 0.95 (True) and threshold value
    """
    if hscaleCount is None:
        hscaleCount = len(MAerror)
    
    # Get recent history
    historyerrors_raw = MAerror[max(0, len(MAerror) - hscaleCount):]

    # Edge case: only one score
    if len(historyerrors_raw) == 1:
        return False, historyerrors_raw[-1]
    
    # Remove consecutive duplicates for better std calculation
    historyerrors = [historyerrors_raw[0]]
    for q in historyerrors_raw[1:]:
        if q == historyerrors[-1]:
            continue
        historyerrors.append(q)

    # Edge case: only one unique value
    if len(historyerrors) == 1:
        return False, historyerrors[-1]

    # Pass 1: Calculate threshold from all scores
    th = statistics.mean(historyerrors) + factor * statistics.stdev(historyerrors)
    
    # Pass 2: Refine threshold using only non-outliers (scores < Pass 1 threshold)
    secondpass = [d for d in historyerrors if d < th]
    if len(secondpass) == 0:
        return False, historyerrors[-1]
    
    final_threshold = statistics.mean(secondpass) + factor * statistics.stdev(secondpass)
    
    # Return whether last score exceeds final threshold
    return MAerror[-1] > final_threshold, final_threshold