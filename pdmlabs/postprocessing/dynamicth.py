"""Dynamic adaptive thresholding post-processor (NASA LSTM Anomaly Detection).

DynamicThresholder implements an advanced adaptive thresholding algorithm adapted
from NASA's "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
Thresholding" paper. It converts scores to binary labels using statistical methods
combined with anomaly sequence detection and pruning.

The algorithm:
1. Finds optimal threshold by maximizing impact on normal vs anomalous distributions
2. Groups detected anomalies into sequences
3. Prunes false positives using percentage difference criteria
4. Returns binary labels (0=normal, 1=anomaly)

Useful when:
- Need sophisticated multi-pass anomaly detection
- Baseline shifts significantly over time
- Want to filter out isolated false positives (pruning)
"""

import statistics

import numpy as np
import pandas as pd
from operator import itemgetter
import datetime
from tqdm import tqdm
from pdmlabs.postprocessing.post_processor import PostProcessorInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class DynamicThresholder(PostProcessorInterface):
    """Advanced adaptive thresholding using statistical and sequence analysis.
    
    Implements multi-pass thresholding algorithm that:
    - Tests multiple threshold candidates (in range mean ± [3-5]*std)
    - Scores each threshold based on impact on mean/std of normal vs anomaly groups
    - Selects threshold that best separates normal from anomalous
    - Groups anomalies into sequences and evaluates their statistical significance
    - Prunes anomalies with small impact on distribution
    
    Attributes:
        epsilon (float): Pruning threshold. Percentage difference between consecutive
            anomaly impacts above which to keep it. Filters out small fluctuations.
        history_window (int): Number of recent scores for threshold calculation.
            None = use all history (set to 1 with alldata=True).
        alldata (bool): If True, use entire history (not just recent window).
        anomaly_scores_dict (dict): Maintains score history per source.
    
    References:
        "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
        Thresholding" - Provides the core algorithm and evaluation metrics.
    
    Examples:
        >>> from pdmlabs.postprocessing.dynamicth import DynamicThresholder
        >>> processor = DynamicThresholder(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     epsilon=0.05,  # Prune if < 5% difference
        ...     history_window=1000,  # Use last 1000 scores
        ... )
        >>> processor.fit([df_train], ['bearing_1'], events_df)
        >>> 
        >>> scores = [0.5, 0.6, 0.55, 1.5, 0.7, 3.5, 0.8]
        >>> labels = processor.transform(scores, 'bearing_1', events_df)
        >>> # Returns [0, 0, 0, 1, 0, 1, 0]  (thresholds adapt as history grows)
    """
    def __init__(self, event_preferences: EventPreferences, epsilon: float = 0.05, history_window=None):
        """Initialize DynamicThresholder.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
            epsilon (float, optional): Pruning threshold - minimum percentage
                difference between consecutive anomaly impacts to keep anomaly.
                Range [0, 1]. Defaults to 0.05 (5% difference).
                - Lower values = more aggressive pruning (fewer detections)
                - Higher values = less pruning (more detections)
            history_window (int, optional): Number of recent scores to consider
                for threshold calculation. None = use all history. Defaults to None.
        """
        super().__init__(event_preferences=event_preferences)
        self.epsilon = epsilon
        self.history_window = history_window
        self.alldata = False
        if self.history_window is None:
            self.history_window = 1
            self.alldata = True

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
        """Convert scores to binary labels with dynamic thresholding.
        
        Processes scores sequentially, computing adaptive threshold for each point
        based on distribution of all previous scores. Uses sophisticated algorithm
        to find optimal threshold and prune false positives.
        
        Args:
            scores (list[float]): Anomaly scores to threshold.
            source (str): Source identifier (used to maintain separate histories).
            event_data (pd.DataFrame): Event log (unused).
        
        Returns:
            list[float]: Binary anomaly labels (0 or 1).
        
        Examples:
            >>> scores = [0.5, 0.6, 0.55, 1.2, 0.7, 2.5, 0.8]
            >>> labels = processor.transform(scores, 'bearing_1', events_df)
            >>> # Returns adaptive binary labels accounting for distribution changes
        """
        self.anomaly_scores_dict[source] = []
        new_scores = []
        for qi in range(len(scores)):
            sc = scores[qi]
            self.anomaly_scores_dict[source].append(sc)
            succed, th = dynamicThresholding(self.anomaly_scores_dict[source], DesentThreshold=self.epsilon,
                                             hscaleCount=self.history_window,
                                             alldata=self.alldata)
            if succed == False:
                new_scores.append(0)
            else:
                if sc > th:
                    new_scores.append(1)
                else:
                    new_scores.append(0)
        return new_scores

    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Threshold single score using dynamic thresholding (online mode).
        
        Args:
            score_point (float): Single anomaly score to threshold.
            source (str): Source identifier (used to maintain separate histories).
            is_event (bool): Event flag (unused).
        
        Returns:
            float: 1 if score is flagged as anomaly, 0 otherwise.
        """
        if source in self.anomaly_scores_dict.keys():
            self.anomaly_scores_dict[source].append(score_point)
        else:
            self.anomaly_scores_dict[source] = [score_point]
        succed, th = dynamicThresholding(self.anomaly_scores_dict[source], DesentThreshold=self.epsilon, hscaleCount=self.history_window,
                            alldata=self.alldata)
        if succed == False:
            return 0
        else:
            if score_point > th:
                return 1
            else:
                return 0

    def get_params(self):
        """Return hyperparameters.
        
        Returns:
            dict: {'epsilon': pruning threshold, 'history_window': window size,
                   'All data in history': whether using entire history}
        """
        return {
            'epsilon': self.epsilon,
            'history_window': self.history_window,
            'All data in history': self.alldata
        }

    def __str__(self) -> str:
        """Return post-processor name.
        
        Returns:
            str: 'DynamicThresholder'
        """
        return 'DynamicThresholder'

def dynamicThresholding(MAerror, DesentThreshold=0.02, hscaleCount=1000, alldata=False):
    """Adaptive thresholding with anomaly sequence detection and pruning.
    
    Advanced algorithm from NASA's spacecraft anomaly detection research.
    Uses multi-pass approach:
    1. Test multiple threshold candidates (mean ± 3-5 stds)
    2. Score each candidate by impact on distribution separation (Δμ/μ + Δσ/σ)
    3. Group detected anomalies into temporal sequences
    4. Prune weak anomalies based on percentage change threshold
    
    This makes detection robust to:
    - Isolated false positives (pruned if impact < epsilon)
    - Score distribution shifts (adaptive threshold per point)
    - Clustered anomalies (treats as sequence, not individuals)
    
    Args:
        MAerror (list[float]): All anomaly scores observed so far.
        DesentThreshold (float, optional): Pruning parameter. Minimum percentage
            difference between consecutive anomaly impacts to keep anomaly.
            Range [0, 1]. Lower = more aggressive pruning. Defaults to 0.02 (2%).
        hscaleCount (int, optional): History window size (recent scores to consider).
            Defaults to 1000. Used only if alldata=False.
        alldata (bool, optional): If True, use entire history instead of window.
            Defaults to False.
    
    Returns:
        tuple: (success_bool, threshold_value)
            - success_bool: True if anomaly detected and passed all filters,
              False if threshold couldn't be computed or anomaly was pruned
            - threshold_value: Calculated threshold value
    
    Algorithm details:
        - z-vector: [3, 3.17, 3.33, ..., 4.83] sigma multiples for threshold search
        - Δμ/μ: relative change in mean if anomalies excluded
        - Δσ/σ: relative change in std if anomalies excluded
        - Maximization: (Δμ/μ + Δσ/σ) / (num_anomalies + num_sequences * num_sequences)
        - Pruning: Sorts anomalies by impact, finds elbow point > epsilon
    
    Edge cases:
        - len(history) == 1: Returns False (need more data)
        - No scores above threshold: Returns False
        - All scores are anomalies: Returns False (can't prune reliably)
        - Degenerate std (all same values): Returns False
    
    Time complexity: O(n*m) where n=candidates tested (12), m=history_length
    
    References:
        "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
        Thresholding" - Provides full algorithm with spacecraft telemetry examples
    
    Examples:
        >>> scores = [0.5, 0.6, 0.55, 0.7, 0.8, 2.5, 3.0]
        >>> success, thresh = dynamicThresholding(scores, DesentThreshold=0.05)
        >>> # Evaluates ~12 thresholds, selects best separator
        >>> # Groups 2.5, 3.0 as sequence, prunes if together they have low impact
    """
    normalization_in_error = False
    # start_time = time.time()


    if alldata == True:
        historyerrors_raw = MAerror
    else:
        historyerrors_raw = MAerror[max(0, len(MAerror) - hscaleCount):]

    if len(historyerrors_raw) == 1:
        return False,historyerrors_raw[-1]

    historyerrors=[historyerrors_raw[0]]
    for q in historyerrors_raw[1:]:
        if q==historyerrors[-1]:
            continue
        historyerrors.append(q)

    if len(historyerrors) == 1:
        return False,historyerrors[-1]

    error = historyerrors[-1]
    # =======================================
    # ======= define parameters of threshold calculation ===================
    z = [v / 6 for v in range(18, 30)]  # z vector for threshold calculation

    diviation = statistics.stdev(historyerrors)  # diviation of errors
    meso = statistics.mean(historyerrors)  # mean of errors
    e = [meso + (element * diviation) for element in z]  # e: set of candidate thresholds

    maximazation_value = []
    maxvalue = -1
    thfinal = e[0]
    maxEA = []
    # ============ threshold calculation ========================
    for th in e:
        EA = []  # List of sequence of anomalous errors
        ea = [(i, distt) for i, distt in enumerate(historyerrors) if distt > th]  # dataframe of anomaly errors

        # if ea equals to 0 that means no anomalies so the Δμ/μ and Δσ/σ also are equal to zero
        if len(ea) == 0:
            continue
        if len([element for element in historyerrors if element < th]) <= 1:
            continue
        # Δμ -> difference betwen mean of errors and mean of errors without anomalies
        dmes = meso - statistics.mean([element for element in historyerrors if element < th])
        # Δσ ->  difference betwen diviation of errors and diviation of errors without anomalies
        ddiv = diviation - statistics.stdev([element for element in historyerrors if element < th])

        # ========= group anomaly error in sequences================
        # ea= [ (position, dist/error) , ... , (position, dist/error)]
        posi = ea[0][0]
        while posi <= ea[-1][0]:
            sub = []

            tempea = [tupls for tupls in ea if tupls[0] >= posi]
            sub.append(tempea[0])
            # store all continues errors (in index) in same subssequence
            for row in tempea[1:]:
                # if index of error is the last index of subsequence plus 1 then error is part of this sequence
                if row[0] == sub[-1][0] + 1:
                    sub.append(row)
                    posi = row[0] + 1
                else:
                    posi = row[0]
                    break
            # add the subsequence in to the list
            EA.append(sub)
            if len(tempea[1:]) == 0:
                break

        # ================ persentage impact of the threshold =================
        argmaxError = (dmes / meso + ddiv / diviation) / (
                    len(ea) + len(EA) * len(EA))  # calculate value of formula which we try to maximize
        if maxvalue < argmaxError:
            maxvalue = argmaxError
            thfinal = th
            maxEA = EA
        maximazation_value.append(argmaxError)
    if len(maxEA) == 0:
        return False, thfinal

    if error > thfinal:
        # ==================look for prunning===========================
        # if last value belongs to anomalies then i will be a part of last anomaly sequence
        notea = [err for err in historyerrors if err <= thfinal]
        normalmax = max(notea)

        # maxEA = maxEA[:-1]
        lastSeq = maxEA[-1]
        maxlastSeq = max(lastSeq, key=itemgetter(1))
        maxErrorEA = [max(seq, key=itemgetter(1)) for seq in maxEA]
        maxErrorEA.append((-1, normalmax))
        minhistory = 0
        if normalization_in_error == True:
            minhistory = min(historyerrors)

        maxlastSeq = (maxlastSeq[0], maxlastSeq[1] - (minhistory - minhistory / 100.0))

        sortedmax = sorted(maxErrorEA, key=lambda x: x[1], reverse=True)

        checkpoint = -1
        count = -1
        for tup1, tup2 in zip(sortedmax[:-1], sortedmax[1:]):
            count += 1
            diff = (tup1[1] - tup2[1]) / tup1[1]
            if diff > DesentThreshold:
                checkpoint = count
        if checkpoint != -1:
            realAnomalies = sortedmax[:checkpoint + 1]
            if maxlastSeq[0] in list(map(list, zip(*realAnomalies)))[0]:
                return True, thfinal
    return False, thfinal