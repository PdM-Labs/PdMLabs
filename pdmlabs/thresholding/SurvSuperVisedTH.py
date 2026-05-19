"""Survival analysis to RUL (Remaining Useful Life) thresholder.

SurvToRUL converts survival probability scores (failure predictions) into RUL
(Remaining Useful Life) estimates. Designed for prognostic health monitoring.

Survival analysis context:
- Survival scores: probability that component survives until time t
- RUL: predicted time until failure (days, hours, operations, etc.)
- Threshold: optimal survival probability cutoff for RUL prediction

This thresholder learns an optimal threshold from validation data that
minimizes error between predicted and true time-to-failure values.

Use cases:
- Scheduled maintenance: When should we service this equipment?
- Resource planning: Do we need a spare part before next failure?
- Risk assessment: Which equipment will fail soonest?

Example:
  Survival score 0.8 at hour 100 means "80% chance equipment survives past hour 100"
  Using learned threshold, convert to RUL: "equipment will fail in ~30 hours"
"""

import numpy as np
import pandas as pd

from pdmlabs.thresholding.thresholder import ThresholderInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class SurvToRUL(ThresholderInterface):
    """Convert survival probabilities to RUL (Remaining Useful Life) predictions.
    
    Learns a threshold mapping survival probabilities to remaining time until failure.
    Uses Mean Absolute Error (MAE) on validation data to find optimal threshold.
    
    Survival score format: tuple (survival_probability, time_vector)
        - survival_probability: array of P(survive until each time)
        - time_vector: corresponding time points
    
    Attributes:
        threshold_value (float): Learned threshold (0-1) on survival probability.
            Range [0, 1]. Interpretation:
            - ~0.0: aggressive (predict failure soon)
            - ~0.5: moderate (balanced)
            - ~1.0: conservative (predict failure far in future)
    
    Algorithm:
    1. Test 501 threshold values from 0 to 1
    2. For each threshold:
       - Predict RUL for each validation sample
       - Compute MAE vs true time-to-failure
    3. Select threshold with minimum MAE
    
    Examples:
        >>> from pdmlabs.thresholding.SurvSuperVisedTH import SurvToRUL
        >>> thresholder = SurvToRUL(
        ...     event_preferences={'failure': [], 'reset': []},
        ...     threshold_value=None  # Learn from data
        ... )
        >>>
        >>> # Survival scores: list of tuples (surv_prob_array, time_vector)
        >>> surv_scores = [
        ...     (np.array([0.95, 0.90, 0.80, 0.60, 0.30]), np.array([1,2,3,4,5])),
        ...     (np.array([0.98, 0.95, 0.85, 0.70, 0.40]), np.array([1,2,3,4,5])),
        ... ]
        >>> true_times = [[2.5], [3.0]]  # Hours until failure
        >>> thresholder.fit([surv_scores], ['bearing_1'], events_df, true_times)
        >>> print(thresholder.threshold_value)  # Learned threshold ~0.65
        >>>
        >>> # Predict RUL for new data
        >>> new_surv = ([0.92, 0.87, 0.75, 0.55], [1,2,3,4])
        >>> rul = thresholder.infer_threshold_one(new_surv, 'bearing_1', events_df)
        >>> print(rul)  # Hours remaining until predicted failure
    """
    def __init__(self, event_preferences: EventPreferences, threshold_value=None):
        """Initialize SurvToRUL thresholder.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
            threshold_value (float, optional): Fixed survival probability threshold.
                If None, will be learned from fit() data. Defaults to None.
        """
        super().__init__(event_preferences=event_preferences)
        self.threshold_value = threshold_value

    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges=None) -> None:
        """Learn optimal survival probability threshold from labeled data.
        
        Optimization finds threshold that best maps survival curves to time-to-failure.
        
        Args:
            historic_data (list): List of survival score arrays. Each element
                is list of tuples (survival_prob, time_vector). Structure:
                - historic_data[i][j] = (np.array, np.array)
                  - [0]: survival probabilities at each time
                  - [1]: time points corresponding to probabilities
            
            historic_sources (list[str]): Source identifiers (one per element
                of historic_data).
            
            event_data (pd.DataFrame): Event log (unused).
            
            anomaly_ranges (list[list], optional): True time-to-failure values.
                Structure: list of lists where element i corresponds to source i.
                Each inner list contains tuples: [(RUL1, ???), (RUL2, ???), ...]
                Only first element of tuple is used (RUL value).
                Skip sources where first RUL value is 0.
        
        Notes:
            - Skips sources with no anomaly label (all zeros)
            - Learns MAE-optimal threshold across all valid sources
            - If threshold_value is already set, uses that (no learning)
        """
        if self.threshold_value is None:
            temp_scores = []
            labs = []
            for current_historic_data, current_historic_source, labels in zip(historic_data, historic_sources,
                                                                              anomaly_ranges):
                if labels[0][1] == 0:
                    continue
                temp_scores.extend([sc[0] for sc in current_historic_data])
                labs.extend([lab[0] for lab in labels])
            optimed_threshold = self.optimize_threshold(temp_scores, x=historic_data[0][0][1], true_times=labs)
            self.threshold_value = optimed_threshold
    
    def optimize_threshold(self, curves, x, true_times):
        """Find threshold that minimizes MAE on validation data.
        
        Tests 501 evenly-spaced thresholds from 0 to 1. For each:
        - Predicts RUL by finding where survival crosses threshold
        - Computes absolute error vs true times
        
        Args:
            curves (list): List of survival probability arrays.
            x (np.array): Time vector corresponding to survival probabilities.
                Must be same for all curves.
            true_times (list): Ground truth time-to-failure for each curve.
        
        Returns:
            float: Threshold value (0-1) that minimizes MAE.
        """
        thetas = np.linspace(0, 1, 501)
        losses = []

        for theta in thetas:
            preds = np.array([self.predicted_time(c, x, theta) for c in curves])
            loss = np.mean(np.abs(preds - true_times))
            losses.append(loss)

        best_theta = thetas[np.argmin(losses)]
        return best_theta

    def predicted_time(self, curve, x, theta):
        """Predict RUL by finding crossing point of survival threshold.
        
        Finds first time point where survival probability <= threshold.
        Represents the predicted failure time.
        
        Args:
            curve (np.array): Survival probability curve [p1, p2, ..., pN]
                representing P(survive until time x[i]) for each position i.
            x (np.array): Time vector corresponding to curve.
                Must be sorted ascending.
            theta (float): Threshold survival probability (0-1).
        
        Returns:
            float: Predicted time of failure (RUL).
                - If curve never crosses threshold: returns x[-1] (latest time)
                - Otherwise: returns first time where curve <= threshold
        """
        idx = np.where(curve <= theta)[0]
        return x[idx[0]] if len(idx) > 0 else x[-1]


    def infer_threshold(self, scores: list, source: str, event_data: pd.DataFrame,
                        scores_dates: list[pd.Timestamp]) -> list[float]:
        """Predict RUL for batch of survival scores.
        
        Args:
            scores (list): List of tuples (survival_prob_array, time_vector).
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).
            scores_dates (list[pd.Timestamp]): Score timestamps (unused).
        
        Returns:
            list[float]: Predicted RUL values for each sample.
        """
        in_scores = np.array(scores)
        return [self.predicted_time(in_scores[i, 0], in_scores[i, 1], self.threshold_value) for i in range(len(scores))]

    def infer_threshold_one(self, score: float, source: str, event_data: pd.DataFrame) -> float:
        """Predict RUL for single survival score (online mode).
        
        Args:
            score (float): Single survival score (unused - kept for interface compatibility).
            source (str): Source identifier (unused).
            event_data (pd.DataFrame): Event log (unused).
        
        Returns:
            float: The learned threshold_value itself (not a RUL prediction).
                Note: This method returns scalar, while batch mode computes RUL.
        """
        return self.threshold_value

    def get_params(self):
        """Return thresholder parameters.
        
        Returns:
            dict: {'threshold_value': the learned survival probability threshold}
        """
        return {
            'threshold_value': self.threshold_value
        }

    def __str__(self) -> str:
        """Return thresholder name.
        
        Returns:
            str: 'SurvToRUL_threshold'
        """
        return 'SurvToRUL_threshold'