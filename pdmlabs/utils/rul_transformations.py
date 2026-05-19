"""Survival probability transformations for RUL (Remaining Useful Life) prediction.

This module provides functions to transform predicted survival probabilities
into RUL (Remaining Useful Life) estimates and time-to-failure predictions.

Key Concepts:
    - Survival probability S(t): Probability that failure hasn't occurred by time t
    - RUL: Time remaining until failure (predicted from survival probabilities)
    - Multiple transformation methods supporting different output shapes

Functions:
    sigmoid_survival: Step survival using logistic sigmoid function
    sigmoid_survival_batch: Batch version for multiple predictions
    softmax_distance_survival: Kernel-based survival via softmax distance
    softmax_distance_survival_batch: Batch version using Laplace/Gaussian kernels
    hard_transform_survival: Hard step function (rectangular) survival curve
    
Example:
    >>> import numpy as np
    >>> from pdmlabs.utils.rul_transformations import sigmoid_survival_batch
    >>> times = np.arange(0, 100)  # 0 to 100 time units
    >>> predictions = [30, 45, 60]  # Predicted RULs for 3 samples
    >>> survivals = sigmoid_survival_batch(times, predictions, tau=5.0)
    >>> # survivals[0] is survival curve for first prediction (RUL=30)
"""

import numpy as np
from math import exp

def sigmoid_survival(times, rhat, tau=5.0):
    """Smooth survival curve using logistic sigmoid function.
    
    Creates a smooth S-shaped survival curve centered at rhat. Uses a logistic
    function to transition from 1 (high survival) to 0 (failed) at the predicted
    time of failure.
    
    Mathematical Formula:
        S(t) = 1 / (1 + exp((t - rhat) / tau))
    
    Parameters
    ----------
    times : array-like
        Time points at which to evaluate survival (typically 0 to max_time).
    rhat : float
        Predicted time of failure (hat-r = "r-hat"). Center of sigmoid curve.
    tau : float, default=5.0
        Temperature parameter controlling curve smoothness.
        - Larger tau: smoother, more gradual transition
        - Smaller tau: sharper, steeper transition
        - tau=1 is standard logistic sigmoid
    
    Returns
    -------
    np.ndarray
        Survival probabilities at each time point.
        Range: (0, 1), monotonically decreasing from ~1 to ~0.
    
    Examples
    --------
    >>> times = np.arange(0, 100)
    >>> rhat = 50  # Predict failure at time 50
    >>> survival = sigmoid_survival(times, rhat, tau=5)
    >>> # survival[50] ≈ 0.5
    >>> # survival[30] ≈ 0.95 (high survival at early time)
    >>> # survival[70] ≈ 0.05 (low survival after failure time)
    
    Notes
    -----
    - At t=rhat, S(t) ≈ 0.5 (50% failure probability)
    - Smooth and differentiable, suitable for gradient-based learning
    - The tau parameter controls the speed of the transition
    """
    times = np.array(times)
    return 1.0 / (1.0 + np.exp((times - rhat) / tau))  # ~1 for t<<rhat, ~0 for t>>rhat

def sigmoid_survival_batch(times, rhat_batch, tau=5.0):
    """Batch version: compute sigmoid survival for multiple RUL predictions.
    
    Efficiently computes survival curves for multiple samples at once.
    
    Parameters
    ----------
    times : array-like
        Shared time grid for all predictions.
    rhat_batch : array-like
        Array of predicted failure times (one per sample).
    tau : float, default=5.0
        Temperature parameter for sigmoid smoothness.
    
    Returns
    -------
    np.ndarray
        Array of shape (len(rhat_batch), len(times)) where each row is a
        survival curve for the corresponding RUL prediction.
    
    Examples
    --------
    >>> times = np.arange(0, 100)
    >>> rhat_list = [30, 50, 70]  # 3 samples with different RULs
    >>> survivals = sigmoid_survival_batch(times, rhat_list, tau=5)
    >>> # survivals.shape = (3, 100)
    """
    all_survivals = []
    for rhat in rhat_batch:
        survival = sigmoid_survival(times, rhat, tau)
        all_survivals.append(survival)
    return np.array(all_survivals)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=0)

def softmax_distance_survival(times, rhat, tau=5.0, kernel='laplace'):
    """Create discrete survival curve using softmax over distance-based kernel.
    
    Instead of using a smooth sigmoid, this method treats the time grid as a discrete
    probability distribution (via softmax) centered at rhat, then converts to survival.
    
    Process:
    1. Compute distances from each time to the predicted failure time rhat
    2. Apply kernel (Laplace or Gaussian) to distances
    3. Normalize distances to a probability mass function (PMF) via softmax
    4. Convert PMF to survival: S(t) = P(T > t)
    
    Parameters
    ----------
    times : array-like
        Discrete time grid (e.g., [0, 1, ..., 100 days]).
    rhat : float
        Predicted time of failure (center of the distribution).
    tau : float, default=5.0
        Temperature parameter for kernel:
        - Laplace: exp(-|t - rhat| / tau)
        - Gaussian: exp(-(t - rhat)^2 / (2*tau^2))
    kernel : {'laplace', 'gaussian'}, default='laplace'
        - 'laplace': Exponential decay from rhat (heavier tails)
        - 'gaussian': Normal distribution around rhat (symmetric, lighter tails)
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (pmf, survival) where:
        - pmf: Probability mass function (sum=1) of failure time distribution
        - survival: Survival curve S(t) = P(T > t)
    
    Examples
    --------
    >>> times = np.arange(0, 100)
    >>> pmf, survival = softmax_distance_survival(times, rhat=50, tau=5, kernel='laplace')
    >>> print(f"Failure most likely at: {times[np.argmax(pmf)]}")
    >>> print(f"Survival at t=60: {survival[60]:.3f}")
    
    Notes
    -----
    - Discrete version of continuous distributions
    - Kernel choice affects tail behavior (how far from rhat matters)
    - Useful for categorical/binned time predictions
    """
    times = np.array(times)
    if kernel == 'laplace':
        logits = -np.abs(times - rhat) / tau
    elif kernel == 'gaussian':
        logits = -((times - rhat )**2) / (2.0 * (tau**2))
    else:
        raise ValueError("kernel must be 'laplace' or 'gaussian'")
    pmf = softmax(logits)
    # compute survival: S(t_k) = sum_{j: t_j > t_k} pmf_j
    cdf = np.cumsum(pmf)
    survival = 1.0 - cdf + pmf  # survival at t_k should include probability at exactly t_k? adjust if desired
    # The line above gives S(t_k)=sum_{j>=k+1} pmf_j + pmf_k; depends on your time grid convention.
    # A clearer approach below: survival at t is P(T>t) -> sum_{j: times[j] > t} pmf_j
    survival_strict = np.array([pmf[times > t_k].sum() for t_k in times])
    return pmf, survival_strict


def softmax_distance_survival_batch(times, rhat_batch, tau=5.0, kernel='laplace'):
    """Batch version: compute kernel-based survival for multiple RUL predictions.
    
    Parameters
    ----------
    times : array-like
        Shared time grid.
    rhat_batch : array-like
        Array of predicted failure times.
    tau : float, default=5.0
        Temperature parameter.
    kernel : {'laplace', 'gaussian'}, default='laplace'
        Kernel type.
    
    Returns
    -------
    np.ndarray
        Array of shape (len(rhat_batch), len(times)) with survival curves.
    """
    # all_pmfs = []
    all_survivals = []
    for rhat in rhat_batch:
        _, survival = softmax_distance_survival(times, rhat, tau, kernel)
        # all_pmfs.append(pmf)
        all_survivals.append(survival)
    return np.array(all_survivals)

def hard_transform_survival(times, flatten_preds):
    """Create hard step survival curves (rectangular, non-smooth).
    
    Simple deterministic survival: object survives until the predicted failure time,
    then immediately fails. Creates sharp binary transitions.
    
    Mathematical Formula:
        S(t) = 1 if t < rhat else 0
    
    Parameters
    ----------
    times : array-like
        Time grid at which to evaluate survival.
    flatten_preds : array-like
        Array of predicted failure times (one per sample).
    
    Returns
    -------
    np.ndarray
        Array of shape (len(flatten_preds), len(times)) with survival curves.
        Each row is [1, 1, ..., 1, 0, 0, ..., 0] with transition at rhat.
    
    Examples
    --------
    >>> times = np.arange(0, 100, 10)
    >>> predictions = [30, 50, 70]
    >>> survivals = hard_transform_survival(times, predictions)
    >>> print(survivals[0])  # Hard step at 30
    [1.0, 1.0, 1.0, 0.0, 0.0, ...]
    
    Notes
    -----
    - Non-smooth and non-differentiable at failure time
    - Useful for deterministic scenarios or baseline comparisons
    - Computationally fastest among survival transformations
    - No probability smoothing or uncertainty modeling
    """
    test_preds = []
    for pred in flatten_preds:
        surv_pred = []
        for t in times:
            if t < pred:
                surv_pred.append(1.0)
            else:
                surv_pred.append(0.0)
        test_preds.append(surv_pred)
    test_preds = np.array(test_preds)
    return test_preds