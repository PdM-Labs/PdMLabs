"""Pairwise distance calculation utilities for time-series data.

This module provides efficient functions for computing distances between
multiple time-series subsequences using various metrics (Euclidean, DTW,
cross-correlation, RBF kernels, etc.).

Functions:
    calculate_distance_many_to_many: Compute all-pairs distances between two dataframes
    calculate_distance_many_to_one: Compute distances from one point to all rows in dataframe
    cross_dists: Compute cross-correlation distance

Supported Metrics:
    - 'euclidean': Euclidean distance
    - 'manhattan': Manhattan/L1 distance
    - 'cosine': Cosine distance
    - 'rbf_kernel': RBF kernel distance (default gamma=0.5)
    - 'rbf_kernel{gamma}': RBF with custom gamma, e.g., 'rbf_kernel0.1'
    - Other scipy.spatial.distance.cdist metrics

Example:
    >>> import pandas as pd
    >>> from pdmlabs.utils.distances import calculate_distance_many_to_many
    >>> reference_data = pd.DataFrame([[1,2,3], [4,5,6]])
    >>> test_data = pd.DataFrame([[1,2,4], [5,6,7]])
    >>> distances = calculate_distance_many_to_many(reference_data, test_data, 'euclidean')
    >>> # Returns 2x2 distance matrix
"""

from scipy.spatial.distance import cdist
# from dtaidistance import dtw
from sklearn.metrics.pairwise import rbf_kernel
#from tslearn.metrics import cycc
import numpy as np


# calculate all pair distances between two dataframes a and b
def calculate_distance_many_to_many(a, b, metric):
    """Compute all-pairs distances between rows of two dataframes.
    
    Calculates pairwise distances between all rows of dataframe `a` and all
    rows of dataframe `b`, essentially building a distance matrix of shape
    (n_samples_a, n_samples_b).
    
    Parameters
    ----------
    a : pd.DataFrame
        First dataframe, each row is a time-series or feature vector.
    b : pd.DataFrame
        Second dataframe, each row is a time-series or feature vector.
    metric : str
        Distance metric to use:
        - 'euclidean': L2 norm (default for most metrics)
        - 'manhattan': L1 norm
        - 'cosine': Cosine distance
        - 'rbf_kernel': RBF kernel distance (1 - rbf_kernel(a, b))
        - 'rbf_kernel{gamma}': RBF with custom gamma, e.g., 'rbf_kernel0.1'
        - Any metric supported by scipy.spatial.distance.cdist
    
    Returns
    -------
    np.ndarray
        Distance matrix of shape (len(a), len(b)). Element [i,j] is the
        distance between row i of `a` and row j of `b`.
    
    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    >>> b = pd.DataFrame([[1, 2, 4], [7, 8, 9]])
    >>> dist = calculate_distance_many_to_many(a, b, 'euclidean')
    >>> print(dist)
    [[1.0, 6.08276...]
     [6.08276..., 1.0]]
    
    Notes
    -----
    - For RBF kernel: returns 1 - rbf_kernel, so distance increases as similarity decreases
    - Handles empty dataframes gracefully
    """
    distances = []
    # if metric == 'dtw':
    #     for s1 in a.values:
    #         distances.append([])
    #         for s2 in b.values:
    #             ddist = dtw.distance_fast(s1, s2)
    #             distances[-1].append(ddist)
    # elif metric == "cc":
    #     for s1 in a.values:
    #         distances.append([])
    #         for s2 in b.values:
    #             ddist = cross_dists(s1, s2)
    #             distances[-1].append(ddist)
    if 'rbf_kernel' in metric:
        if metric == "rbf_kernel":
            gamma = 0.5
        else:
            gamma = float(metric.split("kernel")[-1])
        distances = 1 - rbf_kernel(a.values, b.values, gamma=gamma)
        distances = distances
    else:
        distances = cdist(a.values, b.values, metric)
    return distances


# calculate distance from point to all profile points
def calculate_distance_many_to_one(a, b, metric):
    """Compute distances from one vector to all rows in a dataframe.
    
    Calculates distances between a single vector `b` and each row of
    dataframe `a`. Essentially the first row of calculate_distance_many_to_many
    when b is a single sample.
    
    Parameters
    ----------
    a : pd.DataFrame
        Dataframe where each row is a time-series or feature vector.
    b : np.ndarray or array-like
        Single vector (1D array) to compare against all rows of `a`.
        Will be reshaped to (1, -1) for distance computation.
    metric : str
        Distance metric (same options as calculate_distance_many_to_many):
        'euclidean', 'manhattan', 'cosine', 'rbf_kernel', etc.
    
    Returns
    -------
    np.ndarray
        1D array of shape (len(a),) where distances[i] is the distance
        between `b` and row i of `a`.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> profile_data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> test_point = np.array([1, 2, 4])
    >>> distances = calculate_distance_many_to_one(profile_data, test_point, 'euclidean')
    >>> print(distances)
    [1.0, 5.38516..., 10.29563...]
    
    Notes
    -----
    - More efficient than using calculate_distance_many_to_many with single sample
    - Useful for online/streaming scenarios where you score one sample against many
    - Input `b` doesn't need to be in dataframe format
    """
    distances = []
    # if metric == 'dtw':
    #     for s1 in a.values:
    #         ddist = dtw.distance_fast(s1, b)
    #         distances.append(ddist)
    # # elif metric == "cc":
    #     for s1 in a.values:
    #         ddist = cross_dists(s1, b)
    #         distances.append(ddist)
    if 'rbf_kernel' in metric:
        if metric == "rbf_kernel":
            gamma = 0.5
        else:
            gamma = float(metric.split("kernel")[-1])
        distances = 1 - rbf_kernel(b.reshape(1, -1), a.values, gamma=gamma)
    else:
        distances = cdist(b.reshape(1, -1), a, metric)[0]
    return distances
from tslearn.metrics import cdist_normalized_cc, y_shifted_sbd_vec

def cross_dists( s1, s2):
    """Compute normalized cross-correlation distance between two time-series.
    
    Uses the maximum normalized cross-correlation coefficient between two
    sequences to measure similarity. Distance is 1 - (max correlation).
    
    Parameters
    ----------
    s1 : np.ndarray
        First time-series (1D array).
    s2 : np.ndarray
        Second time-series (1D array).
    
    Returns
    -------
    float
        Distance value between 0 (identical) and 1 (completely anticorrelated).
    
    Notes
    -----
    - Uses tslearn.metrics.cdist_normalized_cc for normalized cross-correlation
    - Effectively measures template matching or shape similarity
    - Invariant to small phase shifts and time warping (local)
    - Range: [0, 1] where 0 is perfect match
    """
    return 1. - cdist_normalized_cc(np.expand_dims(s1, axis=0), np.expand_dims(s2, axis=0)).max()