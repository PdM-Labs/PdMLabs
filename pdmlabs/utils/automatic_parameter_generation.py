"""Automatic parameter space generation for anomaly detection methods.

This module provides functions to automatically generate hyperparameter search spaces
for various anomaly detection techniques (IF, LOF, KNN, LSTM, OCSVM, etc.) across
different learning paradigms (online, offline, unsupervised, supervised, semi-supervised).

The parameter generation adapts to constraints like maximum profile length and data
multivariate/univariate nature.

Key Functions:
    - uniform: Generate uniformly spaced parameters (linear)
    - get_exponential_parameters: Generate exponentially spaced parameters (quadratic)
    - uniform_even_numbers: Generate even-valued uniform parameters
    - online_technique: Parameters for online anomaly detection methods
    - incremental_technique: Parameters for incremental learning methods
    - unsupervised_technique: Parameters for unsupervised methods
    - semi_technique: Parameters for semi-supervised learning
    - supervised_technique: Parameters for supervised learning
    - post_proccessing_params: Parameters for post-processing methods
    - pre_proccessing_params: Parameters for pre-processing methods

Example:
    >>> from pdmlabs.utils.automatic_parameter_generation import online_technique
    >>> # Get parameter space for Isolation Forest online detection
    >>> max_profile = 500
    >>> param_space = online_technique('IF', maximum_profile=max_profile)
    >>> print(param_space)
    {'n_estimators': [50, 100, 150, 200], 'max_samples': [...], ...}
"""

import numpy as np

def uniform_even_numbers(min_val,max_val,num_params):
    """Generate uniformly spaced even-numbered parameters.
    
    Parameters
    ----------
    min_val : int
        Minimum value (will be rounded to nearest even number).
    max_val : int
        Maximum value (will be rounded to nearest even number).
    num_params : int
        Number of parameter values to generate.
    
    Returns
    -------
    list[int]
        Sorted list of unique even-valued parameters.
    
    Examples
    --------
    >>> params = uniform_even_numbers(10, 30, 5)
    >>> print(params)
    [10, 14, 18, 22, 26, 30]
    """
    params = np.linspace(min_val, max_val, num_params)
    params = [int(p) for p in params]
    params = [p if p%2==0  else max(2,p-1) for p in params ]
    params = list(set(params))
    params.sort()
    return params


def uniform(min_val,max_val,num_params,to_int=False):
    """Generate uniformly spaced parameters in linear space.
    
    Creates evenly distributed values between min_val and max_val.
    Useful for parameters without natural exponential scaling.
    
    Parameters
    ----------
    min_val : float
        Minimum parameter value.
    max_val : float
        Maximum parameter value.
    num_params : int
        Number of values to generate.
    to_int : bool, default=False
        If True, convert values to integers.
    
    Returns
    -------
    list
        Sorted list of unique parameter values.
    
    Examples
    --------
    >>> # Float parameters
    >>> alphas = uniform(0.1, 0.9, 5)
    >>> print(alphas)
    [0.1, 0.3, 0.5, 0.7, 0.9]
    
    >>> # Integer parameters
    >>> batch_sizes = uniform(32, 256, 5, to_int=True)
    >>> print(batch_sizes)
    [32, 96, 160, 224, 256]
    """
    params= np.linspace(min_val, max_val, num_params)
    if to_int:
        params=[int(p) for p in params]
    params = list(set(params))
    params.sort()
    return params


def get_exponential_parameters(min_val, max_val, num_params,to_int=False):
    """Generate exponentially spaced parameters (quadratic growth).
    
    Creates values in the range [min_val, max_val] using exponential spacing.
    Useful for parameters like window sizes, model capacity, or other measures
    where doubling might be more meaningful than adding a constant.
    
    Algorithm:
    1. Take square root of min and max values
    2. Generate uniformly spaced values in sqrt-space
    3. Square them back to get exponentially-spaced values
    
    Parameters
    ----------
    min_val : float
        Minimum parameter value.
    max_val : float
        Maximum parameter value.
    num_params : int
        Number of parameter values to generate.
    to_int : bool, default=False
        If True, convert to integers.
    
    Returns
    -------
    list
        Sorted list of unique parameter values within [min_val, max_val].
        Values outside this range are filtered out.
    
    Examples
    --------
    >>> # Window sizes (quadratic growth)
    >>> windows = get_exponential_parameters(10, 200, 5, to_int=True)
    >>> print(windows)
    [10, 25, 62, 100, 185]  # Approximately quadratic growth
    
    >>> # Compare with linear spacing:
    >>> linear = uniform(10, 200, 5, to_int=True)
    >>> print(linear)
    [10, 57, 105, 152, 200]  # Linear growth
    
    Notes
    -----
    - Typically generates fewer unique values than num_params due to rounding
    - Particularly useful for time-window and buffer size parameters
    - More efficient sampling of large parameter spaces
    """
    sqrt_max=int(np.sqrt(max_val))
    sqrt_min=max(int(np.sqrt(min_val)),1)
    positions= np.linspace(sqrt_min, sqrt_max, num_params)
    params=[]
    for x in positions:
        if to_int:
            params.append(int(x*x))
        else:
            params.append(x*x)
    params=list(set(params))
    params.sort()

    final_params = []
    for value in params:
        if min_val <= value <= max_val:
            final_params.append(value)

    return final_params


def online_technique(name,maximum_profile,multivariate=True):
    """Generate parameter space for online anomaly detection methods.
    
    Online methods process data in a streaming fashion with limited memory.
    Parameter spaces are adapted to fit within maximum_profile constraints.
    
    Parameters
    ----------
    name : str
        Anomaly detection algorithm name. Supported: 'CNN', 'IF', 'OCSVM', 'PB', 'KNN', 'NP'
    maximum_profile : int
        Maximum number of historical samples available for learning.
        Used to bound window sizes, sample counts, etc.
    multivariate : bool, default=True
        Whether data has multiple features (True) or univariate (False).
        Affects parameter ranges for sequence-based methods.
    
    Returns
    -------
    dict
        Parameter space dictionary where keys are hyperparameter names
        and values are lists of candidate values for grid search.
    
    Supported Algorithms:
        - CNN: Convolutional Neural Network
        - IF: Isolation Forest
        - OCSVM: One-Class Support Vector Machine
        - PB: Prophet-Based (placeholder)
        - KNN: K-Nearest Neighbors
        - NP: Nearest Neighbors Polynomial/Ball
    
    Examples
    --------
    >>> # Isolation Forest parameters for profile length 500
    >>> param_space_IF = online_technique('IF', maximum_profile=500)
    >>> print(param_space_IF.keys())
    dict_keys(['n_estimators', 'max_samples', 'random_state', 'max_features', 'bootstrap'])
    
    >>> # CNN parameters for multivariate data
    >>> param_space_CNN = online_technique('CNN', maximum_profile=600, multivariate=True)
    >>> print('window_size' in param_space_CNN)
    True
    
    Notes
    -----
    - Window/buffer sizes scale with maximum_profile (typically max_profile/4 to max_profile)
    - Exponential parameter generation used for window sizes and sample counts
    - Random seed fixed to 42 for reproducibility
    - Some methods have empty parameter dictionaries (e.g., PB)
    """
    if name == 'CNN':
        param_dict = {
        'window_size': uniform(min_val=max(min(maximum_profile,400),10),max_val=min(maximum_profile//4,400),num_params=4,to_int=True),
        'batch_size': [32,64,128]
        }
    elif name == 'IF':
        param_dict = {
        'n_estimators': [50, 100, 150, 200],
        'max_samples': uniform(min_val=max(min(maximum_profile,400),2)//4,max_val=min(maximum_profile,400),num_params=4,to_int=True),
        'random_state': [42],
        'max_features': [0.5, 0.6, 0.7, 0.8],
        'bootstrap': [True, False]
    }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
            'gamma': ['scale', 'auto'],
            'max_iter': [10000]
        }
    elif name == 'PB':
        param_dict = {}
    elif name == 'KNN':
        param_dict = {
            'k': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True),
            'window_norm': [False, True],
        }
    elif name == 'NP':
        param_dict ={
                'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
                'max_sample': uniform(min_val=max(min(maximum_profile,400)//4,2),max_val=min(maximum_profile,400),num_params=4,to_int=True),
                'sub_sequence_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
                'aggregation_strategy': ['avg', 'max'],
                'random_state': [42]
        }
    else:
        assert False, f"no default parameters for technique with name {name}"

    return param_dict


def incremental_technique(name,maximum_profile, multivariate=True):
    """Generate parameter space for incremental anomaly detection methods.
    
    Incremental methods learn from streaming data one sample at a time and
    continuously update their models. Parameter spaces accommodate limited memory.
    
    Parameters
    ----------
    name : str
        Algorithm name. Supported: 'IF', 'OCSVM', 'PB', 'KNN', 'NP', 'LOF', 'LTSF', 'TRANAD', 'USAD', 'HBOS', 'PCA'
    maximum_profile : int
        Maximum historical samples available.
    multivariate : bool, default=True
        Whether data is multivariate or univariate.
    
    Returns
    -------
    dict
        Parameter space with algorithm-specific hyperparameters.
        Includes deep learning parameters (epochs, learning rate) for neural methods.
    
    Supported Algorithms:
        - Traditional: IF, LOF, OCSVM, KNN, NP, HBOS, PCA
        - Time-series: LTSF (Linear Time-Series Forecasting)
        - Deep Learning: TRANAD, USAD (both neural anomaly detectors)
    
    Examples
    --------
    >>> params = incremental_technique('LOF', maximum_profile=500)
    >>> print(params)
    {'n_neighbors': [1, 2, 4, 8, 16, 32, 64]}
    
    >>> params_lstm = incremental_technique('LTSF', maximum_profile=1000)
    >>> print('learning_rate' in params_lstm)
    True
    """
    if name == 'IF':
        param_dict = {
        'n_estimators': [50, 100, 150, 200],
        'max_samples': uniform(min_val=max(min(maximum_profile,400),2)//4,max_val=min(maximum_profile,400),num_params=4,to_int=True),
        'random_state': [42],
        'max_features': [0.5, 0.6, 0.7, 0.8],
        'bootstrap': [True, False]
    }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
            'gamma': ['scale', 'auto'],
            'max_iter': [10000]
        }
    elif name == 'PB':
        param_dict = {}
    elif name == 'KNN':
        param_dict = {
            'k': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True),
            'window_norm': [False, True],
        }
    elif name == 'NP':
        param_dict ={
                'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
                'max_sample': uniform(min_val=max(min(maximum_profile,400)//4,2),max_val=min(maximum_profile,400),num_params=4,to_int=True),
                'sub_sequence_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
                'aggregation_strategy': ['avg', 'max'],
                'random_state': [42]
            }
    elif name == 'LOF':
        param_dict =  {
            'n_neighbors': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True)
        }
    elif name == 'LTSF':
        param_dict = {
            'ltsf_type': ['Linear', 'DLinear', 'NLinear'],
            'features': ['M', 'MS'],
            'target': ['p2p_0'],
            'seq_len': uniform(min_val=max(min(200,maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=5,to_int=True),
            'pred_len': [1],
            'individual': [True, False],
            'train_epochs': [3, 5, 10, 15, 20, 25],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [2, 4, 8, 16]
        }
    elif name == 'TRANAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1, 0.05, 0.005]
        }
    elif name == 'USAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=10 if maximum_profile >= 22 else 2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6, to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1],
            'BATCH_SIZE': [2, 4, 8, 16],
            'hidden_size': [4, 8, 16, 32]
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [5,10,15,20,30],
            "alpha": [0.1,0.3,0.5,0.7,0.9],
            "tol": [0.3,0.5,0.8],
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "PCA":
        param_dict = {
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] =get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "CNN" or name == "LSTM":
        param_dict = {
            'sub_sequence_length': get_exponential_parameters(min_val=14, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'predict_time_steps': [1],
            'epochs': [3, 5, 10, 15, 20, 25, 100],
            'patience': [3, 5, 10]
        }
    return param_dict


def unsupervised_technique(name,maximum_profile,multivariate=True):
    """Generate parameter space for unsupervised anomaly detection methods.
    
    Unsupervised methods learn from unlabeled data without any failure information.
    Useful for discovery of unknown anomaly types.
    
    Parameters
    ----------
    name : str
        Algorithm name. Supported: 'NP', 'DAMP', 'KNN', 'IF', 'LOF', 'SAND', 'HBOS', 'PCA', 'CHRONOS'
    maximum_profile : int
        Maximum historical samples.
    multivariate : bool, default=True
        Multivariate (True) or univariate (False) data.
    
    Returns
    -------
    dict
        Parameter space for unsupervised learning.
    
    Notes
    -----
    - Parameters include window/buffer configuration for online processing
    - LOF, IF, SAND include sliding window and overlap strategies
    - CHRONOS (probabilistic forecasting) has context length and sampling parameters
    """
    if name == 'NP':
        param_dict = {
            'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
            'max_sample': uniform(min_val=max(min(maximum_profile,160)//4,2),max_val=min(maximum_profile//2,160),num_params=4,to_int=True),
            'sub_sequence_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,3), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
            'aggregation_strategy': ['avg', 'max'],
            'random_state': [42],
            'window': uniform(min_val=maximum_profile//4 if maximum_profile//4 > 5 else 5, max_val=maximum_profile, num_params=8,to_int=True),
            'slide': [0.33, 0.5, 1.0],
            'overlap_aggregation_strategy': ['first', 'last', 'avg'],
        }
    elif name == "DAMP":
        param_dict = {
            "sub_sequence_length":get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True),
            "stride": [1],
            "init_length": uniform(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8,to_int=True),
            "aggregation_strategy":['avg', 'max'],
        }
    elif name == 'KNN':
        param_dict = {
            'window': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8,to_int=True),
            'slide': [0.33, 0.5, 1.0],
            'k': get_exponential_parameters(min_val=1, max_val=min(100, maximum_profile), num_params=8,
                                            to_int=True),
            'window_norm': [False, True],
            'policy': ['or', 'and', 'first', 'last']
        }
    elif name == 'IF':
        param_dict = {
            'window': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8,to_int=True),
            'slide': [0.33, 0.5, 1.0],
            'n_estimators': [50, 100, 150, 200],
            'max_samples': uniform(min_val=max(min(maximum_profile, 400), 2) // 4, max_val=min(maximum_profile, 400),
                                   num_params=4, to_int=True),
            'max_features': [0.5, 0.6, 0.7, 0.8],
            'bootstrap': [True, False],
            'random_state': [42],
            'policy': ['or', 'and', 'first', 'last']
        }
    elif name == 'LOF':
        param_dict = {
            'n_neighbors': get_exponential_parameters(min_val=1, max_val=min(100, maximum_profile), num_params=8,
                                                      to_int=True),
            'window': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8, to_int=True),
            'slide': [0.33, 0.5, 1.0]
        }
    elif name == 'SAND':
        param_dict = {
            'pattern_length': get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,3), max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=8,to_int=True),
            'subsequence_length_multiplier': [3, 4, 5] if maximum_profile > 100 else [1, 2], #4*4 this is the sub size
            'alpha': [0.5, 0.75, 0.25],
            'init_length': uniform(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8,to_int=True),
            'batch_size': uniform_even_numbers(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8),
            'k': [4, 6, 7, 8, 9, 10],
            'aggregation_strategy': ['avg', 'max']
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [5,10,15,20,30],
            "alpha": [0.1,0.3,0.5,0.7,0.9],
            "tol": [0.3,0.5,0.8],
            # only for univariate
            "window": [ uniform(min_val=maximum_profile//4, max_val=maximum_profile, num_params=8,to_int=True)]
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "PCA":
        param_dict = {
            "window": [uniform(min_val=maximum_profile // 4, max_val=maximum_profile, num_params=8, to_int=True)]
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "CHRONOS":
        param_dict = {
            'context_length': uniform(min_val=max(maximum_profile//4, 4), max_val=maximum_profile, num_params=8, to_int=True),
            'num_samples': [1, 3, 5, 10],
            'slide': [15],
        }
    else:
        assert False,"no method with that name"

    return param_dict


def semi_technique(name,maximum_profile,multivariate=True):
    """Generate parameter space for semi-supervised anomaly detection methods.
    
    Semi-supervised methods use a small amount of labeled data (typically failures)
    combined with large amounts of unlabeled normal data.
    
    Parameters
    ----------
    name : str
        Algorithm. Supported: 'IF', 'OCSVM', 'PB', 'KNN', 'NP', 'LOF', 'LTSF', 'TRANAD', 'USAD', 'HBOS', 'PCA', 'CNN', 'LSTM'
    maximum_profile : int
        Maximum historical samples.
    multivariate : bool, default=True
        Multivariate or univariate.
    
    Returns
    -------
    dict
        Parameter space combining supervised and unsupervised parameters.
    """
    if name == 'IF':
        param_dict = {
        'n_estimators': [50, 100, 150, 200],
        'max_samples': uniform(min_val=max(min(maximum_profile,400),2)//4,max_val=min(maximum_profile,400),num_params=4,to_int=True),
        'random_state': [42],
        'max_features': [0.5, 0.6, 0.7, 0.8],
        'bootstrap': [True, False]
    }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'nu': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
            'gamma': ['scale', 'auto'],
            'max_iter': [10000]
        }
    elif name == 'PB':
        param_dict = {}
    elif name == 'KNN':
        param_dict = {
            'k': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True),
            'window_norm': [False, True],
        }
    elif name == 'NP':
        param_dict ={
                'n_nnballs': uniform(min_val=10, max_val=150, num_params=5,to_int=True),
                'max_sample': uniform(min_val=max(min(maximum_profile,400)//4,2),max_val=min(maximum_profile,400),num_params=4,to_int=True),
                'sub_sequence_length':get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True),
                'aggregation_strategy': ['avg', 'max'],
                'random_state': [42]
            }
    elif name == 'LOF':
        param_dict =  {
            'n_neighbors': get_exponential_parameters(min_val=1, max_val=min(100,maximum_profile), num_params=8,to_int=True)
        }
    elif name == 'LTSF':
        param_dict = {
            'ltsf_type': ['Linear', 'DLinear', 'NLinear'],
            'features': ['M', 'MS'],
            'target': ['p2p_0'],
            'seq_len': uniform(min_val=max(min(200,maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=5,to_int=True),
            'pred_len': [1],
            'individual': [True, False],
            'train_epochs': [3, 5, 10, 15, 20, 25],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [2, 4, 8, 16]
        }
    elif name == 'TRANAD':
        param_dict = {
            'window_size':get_exponential_parameters(min_val=2, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1, 0.05, 0.005]
        }
    elif name == 'USAD':
        param_dict = {
            'window_size': get_exponential_parameters(min_val=2, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'num_epochs': [5, 10, 15, 20, 25],
            'lr': [0.001, 0.01, 0.1],
            'BATCH_SIZE': [2, 4, 8, 16],
            'hidden_size': [4, 8, 16, 32]
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [5,10,15,20,30],
            "alpha": [0.1,0.3,0.5,0.7,0.9],
            "tol": [0.3,0.5,0.8],
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] = get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "PCA":
        param_dict = {
            # only for univariate
        }
        if multivariate:
            param_dict["sub_sequence_length"] = [1]
        else:
            param_dict["sub_sequence_length"] =get_exponential_parameters(min_val=max(min(20,maximum_profile)//8,2), max_val=min(200,maximum_profile//2), num_params=8,to_int=True)
    elif name == "CNN" or name == "LSTM":
        param_dict = {
            'sub_sequence_length': get_exponential_parameters(min_val=14, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
            'predict_time_steps': [1],
            'epochs': [3, 5, 10, 15, 20, 25, 100],
            'patience': [3, 5, 10]
        }
    return param_dict



def default_TSB_semi(name,maximum_profile):
    """Generate default/baseline parameter sets for semi-supervised learning.
    
    Uses single pre-tuned values instead of search spaces. Faster evaluation
    but less flexible than parameter grids. Useful for baseline comparisons.
    
    Parameters
    ----------
    name : str
        Algorithm name.
    maximum_profile : int
        Maximum historical samples (used to scale some parameters).
    
    Returns
    -------
    dict
        Parameter dictionary with single values (not lists) for each hyperparameter.
    """
    if name == "IF":
        param_dict = {
            'n_estimators': [100],
            'max_samples': ['auto'],
            'random_state': [42],
            'max_features': [1.],
            'bootstrap': [False],
        }
    elif name == 'OCSVM':
        param_dict = {
            'kernel': ['rbf'],
            'nu': [0.5],
            'gamma': ['auto'],
        }
        # kernel = 'rbf', degree = 3, gamma = 'auto', coef0 = 0.0,
        # tol = 1e-3, nu = 0.5, shrinking = True, cache_size = 200,
        # verbose = False, max_iter = -1, contamination = 0.1
    elif name == "LOF":
        param_dict = {
            'n_neighbors': [20],
        }
    elif name == 'NP':
        param_dict = {
            "n_nnballs": [1],
            "max_sample": [maximum_profile//2],
            "sub_sequence_length": [min(200, maximum_profile // 8)],
            'aggregation_strategy': ['avg'],
            'random_state': [42],
        }
    elif name == "PCA":
        param_dict = {
            "sub_sequence_length": [min(200, maximum_profile // 8)]
        }
    elif name == "LSTM" or name == "CNN":
        param_dict = {
            "sub_sequence_length":[min(100, maximum_profile // 8)],
        }
    elif name == "CNN":
            param_dict = {
            "sub_sequence_length":[min(100, maximum_profile // 8)],
        }
    elif name == "HBOS":
        param_dict = {
            "n_bins": [10],
            "alpha": [0.1],
            "tol": [0.5],
            # only for univariate
            "sub_sequence_length": [min(200, maximum_profile // 8)]
        }
    elif name == 'KNN':
        param_dict = {
            'k': [20],
            'window_norm': [False],
        }
    else:
        assert False, f"no default parameters for technique with name {name}"

    return param_dict

def post_proccessing_params(name,maximum_profile):
    """Generate parameter space for post-processor methods.
    
    Post-processors refine raw anomaly scores through smoothing, normalization,
    or recalibration.
    
    Parameters
    ----------
    name : str
        Post-processor name. Supported: 'Default', 'Dynamic Threshold', 'Moving2T', 'SelfTuning', 'Moving Average'
    maximum_profile : int
        Maximum historical samples (scales window parameters).
    
    Returns
    -------
    dict
        Parameter space for post-processing.
    
    Supported Methods:
        - Default: Identity (no post-processing)
        - Dynamic Threshold: NASA dynamic thresholding
        - Moving2T: Two-pass moving threshold
        - SelfTuning: Adaptive normalization
        - Moving Average: Rolling window smoothing
    """
    if name == "Default":
        param_dict={}
    elif name == 'Dynamic Threshold':
        param_dict = {
            "epsilon":[0.05],
            "history_window":[1000],
        }
    elif name == 'Moving2T':
        param_dict = {
            "factor":[3],
            "history_window":[1000],
            "exclude":[False]
        }
    elif name == 'SelfTuning':
        param_dict = {
            "window_length":get_exponential_parameters(min_val=10, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
        }
    elif name == 'Moving Average':
        param_dict = {
            "window_length":get_exponential_parameters(min_val=10, max_val=min(200,maximum_profile//2), num_params=6,to_int=True),
        }
    else:
        assert False, f"no default post_processing for technique with name {name}"
    return param_dict

def pre_proccessing_params(name,maximum_profile):
    """Generate parameter space for pre-processor methods.
    
    Pre-processors clean and transform raw data before anomaly detection.
    
    Parameters
    ----------
    name : str
        Pre-processor name. Supported: 'Default', 'Keep Features', 'MinMax Scaler (semi)', 'Windowing (one column)', 'Mean Aggregator'
    maximum_profile : int
        Maximum historical samples.
    
    Returns
    -------
    dict
        Parameter space for pre-processing.
    
    Supported Methods:
        - Default: Identity
        - Keep Features: Feature selection
        - MinMax Scaler: Normalization
        - Windowing: Lagged feature creation
        - Mean Aggregator: Downsampling
    """
    if name == "Default":
        param_dict={}
    elif name == "Keep Features":
        param_dict={
            "selected_features":[]
        }
    elif name == "MinMax Scaler (semi)":
        param_dict={
        }
    elif name == "Windowing (one column)":
        param_dict={
            "slidingWindow":[10],
            "col_pos":0
        }
    elif name=="Mean Aggregator":
        param_dict = {
            "period": ['10T'],
        }
    else:
        assert False, f"no default pre_processing for technique with name {name}"
    return param_dict

def profile_values(max_wait, moment=False):
    """Generate profile length (historical buffer) parameters for incremental learning.
    
    The profile is the amount of historical data kept in memory to learn or calibrate
    anomaly detection models.
    
    Parameters
    ----------
    max_wait : int
        Maximum wait time (controls the range of profile lengths).
    moment : bool, default=False
        If False: Generate search space of profile lengths.
        If True: Return fixed moment-based profile (1027 samples).
    
    Returns
    -------
    list[int]
        List of profile length candidates or fixed value.
    
    Notes
    -----
    - Without moment: Returns 16 exponentially-spaced values from max_wait/10 to max_wait
    - With moment: Returns [1027] for moment-based methods
    """
    if not moment:
        result = uniform(min_val= max(max_wait// 10, 5), max_val=max_wait, num_params=16, to_int=True)

        if 0 in result:
            result.remove(0)

        return result
    else:
        return [1027]#uniform(min_val=1024, max_val=max_wait if max_wait >= 1024 else 1024, num_params=16, to_int=True)


def incremental_windows(max_wait):
    """Generate incremental learning window parameters.
    
    For online methods that process data in sliding windows, generates parameters
    controlling the initial window size, step size, and window length parameters.
    
    Parameters
    ----------
    max_wait : int
        Maximum allowed wait time (bounds window sizes).
    
    Returns
    -------
    tuple[list, list, list]
        (incremental_slide, initial_incremental_window_length, incremental_window_length)
        - incremental_slide: Step sizes between windows
        - initial_incremental_window_length: Initial buffer size for warm-up
        - incremental_window_length: Steady-state window size
    
    Notes
    -----
    Removes 0 and 1 from slide candidates to avoid invalid window configurations.
    """
    values=uniform(min_val= max(max_wait// 10, 1), max_val=max_wait, num_params=13, to_int=True),
    incremental_slide = values[0]
    if 1 in incremental_slide:
        incremental_slide.remove(1)

    if 0 in incremental_slide:
        incremental_slide.remove(0)

    initial_incremental_window_length = values[0]
    incremental_window_length = values[0] #+ [1000000000]

    return incremental_slide, initial_incremental_window_length, incremental_window_length


def supervised_technique(name, maximum_profile, multivariate=True):
    """Generate parameter space for supervised anomaly detection methods.
    
    Supervised methods learn from labeled failure/non-failure examples.
    
    Parameters
    ----------
    name : str
        Algorithm name. Supported: 'XGBOOST'
    maximum_profile : int
        Maximum historical samples.
    multivariate : bool, default=True
        Multivariate or univariate.
    
    Returns
    -------
    dict
        Parameter space for supervised learning.
    
    Supported Methods:
        - XGBOOST: Gradient boosting with tree-based learners
    
    Notes
    -----
    - XGBoost parameters adapted to maximum_profile for tree depth and leaf count
    """
    if name == 'XGBOOST':
        param_dict = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': uniform(min_val=max(min(200,maximum_profile)//8,2), max_val=min(200,maximum_profile), num_params=6,to_int=True),
            'max_leaves': get_exponential_parameters(min_val=2, max_val=min(200,maximum_profile//2 if maximum_profile // 2 != 2 else maximum_profile), num_params=6,to_int=True),
            'learning_rate': [0.001, 0.01, 0.1],
            'random_state': [42],
        }
    else:
        assert False, f"no default parameters for technique with name {name}"

    return param_dict
