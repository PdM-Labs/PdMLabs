"""Utility functions for time-series processing and event preference handling.

This module provides helper functions for:
- Sliding window feature generation from time-series data
- Automatic sliding window length detection using autocorrelation
- Event preference expansion with wildcard matching
- Parameter calculation for MANGO optimization

Key Functions:
    sliding_window: Convert a time-series into sliding windows
    find_length: Automatically determine window length using ACF
    Window: Class for rolling window feature mapping
    process_event_preferences_key: Expand event preferences with wildcards
    expand_event_preferences: Process all event preferences
    calculate_mango_parameters: Compute MANGO optimization parameters

Example:
    >>> import pandas as pd
    >>> from pdmlabs.utils.utils import sliding_window, find_length
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> windows = sliding_window(data, window_len=3, step=1)
    >>> # Estimate window length from ACF
    >>> window_len = find_length(data.values)
    >>> print(f"Recommended window length: {window_len}")
"""

import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np


from pdmlabs.pdm_evaluation_types.types import EventPreferences, EventPreferencesTuple


def sliding_window(dfcol, window_len, step):
    """Convert a time-series column into sliding windows.
    
    Creates non-overlapping (or optionally overlapping based on step) windows
    from a time-series, transforming 1D data into 2D array suitable for ML.
    
    Parameters
    ----------
    dfcol : pd.Series
        The input time-series data.
    window_len : int
        Length of each window.
    step : int
        Step size between consecutive windows. If step < window_len, windows overlap.
    
    Returns
    -------
    pd.DataFrame
        DataFrame where each row is a sliding window, with columns named 'col_1', 'col_2', etc.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    >>> windows = sliding_window(data, window_len=3, step=2)
    >>> print(windows)
       col_1  col_2  col_3
    0     1      2      3
    1     3      4      5
    2     5      6      7
    """
    arr=dfcol.values
    num_windows = (len(arr) - window_len) // step + 1
    windows = np.array([arr[i:i + window_len] for i in range(0, num_windows * step, step)])

    sliding_df = pd.DataFrame(windows, columns=[f'col_{i + 1}' for i in range(window_len)])
    return sliding_df

# determine sliding window (period) based on ACF
def find_length(data):
    """Automatically determine optimal sliding window length using autocorrelation.
    
    Analyzes the autocorrelation function (ACF) to find the first major periodicity
    in the time-series. Uses local maxima detection to identify natural periods.
    
    Parameters
    ----------
    data : array-like
        1D time-series data. Multidimensional arrays return 0.
    
    Returns
    -------
    int
        Recommended window length based on ACF periodicity. Returns 125 as default
        if no clear period is detected or period is outside acceptable range [3, 300].
    
    Notes
    -----
    - Uses first 20,000 samples for efficiency
    - Analyzes up to 400 lags of ACF
    - Returns full period including the base offset
    - Default fallback is 125 samples
    
    Examples
    --------
    >>> import numpy as np
    >>> # Seasonal data with period ~50
    >>> data = np.sin(np.arange(1000) * 2 * np.pi / 50)
    >>> window = find_length(data)
    >>> print(f"Detected window length: {window}")
    """
    if len(data.shape) > 1:
        return 0
    data = data[:min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except:
        return 125


class Window:
    """Rolling window feature mapping for time-series data.
    
    Converts a time-series into a matrix of consecutive overlapping windows,
    where each row represents a window of consecutive timesteps. This is useful
    for creating features for deep learning models and time-series analysis.
    
    The transformation creates lagged features by shifting the series by
    n steps to create n sequential features for each window.
    
    Parameters
    ----------
    window : int, default=100
        The size of each rolling window (number of timesteps to include).
        Use window=0 for no windowing (returns original series).
    
    Attributes
    ----------
    detector : object, optional
        Reference to an anomaly detector (for compatibility).
    
    Examples
    --------
    >>> import numpy as np
    >>> from pdmlabs.utils.utils import Window
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> windower = Window(window=3)
    >>> windowed = windower.convert(data)
    >>> print(windowed)
         0    1    2
    0  NaN  NaN  1.0
    1  NaN  2.0  2.0
    2  3.0  3.0  3.0
    3  4.0  4.0  4.0
    ...
    
    Notes
    -----
    The first (window-1) rows will contain NaN values due to the shifting operation.
    """

    def __init__(self,  window = 100):
        self.window = window
        self.detector = None

    def convert(self, X):
        """Convert a time-series into rolling windows.
        
        Parameters
        ----------
        X : array-like
            1D time-series data to convert.
        
        Returns
        -------
        pd.DataFrame
            DataFrame where each row is a window of consecutive values.
            Shape is (n-window+1, window) where n is the input length.
            First (window-1) rows contain NaN values.
        
        Examples
        --------
        >>> windower = Window(window=3)
        >>> result = windower.convert([1, 2, 3, 4, 5])
        >>> # Returns dataframe with lagged features
        """
        n = self.window
        X = pd.Series(X)
        L = []
        if n == 0:
            df = X
        else:
            for i in range(n):
                L.append(X.shift(i))
            df = pd.concat(L, axis = 1)
            df = df.iloc[n-1:]
        
        return df


def process_event_preference_with_one_dont_care_bit(event_preference: EventPreferencesTuple, event_data: pd.DataFrame, dont_care_bit_index: int) -> list[EventPreferencesTuple]:
    """Expand an event preference rule with one wildcard dimension.
    
    Replaces one wildcard '*' in an event preference with all matching values from
    event_data, generating multiple concrete preference rules.
    
    Parameters
    ----------
    event_preference : EventPreferencesTuple
        Base event preference with one field set to '*' (wildcard).
        Fields: description, type, source, target_sources.
    event_data : pd.DataFrame
        Available event data with columns: ['type', 'source', 'description']
    dont_care_bit_index : int
        Which field is the wildcard: 0=type, 1=description, 2=source
    
    Returns
    -------
    list[EventPreferencesTuple]
        List of expanded concrete preferences for each matching event value.
    
    Examples
    --------
    >>> # Wildcard in 'type' field (dont_care_bit_index=0)
    >>> base_pref = EventPreferencesTuple(type='*', description='failure', source='pump1', target_sources=['target1'])
    >>> expanded = process_event_preference_with_one_dont_care_bit(base_pref, events_df, 0)
    >>> # Results in preferences for each type matching ('failure', source='pump1')
    """
    result = []

    if dont_care_bit_index == 0:
        filtered_event_data = event_data[(event_data['type'] == event_preference.type) & (event_data['source'] == event_preference.source)]
    elif dont_care_bit_index == 1:
        filtered_event_data = event_data[(event_data['description'] == event_preference.description) & (event_data['source'] == event_preference.source)]
    else: # dont_care_bit_index == 2
        filtered_event_data = event_data[(event_data['description'] == event_preference.description) & (event_data['type'] == event_preference.type)]

    for _, current_event_data_row in filtered_event_data.iterrows():
        result.append(EventPreferencesTuple(description=current_event_data_row.description, type=current_event_data_row.type, source=current_event_data_row.source, target_sources=event_preference.target_sources))


    return result


def process_event_preference_with_two_dont_care_bits(event_preference: EventPreferencesTuple, event_data: pd.DataFrame, dont_care_bit_1_index: int, dont_care_bit_2_index: int) -> list[EventPreferencesTuple]:
    """Expand an event preference rule with two wildcard dimensions.
    
    Replaces two wildcards '*' in an event preference with all matching value
    combinations from event_data, generating multiple concrete preference rules.
    
    Parameters
    ----------
    event_preference : EventPreferencesTuple
        Base event preference with two fields set to '*' (wildcards).
    event_data : pd.DataFrame
        Available event data with columns: ['type', 'source', 'description']
    dont_care_bit_1_index : int
        First wildcard position: 0=type, 1=description, 2=source
    dont_care_bit_2_index : int
        Second wildcard position: 0=type, 1=description, 2=source
    
    Returns
    -------
    list[EventPreferencesTuple]
        List of expanded concrete preferences for each matching combination.
    
    Examples
    --------
    >>> # Wildcards in 'type' and 'source' fields
    >>> base = EventPreferencesTuple(type='*', description='anomaly', source='*', target_sources=['pump1'])
    >>> expanded = process_event_preference_with_two_dont_care_bits(base, events_df, 0, 2)
    """
    result = []

    if dont_care_bit_1_index == 0 and dont_care_bit_2_index == 1:
        filtered_event_data = event_data[event_data['source'] == event_preference.source]
    elif dont_care_bit_1_index == 0 and dont_care_bit_2_index == 2:
        filtered_event_data = event_data[event_data['type'] == event_preference.type]
    else: # dont_care_bit_1_index == 1 and dont_care_bit_2_index == 2
        filtered_event_data = event_data[event_data['description'] == event_preference.description]

    for _, current_event_data_row in filtered_event_data.iterrows():
        result.append(EventPreferencesTuple(description=current_event_data_row.description, type=current_event_data_row.type, source=current_event_data_row.source, target_sources=event_preference.target_sources))


    return result


def process_event_preferences_key(event_data: pd.DataFrame, event_preferences: list[EventPreferencesTuple]) -> list[EventPreferencesTuple]:
    """Expand event preferences by resolving all wildcards.
    
    Processes a list of event preference rules containing wildcards ('*') and expands
    them into concrete rules by matching against available event data. Supports
    wildcard patterns with 0, 1, 2, or 3 don't-care bits.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Available event data with columns: ['type', 'source', 'description']
    event_preferences : list[EventPreferencesTuple]
        Event preference rules with potential wildcards (*).
        Rules are processed for specific patterns:
        - 0 wildcards: Returned as-is (concrete rule)
        - 1 wildcard: Expanded using one don't-care dimension
        - 2 wildcards: Expanded using two don't-care dimensions
        - 3 wildcards: Expands to all events (stops processing remaining rules)
    
    Returns
    -------
    list[EventPreferencesTuple]
        List of fully expanded concrete event preferences with duplicates removed.
    
    Examples
    --------
    >>> events = pd.DataFrame({
    ...     'type': ['critical', 'warning'],
    ...     'source': ['pump1', 'pump2'],
    ...     'description': ['failure', 'anomaly']
    ... })
    >>> prefs = [EventPreferencesTuple('*', 'failure', 'pump1', ['target1'])]
    >>> concrete_prefs = process_event_preferences_key(events, prefs)
    >>> # Returns preferences for all types matching ('failure', 'pump1')
    """
    result_preferences = []
    for current_preference in event_preferences:
        if current_preference.description == '*' and current_preference.type != '*' and current_preference.source != '*':
            result_preferences = result_preferences + process_event_preference_with_one_dont_care_bit(current_preference, event_data, 0)

        elif current_preference.description != '*' and current_preference.type == '*' and current_preference.source != '*':
            result_preferences = result_preferences + process_event_preference_with_one_dont_care_bit(current_preference, event_data, 1)

        elif current_preference.description != '*' and current_preference.type != '*' and current_preference.source == '*':
            result_preferences = result_preferences + process_event_preference_with_one_dont_care_bit(current_preference, event_data, 2)

        # 2 dont care bits
        elif current_preference.description == '*' and current_preference.type == '*' and current_preference.source != '*':
            result_preferences = result_preferences + process_event_preference_with_two_dont_care_bits(current_preference, event_data, 0, 1)

        elif current_preference.description == '*' and current_preference.type != '*' and current_preference.source == '*':
            result_preferences = result_preferences + process_event_preference_with_two_dont_care_bits(current_preference, event_data, 0, 2)

        elif current_preference.description != '*' and current_preference.type == '*' and current_preference.source == '*':
            result_preferences = result_preferences + process_event_preference_with_two_dont_care_bits(current_preference, event_data, 1, 2)

        # 3 dont care bits
        elif current_preference.description == '*' and current_preference.type == '*' and current_preference.source == '*':
            for _, current_event_data_row in event_data.iterrows():
                result_preferences.append(EventPreferencesTuple(description=current_event_data_row.description, type=current_event_data_row.type, source=current_event_data_row.source, target_sources=current_preference.target_sources))

            break # we encountered a preference with 3 dont care bits so no need to continue looping through the rest of the preferences

        else: # 0 dont care bits
            result_preferences.append(current_preference)

    
    return list(set(result_preferences)) # remove duplicates


def expand_event_preferences(event_data: pd.DataFrame, event_preferences: EventPreferences) -> EventPreferences:
    """Expand failure and reset event preferences by resolving wildcards.
    
    Convenience wrapper around process_event_preferences_key that expands both
    'failure' and 'reset' event preference categories.
    
    Parameters
    ----------
    event_data : pd.DataFrame
        Available event data with columns: ['type', 'source', 'description']
    event_preferences : EventPreferences
        Dictionary with keys 'failure' and 'reset', each containing lists of
        EventPreferencesTuple objects with potential wildcards.
    
    Returns
    -------
    EventPreferences
        Dictionary with same structure, but all wildcards expanded into concrete rules.
    
    Examples
    --------
    >>> event_prefs = {
    ...     'failure': [EventPreferencesTuple('*', 'critical', 'pump1', ['target1'])],
    ...     'reset': [EventPreferencesTuple('maintenance', '*', '*', ['target2'])]
    ... }
    >>> expanded = expand_event_preferences(event_data, event_prefs)
    """
    result_event_preferences: EventPreferences = {
        'failure': [],
        'reset': [],
    }

    result_event_preferences['failure'] = process_event_preferences_key(event_data, event_preferences['failure'])
    result_event_preferences['reset'] = process_event_preferences_key(event_data, event_preferences['reset'])


    return result_event_preferences


def calculate_mango_parameters(current_param_space_dict, MAX_JOBS, INITIAL_RANDOM, MAX_RUNS):
    """
    Calculate MANGO optimization parameters (num, jobs, initial_random) based on the current parameter space and constraints.
    """
    if MAX_RUNS <= MAX_JOBS:
        MAX_JOBS = MAX_RUNS
        if MAX_JOBS==1:
            return 0, MAX_JOBS, 1
        else:
            return 1, MAX_JOBS, 1
    
    param_space_size = 1
    for _, item in current_param_space_dict.items():
        param_space_size *= len(item)

    if param_space_size<=MAX_JOBS:
        num=max(1, param_space_size-INITIAL_RANDOM)
        jobs=1
        initial_random=min(INITIAL_RANDOM, param_space_size)
    elif min(MAX_RUNS,param_space_size)%MAX_JOBS <INITIAL_RANDOM:
        initial_random=min(MAX_RUNS,param_space_size)%MAX_JOBS+MAX_JOBS
        num=max(1, min(MAX_RUNS,param_space_size)//MAX_JOBS -1)
        jobs=MAX_JOBS
    else:
        initial_random=min(MAX_RUNS,param_space_size)%MAX_JOBS
        num=max(1, min(MAX_RUNS,param_space_size)//MAX_JOBS)
        jobs=MAX_JOBS

    return num, jobs, initial_random




