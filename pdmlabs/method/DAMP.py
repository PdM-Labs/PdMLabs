import argparse
import os
import statistics
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

#score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
from pdmlabs.exceptions.exception import NotFitForSourceException
from pdmlabs.pdm_evaluation_types.types import EventPreferences
import pandas as pd
import warnings


class Damp(UnsupervisedMethodInterface):
    """
    For multivariate data we run Damp on each dimension and then combine results using average or maximum.

    In the future, we can use multivariate sub-sequence distances.
    """

    def __init__(self,
                 event_preferences: EventPreferences,
                 sub_sequence_length: int,
                 init_length: int,
                 stride: int = 1,
                 aggregation_strategy='avg',
                 *args,
                 **kwargs
                 ):
        super().__init__(event_preferences=event_preferences)
        self.sub_sequence_length = sub_sequence_length
        self.init_length = init_length
        self.stride = stride
        self.aggregation_strategy = aggregation_strategy

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        scores_per_column = []
        for col in target_data.columns:
            scores, _, _ = DAMP_2_0(
                time_series=target_data[col].values,
                subsequence_length=self.sub_sequence_length,
                stride=self.stride,
                location_to_start_processing=self.init_length)
            score = [scores[0]] * (target_data.shape[0] - len(scores)) + list(scores)
            scores_per_column.append(score)
        final_scores = self._combinescores(scores_per_column, target_data)
        return final_scores

    def _combinescores(self, all_scores, target_data):
        finalscores = []
        if self.aggregation_strategy == "max":
            for i in range(len(target_data.index)):
                avg_sc = max([sc[i] for sc in all_scores]) / len(all_scores)
                finalscores.append(avg_sc)
        else:
            for i in range(len(target_data.index)):
                avg_sc = sum([sc[i] for sc in all_scores]) / len(all_scores)
                finalscores.append(avg_sc)
        return finalscores

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'DAMP'

    def get_params(self) -> dict:
        return {
            "sub_sequence_length": self.sub_sequence_length,
            "stride": self.stride,
            "init_length": self.init_length,
            'aggregation_strategy': self.aggregation_strategy,
        }

    def get_all_models(self):
        pass


def DAMP_2_0(
        time_series: np.ndarray,
        subsequence_length: int,
        stride: int,
        location_to_start_processing: int,
) -> Tuple[np.ndarray, float, int]:
    """Computes DAMP of a time series.
    Website: https://sites.google.com/view/discord-aware-matrix-profile/home
    Algorithm: https://drive.google.com/file/d/1FwiLHrgoOUOTHeIXHAFgy2flQ1alRSoN/view

    Args:
        time_series (np.ndarray): Univariate time series
        subsequence_length (int): Window size
        stride (int): Window stride
        location_to_start_processing (int): Start/End index of test/train set
        lookahead (int, optional): How far to look ahead for pruning. Defaults to 0.
        enable_output (bool, optional): Print results and save plot. Defaults to True.

    Raises:
        Exception: See code.
        Description: https://docs.google.com/presentation/d/1_-LGilUJpYRbRZpitw05EgkiOZX52kRd/edit#slide=id.p11

    Returns:
        Tuple[np.ndarray, float, int]: Matrix profile, discord score and its corresponding position in the profile
    """
    assert (subsequence_length > 10) and (
            subsequence_length <= 1000
    ), "`subsequence_length` must be > 10 or <= 1000."

    # # Lookahead indicates how long the algorithm has a delay
    # if lookahead is None:
    #     lookahead = int(2 ** nextpow2(16 * subsequence_length))
    # elif (lookahead != 0) and (lookahead != 2 ** nextpow2(lookahead)):
    #     lookahead = int(2 ** nextpow2(lookahead))

    # Handle invalid inputs
    # 1. Constant Regions
    # if contains_constant_regions(
    #     time_series=time_series, subsequence_length=subsequence_length
    # ):
    #     raise Exception(
    #         "ERROR: This dataset contains constant and/or near constant regions.\nWe define the time series with an overall variance less than 0.2 or with a constant region within its sliding window as the time series containing constant and/or near constant regions.\nSuch regions can cause both false positives and false negatives depending on how you define anomalies.\nAnd more importantly, it can also result in imaginary numbers in the calculated Left Matrix Profile, from which we cannot get the correct score value and position of the top discord.\n** The program has been terminated. **"
    #     )

    # 2. Location to Start Processing
    if (location_to_start_processing / subsequence_length) < 4:
        print(
            "WARNING: location_to_start_processing/subsequence_length is less than four.\nWe recommend that you allow DAMP to see at least four cycles, otherwise you may get false positives early on.\nIf you have training data from the same domain, you can prepend the training data, like this Data = [trainingdata, testdata], and call DAMP(data, S, length(trainingdata))"
        )
        if location_to_start_processing < subsequence_length:
            print(
                f"location_to_start_processing cannot be less than the subsequence length.\nlocation_to_start_processing has been set to {subsequence_length}"
            )
            location_to_start_processing = subsequence_length
        print("------------------------------------------\n\n")
    else:
        if location_to_start_processing > (len(time_series) - subsequence_length + 1):
            print(
                "WARNING: location_to_start_processing cannot be greater than length(time_series)-S+1"
            )
            location_to_start_processing = len(time_series) - subsequence_length + 1
            print(
                f"location_to_start_processing has been set to {location_to_start_processing}"
            )
            print("------------------------------------------\n\n")

    # 3. Subsequence length
    # Subsequence length recommendation based on a peak-finding algorithm TBD.

    # Initialization
    # This is a special Matrix Profile, it only looks left (backwards in time)
    left_mp = np.zeros(time_series.shape)

    # The best discord score so far
    best_so_far = -np.inf

    # A Boolean vector where 1 means execute the current iteration and 0 means skip the current iteration
    bool_vec = np.ones(len(time_series))

    # Handle the prefix to get a relatively high best so far discord score
    pos_pre = 0
    time_series_clean = []
    for i in range(
            location_to_start_processing - 1,
            len(time_series),
            stride,
    ):
        # Skip the current iteration if the corresponding boolean value is 0, otherwise execute the current iteration
        if not bool_vec[i]:
            left_mp[i] = left_mp[i - 1] - 1e-05
            continue

        # Use the brute force for the left Matrix Profile value
        if i + subsequence_length - 1 > len(time_series):
            break

        ### code by agiannoul  to handle costant subsequencies nan values
        pre_len=len(time_series_clean)
        if len(time_series_clean)<1:
            new_x = [v for v in time_series[: subsequence_length- 1]]
            for qi in range(subsequence_length, len(time_series[:i])):
                subs = time_series[qi - subsequence_length:qi]
                if sum(subs) == 0 or statistics.stdev(subs) < 0.2:
                    continue
                else:
                    new_x.append(time_series[qi])
            time_series_clean = np.array(new_x)
            pos_pre=i
        else:
            for qi in range(pos_pre, i):
                subs = time_series[qi - subsequence_length:i]
                if sum(subs) == 0 or statistics.stdev(subs) < 0.2:
                    continue
                else:
                    time_series_clean=np.append(time_series_clean,np.array([time_series[qi]]))
            pos_pre=i



        query = time_series[i: i + subsequence_length]
        if np.std(query) < 0.01:
            left_mp[i] = left_mp[min(0,i-1)]
        else:
            res=MASS_V2(time_series_clean, query)

            res=np.array([0 if np.isnan(r) else r for r in res ])
            res=np.array([max(res) if np.isnan(r) else r for r in res ])
            res=np.amin(res)
            # if np.isnan(res):
            #     ok="ok"
            #     res=0
            left_mp[i] = res
        #####################################################

        # Update the best so far discord score
        best_so_far = np.amax(left_mp)

        # If lookahead is 0, then it is a pure online algorithm with no pruning
        # if lookahead != 0:
        #     # Perform forward MASS for pruning
        #     # The index at the beginning of the forward mass should be avoided in the exclusion zone
        #     start_of_mass = min(i + subsequence_length - 1, len(time_series))
        #     end_of_mass = min(start_of_mass + lookahead - 1, len(time_series))
        #
        #     # The length of lookahead should be longer than that of the query
        #     if (end_of_mass - start_of_mass + 1) > subsequence_length:
        #         distance_profile = MASS_V2(
        #             time_series[start_of_mass : end_of_mass + 1], query
        #         )
        #
        #         # Find the subsequence indices less than the best so far discord score
        #         dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]
        #
        #         # Converting indexes on distance profile to indexes on time series
        #         ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass
        #
        #         # Update the Boolean vector
        #         bool_vec[ts_index_less_than_BSF] = 0

    # Get pruning rate
    # PV = bool_vec[
    #     location_to_start_processing - 1 : len(time_series) - subsequence_length + 1
    # ]
    #PR = (len(PV) - sum(PV)) / len(PV)

    # Get top discord
    discord_score, position = np.amax(left_mp), np.argmax(left_mp)
    # print("\nResults:")
    # print(f"Pruning Rate: {PR}")
    # print(f"Predicted discord score/position: {discord_score} / {position}")

    return left_mp, discord_score, position


def MASS_V2(x=None, y=None):
    # x is the data, y is the query

    # exclude constant regions from x:

    m = len(y)
    n = len(x)

    # Compute y stats -- O(n)
    meany = np.mean(y)
    sigmay = np.std(y)

    # Compute x stats
    x_less_than_m = x[: m - 1]
    divider = np.arange(1, m, dtype=float)
    cumsum_ = x_less_than_m.cumsum()
    square_sum_less_than_m = (x_less_than_m ** 2).cumsum()
    mean_less_than_m = cumsum_ / divider
    std_less_than_m = np.sqrt(
        (square_sum_less_than_m - (cumsum_ ** 2) / divider) / divider
    )

    windows = np.lib.stride_tricks.sliding_window_view(x, m)
    mean_greater_than_m = windows.mean(axis=1)
    std_greater_than_m = windows.std(axis=1)

    meanx = np.concatenate([mean_less_than_m, mean_greater_than_m])
    sigmax = np.concatenate([std_less_than_m, std_greater_than_m])

    y = y[::-1]
    y = np.concatenate((y, [0] * (n - m)))

    # The main trick of getting dot products in O(n log n) time
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    Z = np.multiply(X, Y)
    z = np.fft.ifft(Z).real

    dist = 2 * (
            m - (z[m - 1: n] - m * meanx[m - 1: n] * meany) / (sigmax[m - 1: n] * sigmay)
    )
    dist = np.sqrt(dist)
    return dist
