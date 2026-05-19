import pandas as pd
from pdmlabs.method.distance_based_k_r_Core import distance_based_k_r
from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences
from pdmlabs.exceptions.exception import NotFitForSourceException

import warnings
class Distance_Based_Semi(SemiSupervisedMethodInterface):
    def __init__(self,
                 event_preferences: EventPreferences,
                 k: int,
                 metric:str='euclidean',
                 window_norm=False,
                 *args,
                 **kwargs
                 ):
        super().__init__(event_preferences=event_preferences)
        self.k = k
        self.metric = metric
        self.window_norm=window_norm
        self.model_per_source={}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        """
        If the length of data in fit is smaller than k then the method will set the k equal to len(fitted_data)-1.

        :param historic_data:
        :param historic_sources:
        :param event_data:
        :return:
        """
        for df,source in zip(historic_data,historic_sources):
            if df.shape[0]<=3:
                continue
            self.model_per_source[source]=distance_based_k_r(k=self.k,window_norm=self.window_norm)
            self.model_per_source[source].fit(df.copy().values)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if source in self.model_per_source.keys():
            pos=0
            scores=[]
            for i in range(0,len(target_data.values),1000):
                temp_scores=self.model_per_source[source].predict(target_data.values[pos:i])
                scores.extend(temp_scores)
                pos=i
            if pos<len(target_data.values):
                temp_scores=self.model_per_source[source].predict(target_data.values[pos:])
                scores.extend(temp_scores)
            return scores
        else:
            raise NotFitForSourceException()

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'Distance Based k (semi)'

    def get_params(self) -> dict:
        return {
            'k': self.k,
            'window_norm': self.window_norm,
            'metric': self.metric,
        }

    def get_all_models(self):
        pass