from sklearn.neighbors import LocalOutlierFactor as local_outlier_factor
import pandas as pd
import mlflow

from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class LocalOutlierFactor(SemiSupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs

        if 'profile_size' in kwargs:
            del self.initial_kwargs['profile_size']
        
        self.clf_class = local_outlier_factor
        self.model_per_source = {}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            self.model_per_source[current_historic_source] = self.clf_class(novelty=True, *(self.initial_args), **(self.initial_kwargs))
            self.model_per_source[current_historic_source].fit(current_historic_data)
        

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        # TODO need to check if a model is available for the provided source
        return (-self.model_per_source[source].score_samples(target_data)).tolist()


    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return -self.model_per_source[source].score_samples([new_sample.to_numpy()]).tolist()[0]
    

    def get_library(self) -> str:
        return 'no_save'
    
    
    def __str__(self) -> str:
        return 'LocalOutlierFactor'


    def get_params(self) -> dict:
        return self.model_per_source[list(self.model_per_source.keys())[0]].get_params()


    def get_all_models(self):
        pass