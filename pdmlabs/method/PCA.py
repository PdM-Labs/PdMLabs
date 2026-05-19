import pandas as pd
import sys
from pdmlabs.utils import utils
import numpy as np
from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences
from pdmlabs.method.pca_tsb import PCA as PCAcore


class PCA_semi(SemiSupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, sub_sequence_length=1, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)

        # This is used only in case of univariate data.
        self.sub_sequence_length = sub_sequence_length
        self.model_per_source={}
        self.sub_len_per_source={}


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            if len(current_historic_data.columns)>1:
                X_data = current_historic_data.values
            else:
                if len(current_historic_data.index)<self.sub_sequence_length:
                    X_data = utils.Window(window=len(current_historic_data.index)-1).convert(
                        current_historic_data[0].values).to_numpy()
                    self.sub_len_per_source[current_historic_source]=len(current_historic_data.index)-1
                else:
                    X_data = utils.Window(window=self.sub_sequence_length).convert(current_historic_data[0].values).to_numpy()
                    self.sub_len_per_source[current_historic_source]=self.sub_sequence_length

            self.model_per_source[current_historic_source] = PCAcore(random_state=42)
            self.model_per_source[current_historic_source].fit(X_data)


    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        if len(target_data.columns) > 1:
            X_data = target_data.values
        else:
            if len(target_data.index)<self.sub_len_per_source[source]:
                return [0.0 for i in range(len(target_data.index))]
            X_data = utils.Window(window=self.sub_len_per_source[source]).convert(target_data[0].values).to_numpy()

        scores=self.model_per_source[source].decision_function(X_data)
        scores=[min(scores) for i in range(len(target_data.index)-len(scores))]+[sc for sc in scores]
        maxx=-1
        for sc in scores:
            if sc==float("inf") or sc>sys.float_info.max:
                continue
            if sc >maxx:
                maxx = sc
        scores=[sc if sc<maxx else maxx for sc in scores]
        return scores
    
    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return -self.model_per_source[source].score_samples([new_sample.to_numpy()]).tolist()[0]

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'PCA'

    def get_params(self) -> dict:
        return {
            "sub_sequence_length": self.sub_sequence_length,
        }

    def get_all_models(self):
        pass