from pdmlabs.method.PCA import PCA_semi
import pandas as pd

from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class PCA_uns(UnsupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences,window,n_bins=10, alpha=0.1, tol=0.5, sub_sequence_length=1, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs
        # This is used only in case of univariate data.
        self.sub_sequence_length = sub_sequence_length
        self.window = window
        self.semi_model=PCA_semi(event_preferences=event_preferences, sub_sequence_length=self.sub_sequence_length)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        first = True
        scores=[]
        for i in range(len(target_data.index)):
            if first:
                wdf = target_data.iloc[0:min(len(target_data.index), self.window + i * self.window)]
                first = False
            else:
                wdf = target_data.iloc[i * self.window:min(len(target_data.index),self.window + i * self.window)]
            self.semi_model.fit([wdf],[source],event_data)
            w_scores=self.semi_model.predict(wdf,source,event_data)
            scores.extend(w_scores)
            if self.window+i*self.window>=len(target_data.index):
                break
        return scores
    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return 0

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'PCA_uns'

    def get_params(self) -> dict:
        return {
            "sub_sequence_length": self.sub_sequence_length,
            'window': self.window,
        }

    def get_all_models(self):
        pass