import torch
from pdmlabs.method.cnn_inner import CNN
from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences
import pandas as pd

def run_CNN(data_train, data_test, window_size=100, num_channel=[32, 32, 40], lr=0.0008, n_jobs=1):

    clf = CNN(window_size=window_size, num_channel=num_channel, feats=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()





class Cnn(SemiSupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences,
                 window_size=100, num_channel=[32, 32, 40], lr=0.0008, batch_size=128
                 , n_jobs=1,
                 *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.window_size=window_size
        self.num_channel=num_channel
        self.lr=lr
        self.batch_size=batch_size
        if 'profile_size' in kwargs:
            del self.initial_kwargs['profile_size']

        self.clf_class = CNN
        self.model_per_source = {}

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        for current_historic_data, current_historic_source in zip(historic_data, historic_sources):
            if current_historic_data.shape[0] <= 3:
                continue
            torch.set_default_dtype(torch.float32)
            self.model_per_source[current_historic_source] = self.clf_class(window_size=self.window_size, num_channel=self.num_channel, feats=current_historic_data.shape[1], lr=self.lr, batch_size=self.batch_size)
            self.model_per_source[current_historic_source].fit(current_historic_data.values.astype('float32'))

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        # TODO need to check if a model is available for the provided source
        torch.set_default_dtype(torch.float32)
        score = self.model_per_source[source].decision_function(target_data.values.astype('float32'))
        return [s for s in score.ravel()]

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        # TODO need to keep buffer until profile size are encountered and then start predicting
        return -self.model_per_source[source].score_samples([new_sample.to_numpy()]).tolist()[0]

    def get_library(self) -> str:
        return 'no_save'

    def __str__(self) -> str:
        return 'IsolationForest'

    def get_params(self) -> dict:
        return {"window_size":self.window_size,
        "num_channel":self.num_channel,
        "lr":self.lr,
        "batch_size":self.batch_size}

    def get_all_models(self):
        pass