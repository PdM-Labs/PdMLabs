from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
import numpy as np

from pdmlabs.method.supervised_method import SupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class CoxPH(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, save_model=False, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.avail_times_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model = save_model

    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges: list[list]) -> None:
        """
        This method is used to fit a anomaly detection model in supervised way (training), where the data are passed in form
        of Dataframes along with their respected source and labels.

        :param historic_data: a list of Dataframes (used to fit a semi-supervised model). The `historic_data` list parameter elements should be copied if a corresponding method needs to store them for future processing
        :param historic_sources: a list with strings (names) of the different sources
        :param event_data: event data that are produced from the different sources
        :param anomaly_ranges: labels regarding if the data are normal or not. It is a list of lists, where each inner list corresponds to a source and contains the labels for the data in that source.
        :return: None.
        """

        for current_historic_data, current_historic_source, labels in zip(historic_data, historic_sources,
                                                                          anomaly_ranges):
            print(current_historic_data.shape)
            from sksurv.util import Surv
            ydf=pd.DataFrame({'event': [lb[1] for lb in labels],'RUL': [lb[0] for lb in labels]})
            y = Surv.from_dataframe("event", "RUL", ydf)

            # RandomSurvivalForest(n_estimators=100, min_samples_split=6, min_samples_leaf=5, verbose=1, n_jobs=4)
            self.model_per_source[current_historic_source] = CoxPHSurvivalAnalysis(*self.initial_args,**self.initial_kwargs)
            self.model_per_source[current_historic_source].fit(current_historic_data, y)
            self.avail_times_per_source[current_historic_source]=np.unique([ty for ty in y['RUL']])

            if self.save_model:
                import pickle
                with open(f"xgboost_model_{current_historic_source}.pkl", "wb") as f:
                    pickle.dump(self.model_per_source[current_historic_source], f)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame):
        # TODO need to check if a model is available for the provided source
        predictions = self.model_per_source[source].predict_survival_function(target_data, True)
        n, T = predictions.shape

        # Repeat the time array for every curve → shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))

        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        pass

    def get_params(self) -> dict:
        params = {}
        for i, arg in enumerate(self.initial_args):
            params[f"arg{i}"] = arg
        # include keyword args normally
        params.update(self.initial_kwargs)

        return params

    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "CoxPH"

    def get_all_models(self):
        pass