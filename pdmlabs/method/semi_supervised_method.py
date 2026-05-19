import abc

import pandas as pd

from pdmlabs.pdm_evaluation_types.types import EventPreferences
from pdmlabs.method.method import MethodInterface

class SemiSupervisedMethodInterface(MethodInterface):
    @abc.abstractmethod
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame) -> None:
        """
        This method is used to fit a anomaly detection model in (relative) normal data, where the data are passed in form
        of Dataframes along with their respected source.

        :param historic_data: a list of Dataframes (used to fit a semi-supervised model). The `historic_data` list parameter elements should be copied if a corresponding method needs to store them for future processing
        :param historic_sources: a list with strings (names) of the different sources
        :param event_data: event data that are produced from the different sources
        :return: None.
        """

        pass