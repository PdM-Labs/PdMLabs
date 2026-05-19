# Supervised flavor

## Classification
We will use the `XGBoost` method to run a classification experiment on a dataset.
```
pip install xgboost
```

```python
from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
from pdmlabs.RunExperiment import run_experiment
experiments = [SupervisedPdMExperiment]
experiment_names = ['My classification experiment']

from pdmlabs.method.xgboost import XGBoost

methods = [XGBoost]
param_space_dict_per_method = [{}]
method_names = ["XGBoost"]

# generate labels for classification training
from pdmlabs.loadAnomalyDetectionDataset import generate_labels
anomaly_ranges,_=generate_labels(dataset)
dataset["anomaly_labels"] = anomaly_ranges

run_experiment(dataset, methods, param_space_dict_per_method, method_names,
                                            experiments, experiment_names,
                                            MAX_RUNS=4, MAX_JOBS=1, INITIAL_RANDOM=1)
```

## Dataset

For supervised experiments the dataset should have `"anomaly_labels"` field,
which is a list of 0s and 1s, where 1 means that the sample is anomalous.
This list should have the same size as the `dataset["historic_data"]`,
and each of its components should correspond to the same sample in the dataset,
with length being equal to the shape of the corresponding dataframe of "historic_data".

The `"anomaly_labels"` field can be generated using the `generate_anomaly_labels` function
```python
from pdmlabs.loadAnomalyDetectionDataset import generate_labels
anomaly_ranges,_=generate_labels(dataset)
dataset["anomaly_labels"] = anomaly_ranges
```
The passing dataset should have `"historic_data"`, `"predictive_horizon"` and `"lead"` fields.

In case we want to make predictions for different sources than the ones used for training,
we have to define `dataset["match_sources"]` which is a dictionary.

For example if in historical data there is `"sourceA"` and we want to make predictions for `"sourceB"` using,
`"sourceA"`, then we define:
```python
dataset["match_sources"] = {"sourceB": "sourceA"}
```


## Regression

Now we will use the `XGBoost` method to run a regression experiment on a dataset, where we try to predict the Remaining Useful Life (RUL) of a machine.

RUL can be expressed in terms of operational cycles, or percentage of remaining life.

We can generate the RUL labels using the `generate_rul_labels` function (similar as done for classification).
This method except dataset, uses cut_off (default is 0) and a boolean parameter `percentage` (default is False).
- `cut_off` defines the point in time from which we start counting the RUL (using the cut_off value for the past timestamps).
This parameter can be defined by providing int, or float value smaller than 1 (used to define cut_off as percentage of the historical episode).
- `percentage` defines whether the RUL is expressed in percentage (default is False, meaning RUL is expressed in cycles).

**NOTE:** For now we revert the predictions of RUL models (i.e. using max()-prediction) to make them compatible with the classification experiments evaluation.
This will change in the future, by supporting dedicated evaluation for regression experiments. 
In the future in case we want to evaluate the RUL experiment using metrics of classifications experiments, the reversion will not be needed, and will be handled
by the evaluation function.

```python

Example:

```python
from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
from pdmlabs.RunExperiment import run_experiment
experiments = [SupervisedPdMExperiment]
experiment_names = ['My RUL experiment']

from xgboostRUL import XGBoostRUL # not implemented in pdmlabs, but provided as an example below

methods = [XGBoostRUL]
param_space_dict_per_method = [{}]
method_names = ["XGBoost"]

# generate labels for classification training
from pdmlabs.loadAnomalyDetectionDataset import generate_RUL_labels
rul_labels=generate_RUL_labels(dataset)
dataset["anomaly_labels"] = rul_labels

run_experiment(dataset, methods, param_space_dict_per_method, method_names,
                                            experiments, experiment_names,
                                            MAX_RUNS=4, MAX_JOBS=1, INITIAL_RANDOM=1)
```

### Dataset requirements

Dataset should have the following fields:
- `match_sources` : optional, in case we want to map target sources to historic sources.
- `target_sources` : list of strings, names of the different target sources.
- `anomaly_labels` : list of lists, where each inner list corresponds to a source and contains the RUL labels for the data in that source (for historical/training data).
- `historic_data` : list of DataFrames, where each DataFrame corresponds to the historical/training data of a source.
- `historic_sources` : list of strings, names of the different historical/training sources.
- `target_data` : list of DataFrames, where each DataFrame corresponds to the target/testing data of a source.
- `target_sources` : list of strings, names of the different target/testing sources.
- `target_labels` : list of lists, where each inner list corresponds to a source and contains the RUL labels for the data in that source (for target/testing data).

### Method Implementation
XGBoostRUL is a method that uses XGBoost for regression tasks.
Implementation example:
```python
import pandas as pd
import xgboost as xgb
from pdmlabs.method.supervised_method import SupervisedMethodInterface
from pdmlabs.pdm_evaluation_types.types import EventPreferences


class XGBoostRUL(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs

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
            self.model_per_source[current_historic_source] = xgb.XGBRegressor(*self.initial_args,
                                                                               **self.initial_kwargs)
            self.model_per_source[current_historic_source].fit(current_historic_data, labels)

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
        # TODO need to check if a model is available for the provided source
        predictions= self.model_per_source[source].predict(target_data)[:].tolist()
        predictions= [max(predictions)-x for x in predictions]
        return predictions

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        return self.model_per_source[source].predict_proba([new_sample.to_numpy()])[:, 1].tolist()[0]

    def get_params(self) -> dict:
        return {
            **(xgb.XGBRegressor(novelty=False, *(self.initial_args), **(self.initial_kwargs)).get_params()),
        }
    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "XGBOOST"
    def get_all_models(self):
        pass
```