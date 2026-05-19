🧬 Implementing Survival Analysis Methods
===========================================

Survival analysis methods model time-to-failure while handling **censored data** (systems that haven't failed yet). They're ideal for real-world scenarios where failure times are unknown for some equipment.

**When to Use:** 
- You have time-to-failure data with censoring indicators
- You want to model both time AND event occurrence jointly
- You need to handle right-censored data (equipment still operational)
- Works only with ``Supervised_SA_PdMExperiment``

**Key Requirement:** Training labels are **tuples** (time, event) where event ∈ {0=censored, 1=failed}


Survival Analysis Concepts
---------------------------

**Survival Data Structure:**

.. code-block:: python

   # Each sample has TWO labels:
   labels = [
       (100, 1),  # Failed at time 100
       (200, 0),  # Censored (still alive) at time 200
       (150, 1),  # Failed at time 150
       (250, 0),  # Censored at time 250
   ]

- **Time**: When failure occurred or when observation ended
- **Event**: 1 if failure occurred, 0 if censored (still operating)

**Cox Proportional Hazards:**

Models the hazard function (instantaneous failure rate):

.. math::

    h(t|X) = h_0(t) \times e^{\beta_1 X_1 + ... + \beta_p X_p}

- Combines baseline hazard with feature effects
- Outputs survival curves: probability of surviving beyond time t


Interface Overview
------------------

Survival methods inherit from ``SupervisedMethodInterface``:

.. code-block:: python

   from pdmlabs.method.supervised_method import SupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences

   class MySurvivalMethod(SupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, **kwargs):
           super().__init__(event_preferences=event_preferences)


**Key Characteristics:**
- Has ``fit()`` method on (time, event) tuples
- Models both failure time AND occurrence
- Returns **structured prediction**: survival curves
- Works with ``Supervised_SA_PdMExperiment`` only


Example: Cox Proportional Hazards
---------------------------------

The ``CoxPH`` is a reference implementation of survival analysis.

**File Location:** ``pdmlabs/method/CoxModel.py``

**What It Does:**
- Fits Cox PH model to time-to-failure data with censoring
- Maintains separate models per data source
- Returns survival function predictions (probability of surviving past each time point)

**Implementation Details:**

.. code-block:: python

   class CoxPH(SupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, 
                    save_model=False, *args, **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.model_per_source = {}
           self.avail_times_per_source = {}  # Store time points for curves
           self.initial_args = args
           self.initial_kwargs = kwargs
           self.save_model = save_model

**Training Phase:**

.. code-block:: python

   def fit(self, historic_data, historic_sources, event_data, anomaly_ranges):
       """
       Train Cox PH model on survival data.
       
       anomaly_ranges structure for survival analysis:
       - List per source
       - Each element is list of (rul, event) tuples
       """
       for data, source, labels in zip(historic_data, historic_sources, anomaly_ranges):
           # Convert labels [(rul1, event1), (rul2, event2), ...] to sklearn format
           from sksurv.util import Surv
           
           rul_times = [lb[0] for lb in labels]
           events = [lb[1] for lb in labels]
           
           y_df = pd.DataFrame({'event': events, 'RUL': rul_times})
           y = Surv.from_dataframe("event", "RUL", y_df)
           
           # Train Cox PH model
           model = CoxPHSurvivalAnalysis(*self.initial_args, **self.initial_kwargs)
           model.fit(data, y)
           
           self.model_per_source[source] = model
           self.avail_times_per_source[source] = np.unique(rul_times)

**Prediction Phase:**

.. code-block:: python

   def predict(self, target_data, source, event_data):
       """
       Predict survival curves for test data.
       
       Returns:
           Array of shape (n_samples, 2, n_timepoints)
           - First slice: survival probabilities
           - Second slice: time points
       """
       model = self.model_per_source[source]
       times = self.avail_times_per_source[source]
       
       # Get survival functions at each time point
       survival_funcs = model.predict_survival_function(target_data)
       
       # Stack into (n, 2, T) format
       result = np.stack([survival_funcs, np.tile(times, ...)], axis=1)
       return result


Creating Your Own Survival Analysis Method
-------------------------------------------

Follow this template:

**Step 1: Create File**

Create ``pdmlabs/method/my_survival_method.py``:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sksurv.linear_model import CoxPHSurvivalAnalysis  # Survival model

   from pdmlabs.method.supervised_method import SupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences


   class MySurvivalMethod(SupervisedMethodInterface):
       """Survival analysis for predictive maintenance with censoring.
       
       This method models time-to-failure while accounting for censored
       observations (equipment that hasn't failed yet).
       """

       def __init__(self, 
                    event_preferences: EventPreferences,
                    alpha: float = 0.05,
                    *args,
                    **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.alpha = alpha  # Confidence interval level
           self.initial_args = args
           self.initial_kwargs = kwargs
           self.model_per_source = {}
           self.available_times_per_source = {}


**Step 2: Implement fit()**

Train on survival data with censoring:

.. code-block:: python

   def fit(self, historic_data: list[pd.DataFrame], 
           historic_sources: list[str], 
           event_data: pd.DataFrame,
           anomaly_ranges: list[list]) -> None:
       """
       Train survival model for each source.
       
       Args:
           historic_data: Training features (one per source)
           historic_sources: Source names
           event_data: Event log
           anomaly_ranges: **Survival tuples** [(time1, event1), (time2, event2), ...]
                          - time: time to failure or censoring
                          - event: 1 if failed, 0 if censored
       
       Censoring example:
       - (100, 1) → Failed at time 100
       - (200, 0) → Still operational at time 200 (censored)
       """
       from sksurv.util import Surv
       
       for data, source, labels in zip(historic_data, historic_sources, anomaly_ranges):
           # Extract times and events from tuples
           times = [lb[0] for lb in labels]
           events = [lb[1] for lb in labels]
           
           # Convert to sksurv format
           y_df = pd.DataFrame({
               'event': events,
               'time': times
           })
           y = Surv.from_dataframe("event", "time", y_df)
           
           # Train survival model
           model = CoxPHSurvivalAnalysis(alpha=self.alpha, *self.initial_args, **self.initial_kwargs)
           model.fit(data, y)
           
           self.model_per_source[source] = model
           self.available_times_per_source[source] = np.unique(np.sort(times))


**Step 3: Implement predict() - Survival Curves**

Predict survival probabilities:

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame):
       """
       Predict survival curves for test samples.
       
       Args:
           target_data: Test features
           source: Source identifier
           event_data: Event log
       
       Returns:
           Array of shape (n_samples, 2, n_timepoints):
           - [:, 0, :] = survival probabilities at each time
           - [:, 1, :] = time points
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model for source '{source}'")
       
       model = self.model_per_source[source]
       times = self.available_times_per_source[source]
       
       # Get survival function (probability of survival at each time point)
       survival_functions = model.predict_survival_function(target_data)
       
       # survival_functions has shape (n_samples, n_timepoints)
       n_samples = survival_functions.shape[0]
       n_times = len(times)
       
       # Stack into output format: (n_samples, 2, n_times)
       # First dimension: survival probabilities
       # Second dimension: time values
       result = np.stack([
           survival_functions.values,  # Survival probabilities
           np.tile(times, (n_samples, 1))  # Repeated times
       ], axis=1)
       
       return result


**Step 4: Implement predict_one() - Single Sample**

Predict survival curve for one sample:

.. code-block:: python

   def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
       """
       Predict survival curve for a single sample.
       
       Args:
           new_sample: Single features as Series
           source: Source identifier
           is_event: Event flag (context)
       
       Returns:
           Survival curve data
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model for source '{source}'")
       
       model = self.model_per_source[source]
       times = self.available_times_per_source[source]
       
       sample_array = new_sample.to_numpy().reshape(1, -1)
       
       survival_func = model.predict_survival_function(sample_array)
       
       n_times = len(times)
       
       result = np.stack([
           survival_func.values.flatten(),
           times
       ], axis=0)
       
       return result


**Step 5: Implement remaining methods**

.. code-block:: python

   def get_params(self) -> dict:
       """Return hyperparameters."""
       first_source = list(self.model_per_source.keys())[0]
       model = self.model_per_source[first_source]
       
       return {
           **model.get_params(),
           'alpha': self.alpha,
       }

   def __str__(self) -> str:
       return 'MySurvivalMethod'

   def get_library(self) -> str:
       return 'no_save'

   def get_all_models(self):
       return self.model_per_source


Survival Data Preparation
--------------------------

**Creating Survival Labels from Event Data:**

If you have event timestamps, compute survival tuples:

.. code-block:: python

   def compute_survival_label(sample_time, failure_time, observation_end):
       """
       Create survival label (time, event).
       
       Args:
           sample_time: When sample was collected
           failure_time: When failure occurred (or None if not yet happened)
           observation_end: When observation window ended
       
       Returns:
           (time_to_event, event_indicator)
       """
       if failure_time is not None:
           # Equipment failed - record time to failure
           time_to_event = failure_time - sample_time
           return (time_to_event, 1)  # event=1 (failure occurred)
       else:
           # Equipment still operational - censored
           time_to_event = observation_end - sample_time
           return (time_to_event, 0)  # event=0 (censored/right-censored)


Testing Your Implementation
----------------------------

With your survival dataset prepared, test your custom survival method using ``run_experiment``:

.. code-block:: python

   from pdmlabs.utils.dataset import Dataset
   from pdmlabs.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment
   from pdmlabs.RunExperiment import run_experiment
   from my_survival_method import MySurvivalMethod
   from pdmlabs.pdm_evaluation_types.types import EventPreferences
   
   # 1. Load data (must have survival tuples with censoring indicators)
   df = pd.read_csv('your_survival_data.csv')
   dataset_handler = Dataset(
       data=df,
       datetime_column="timestamp",
       source_column="source",
       train_sources=0.6,
       val_sources=0.2,
       test_sources=0.2
   )
   ds_sa, _ = dataset_handler.get_SA_dataset()
   
   # 2. Define hyperparameters for your survival method
   method_param_space = {
       'alpha': [0.01, 0.05, 0.1],
   }
   
   
   # 3. Run experiment with run_experiment
   best_params = run_experiment(
       dataset=ds_sa,
       methods=[MySurvivalMethod],
       param_space_dict_per_method=[method_param_space],
       method_names=['MySurvivalMethod'],
       experiments=[Supervised_SA_PdMExperiment],
       experiment_names=['Survival Analysis'],
       MAX_RUNS=12,
       MAX_JOBS=2,
       INITIAL_RANDOM=2,
       profile_size=10,
       optimization_param='C_index',
       maximize=True
   )
   
   # 5. Check results
   print(f"Best parameters: {best_params[0]}")


Next Steps
----------

- Review ``CoxModel`` (CoxPH) in ``pdmlabs/method/CoxModel.py``
- Explore sksurv documentation (https://scikit-survival.readthedocs.io/)
- Check ``Supervised_SA_PdMExperiment`` in ``pdmlabs/experiment/batch/``
- Review dataset SA preparation in ``pdmlabs/utils/dataset.py::get_SA_dataset()``
