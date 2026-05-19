⏱️ Implementing RUL Regression Methods
=======================================

RUL (Remaining Useful Life) regression methods predict how much time or cycles remain before failure. Instead of binary classification, they predict a continuous target: time-to-failure.

**When to Use:** When you have training data with exact failure times/cycles. The method learns to predict RUL as a regression target. Works only with ``SupervisedRULPdMExperiment``.

**Key Requirement:** Training labels are **continuous values** (days, cycles, hours until failure) not binary classes.


Interface Overview
------------------

RUL methods inherit from ``SupervisedMethodInterface`` (same as classification):

.. code-block:: python

   from pdmlabs.method.supervised_method import SupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences

   class MyRULMethod(SupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, **kwargs):
           super().__init__(event_preferences=event_preferences)
           # Your initialization


**Key Characteristics:**
- Has ``fit()`` method (training on labeled data)
- **Regression** target: continuous RUL values (not 0/1 classification)
- Returns RUL predictions as scores
- Works with ``SupervisedRULPdMExperiment`` only


Required Methods
----------------

All RUL methods implement the same interface as classification, but:

1. ``fit()`` receives continuous RUL labels, not binary
2. ``predict()`` returns continuous RUL values, not probabilities
3. The evaluation metrics differ (MAE, MSE instead of ROC-AUC)


Example: XGBoost RUL Regression
-------------------------------

The ``XGBoostRUL`` is a reference implementation of RUL regression.

**File Location:** ``pdmlabs/method/xgboostRUL.py``

**What It Does:**
- Trains XGBoost regressor to predict remaining useful life
- Maintains separate regressors per data source
- Returns RUL predictions as scores

**Implementation Details:**

.. code-block:: python

   class XGBoostRUL(SupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, 
                    save_model=False, *args, **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.model_per_source = {}
           self.initial_args = args
           self.initial_kwargs = kwargs
           self.save_model = save_model

**Training Phase:**

.. code-block:: python

   def fit(self, historic_data: list[pd.DataFrame], 
           historic_sources: list[str], 
           event_data: pd.DataFrame,
           anomaly_ranges: list[list]) -> None:
       """
       Train XGBoost regressor on RUL data.
       
       Args:
           historic_data: Training features (one DataFrame per source)
           historic_sources: Source identifiers
           event_data: Event log
           anomaly_ranges: **RUL values** (not binary labels!)
                          - list per source
                          - each element is continuous RUL value
       """
       for data, source, rul_labels in zip(historic_data, historic_sources, anomaly_ranges):
           # Create REGRESSOR (not classifier!)
           model = xgb.XGBRegressor(*self.initial_args, **self.initial_kwargs)
           model.fit(data, rul_labels)
           self.model_per_source[source] = model
           
           # Optional: Save model to disk
           if self.save_model:
               import pickle
               with open(f"model_{source}.pkl", "wb") as f:
                   pickle.dump(model, f)

**Key Difference:** Uses ``XGBRegressor`` not ``XGBClassifier``

**Prediction Phase:**

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
       """Score test data as RUL predictions."""
       model = self.model_per_source[source]
       # predict() returns continuous values (not probabilities!)
       predictions = model.predict(target_data)
       return predictions.tolist()


Creating Your Own RUL Regression Method
----------------------------------------

Follow this template:

**Step 1: Create File**

Create ``pdmlabs/method/my_rul_regressor.py``:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.ensemble import RandomForestRegressor  # Regression algorithm

   from pdmlabs.method.supervised_method import SupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences


   class MyRULRegressor(SupervisedMethodInterface):
       """RUL regression for predictive maintenance.
       
       This method learns to predict Remaining Useful Life (RUL) as a
       continuous regression target.
       """

       def __init__(self, 
                    event_preferences: EventPreferences,
                    n_estimators: int = 100,
                    max_depth: int = 15,
                    *args,
                    **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.n_estimators = n_estimators
           self.max_depth = max_depth
           self.initial_args = args
           self.initial_kwargs = kwargs
           self.model_per_source = {}


**Step 2: Implement fit()**

Train regressors on RUL data:

.. code-block:: python

   def fit(self, historic_data: list[pd.DataFrame], 
           historic_sources: list[str], 
           event_data: pd.DataFrame,
           anomaly_ranges: list[list]) -> None:
       """
       Train RUL regressor for each source.
       
       Args:
           historic_data: Training features (one per source)
           historic_sources: Source names
           event_data: Event log (optional reference)
           anomaly_ranges: **Continuous RUL values** (e.g., [100, 200, 50, 30, ...])
       
       The labels represent time (days, cycles, hours) until failure.
       A sample with RUL=100 means it has 100 time units remaining.
       """
       for data, source, rul_values in zip(historic_data, historic_sources, anomaly_ranges):
           # Create REGRESSOR (not classifier)
           regressor = RandomForestRegressor(
               n_estimators=self.n_estimators,
               max_depth=self.max_depth,
               *self.initial_args,
               **self.initial_kwargs
           )
           
           # Train on RUL targets
           regressor.fit(data, rul_values)
           self.model_per_source[source] = regressor


**Step 3: Implement predict()**

Return RUL predictions:

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
       """
       Predict RUL for test data.
       
       Args:
           target_data: Test features
           source: Source identifier
           event_data: Event log (optional)
       
       Returns:
           List of RUL predictions (continuous values)
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model for source '{source}'")
       
       regressor = self.model_per_source[source]
       
       # predict() returns continuous values
       rul_predictions = regressor.predict(target_data)
       
       # Ensure positive RUL values (clamp at 0 if needed)
       rul_predictions = np.maximum(rul_predictions, 0)
       
       return rul_predictions.tolist()


**Step 4: Implement predict_one()**

Predict RUL for single sample:

.. code-block:: python

   def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
       """
       Predict RUL for a single sample.
       
       Args:
           new_sample: Single features as Series
           source: Source identifier
           is_event: Event flag (context)
       
       Returns:
           RUL prediction (continuous value)
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model for source '{source}'")
       
       regressor = self.model_per_source[source]
       sample_array = new_sample.to_numpy().reshape(1, -1)
       
       rul_prediction = regressor.predict(sample_array)[0]
       
       # Ensure positive RUL
       rul_prediction = max(rul_prediction, 0.0)
       
       return float(rul_prediction)


**Step 5: Implement get_params() and other methods**

.. code-block:: python

   def get_params(self) -> dict:
       """Return hyperparameters."""
       first_source = list(self.model_per_source.keys())[0]
       model = self.model_per_source[first_source]
       
       return {
           **model.get_params(),
           'n_estimators': self.n_estimators,
           'max_depth': self.max_depth,
       }

   def __str__(self) -> str:
       return 'MyRULRegressor'

   def get_library(self) -> str:
       return 'no_save'

   def get_all_models(self):
       return self.model_per_source


RUL Data Preparation
---------------------

**Computing RUL from Time-to-Failure:**

If you have time-to-failure information, compute RUL for each sample:

.. code-block:: python

   def compute_rul(failure_time, sample_time):
       """
       Compute RUL as time until failure.
       
       Args:
           failure_time: Time when failure occurs (datetime or numeric)
           sample_time: Time when sample was collected
       
       Returns:
           RUL in time units (days, cycles, hours)
       """
       rul = failure_time - sample_time
       return max(rul, 0)  # RUL cannot be negative
   
   # Example: Compute RUL for Beijing dataset
   df['rul'] = df.groupby('bearing_id')['time_to_failure'].transform(
       lambda x: range(len(x), 0, -1)
   )


**RUL Transformation Patterns:**

1. **Linear degradation:**

   .. code-block:: python

      df['rul'] = df.groupby('source').cumcount(ascending=False)

2. **Exponential weighting (early samples more important):**

   .. code-block:: python

      df['rul'] = df.groupby('source').cumcount(ascending=False).pow(1.5)

3. **Truncated RUL (cap at maximum):**

   .. code-block:: python

      max_rul = 100  # Cap RUL at 100 units
      df['rul'] = df.groupby('source').cumcount(ascending=False).clip(upper=max_rul)


Testing Your Implementation
----------------------------

With your RUL dataset prepared, test your custom regressor using ``run_experiment``:

.. code-block:: python

   from pdmlabs.utils.dataset import Dataset
   from pdmlabs.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment
   from pdmlabs.RunExperiment import run_experiment
   from my_rul_regressor import MyRULRegressor
   from pdmlabs.pdm_evaluation_types.types import EventPreferences
   
   # 1. Load data (must have continuous RUL labels)
   df = pd.read_csv('your_rul_data.csv')
   dataset_handler = Dataset(
       data=df,
       datetime_column="timestamp",
       source_column="source",
       train_sources=0.6,
       val_sources=0.2,
       test_sources=0.2
   )
   ds_rul, _ = dataset_handler.get_rul_dataset()
   
   # 2. Define hyperparameters for your RUL regressor
   method_param_space = {
       'n_estimators': [50, 100, 150],
       'max_depth': [10, 15, 20],
   }
   
   # 3. Define event preferences
   event_prefs = EventPreferences(
       preprocess_target_events=True,
       postprocess_target_events=True,
       keep_internal_target_events=False,
       keep_internal_nontarget_events=False
   )
   
   # 4. Run experiment with run_experiment
   best_params = run_experiment(
       dataset=ds_rul,
       methods=[MyRULRegressor(event_preferences=event_prefs)],
       param_space_dict_per_method=[method_param_space],
       method_names=['MyRULRegressor'],
       experiments=[SupervisedRULPdMExperiment],
       experiment_names=['RUL Regression'],
       MAX_RUNS=15,
       MAX_JOBS=2,
       INITIAL_RANDOM=2,
       profile_size=10,
       optimization_param='MAE',
       maximize=False
   )
   
   # 5. Check results
   print(f"Best parameters: {best_params[0]}")


Next Steps
----------

- Review ``XGBoostRUL`` implementation in ``pdmlabs/method/xgboostRUL.py``
- Check RUL transformations in ``pdmlabs/utils/rul_transformations.py``
- Explore ``SupervisedRULPdMExperiment`` in ``pdmlabs/experiment/batch/``
- Review dataset RUL preparation in ``pdmlabs/utils/dataset.py::get_rul_dataset()``
