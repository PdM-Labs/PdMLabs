📊 Implementing Semi-Supervised Methods
========================================

Semi-supervised methods learn from clean profile/baseline data to establish normal behavior patterns, then detect deviations as anomalies. This is the most common scenario in predictive maintenance.

**When to Use:** When you have clean operational data (profiles) and want to detect degradation (anomalies). Works with AutoProfile, Incremental, and Historical semi-supervised experiments.

**Key Difference from Unsupervised:** Your method receives labeled training data representing normal operation.


Interface Overview
------------------

Semi-supervised methods inherit from ``SemiSupervisedMethodInterface``:

.. code-block:: python

   from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences

   class MySemiSupervisedMethod(SemiSupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, **kwargs):
           super().__init__(event_preferences=event_preferences)
           # Your initialization


**Key Characteristics:**
- Has ``fit()`` method (training on clean profiles)
- Trains separate models per data source
- Can run with AutoProfile, Incremental, or Historical experiments
- Scores test data against learned normal patterns


Required Methods
----------------

All semi-supervised methods must implement:

1. ``fit(historic_data, historic_sources, event_data)`` — Train on clean profiles
2. ``predict(target_data, source, event_data)`` — Score test data
3. ``predict_one(new_sample, source, is_event)`` — Score single sample (online)
4. ``get_params()`` — Return configuration dictionary
5. ``__str__()`` — Return method name
6. ``get_library()`` — Return library name for serialization (usually ``'no_save'``)
7. ``get_all_models()`` — Return trained models (for export)


Example: Local Outlier Factor Semi-Supervised
----------------------------------------------

The ``LocalOutlierFactor`` is a reference implementation of semi-supervised anomaly detection.

**File Location:** ``pdmlabs/method/lof_semi.py``

**What It Does:**
- Trains LOF models on historical clean profiles
- Detects anomalies as deviations from learned local density
- Maintains separate models per data source (bearing, pump, etc.)

**Implementation Details:**

.. code-block:: python

   class LocalOutlierFactor(SemiSupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, *args, **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.initial_args = args
           self.initial_kwargs = kwargs
           
           # Remove profile_size if present (framework-specific param)
           if 'profile_size' in kwargs:
               del self.initial_kwargs['profile_size']
           
           self.clf_class = local_outlier_factor  # sklearn's LocalOutlierFactor
           self.model_per_source = {}  # Stores trained models per source

**Training Phase:**

.. code-block:: python

   def fit(self, historic_data: list[pd.DataFrame], 
           historic_sources: list[str], 
           event_data: pd.DataFrame) -> None:
       """Train LOF model on each data source."""
       for data, source in zip(historic_data, historic_sources):
           # Create model with novelty=True for out-of-distribution detection
           model = self.clf_class(novelty=True, *self.initial_args, **self.initial_kwargs)
           model.fit(data)
           self.model_per_source[source] = model

**Key Parameters:**
- ``novelty=True`` — Switches to novelty detection (one-class classification)
- ``n_neighbors`` — Number of neighbors for local density (default: 20)
- ``contamination`` — Expected fraction of anomalies (default: auto)

**Prediction Phase:**

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
       """Score test data against trained LOF model."""
       model = self.model_per_source[source]
       # score_samples returns decision scores (negative = anomaly)
       scores = -model.score_samples(target_data)
       return scores.tolist()


Creating Your Own Semi-Supervised Method
-----------------------------------------

Follow this template:

**Step 1: Create File**

Create ``pdmlabs/method/my_semi_supervised_method.py``:

.. code-block:: python

   import pandas as pd
   from sklearn.base import BaseEstimator  # Your chosen algorithm

   from pdmlabs.method.semi_supervised_method import SemiSupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences


   class MySemiSupervisedMethod(SemiSupervisedMethodInterface):
       """Describe your semi-supervised anomaly detection method.
       
       This method learns normal behavior from clean training data,
       then detects anomalies as deviations from the learned model.
       """

       def __init__(self, 
                    event_preferences: EventPreferences,
                    n_neighbors: int = 20,
                    contamination: str = 'auto',
                    *args,
                    **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.n_neighbors = n_neighbors
           self.contamination = contamination
           self.initial_args = args
           self.initial_kwargs = kwargs
           
           # Important: Clean up framework-specific parameters
           if 'profile_size' in self.initial_kwargs:
               del self.initial_kwargs['profile_size']
           
           self.model_per_source = {}  # Will store trained models


**Step 2: Implement fit()**

Train on clean profiles (one model per source):

.. code-block:: python

   def fit(self, historic_data: list[pd.DataFrame], 
           historic_sources: list[str], 
           event_data: pd.DataFrame) -> None:
       """Train anomaly detector on clean historical data.
       
       Args:
           historic_data: List of training DataFrames (one per source).
               Each represents clean operation periods.
           historic_sources: List of source names (e.g., ['bearing_1', 'bearing_2'])
           event_data: Event log for reference (optional)
       """
       for data, source in zip(historic_data, historic_sources):
           # Initialize model for this source
           model = SomeAlgorithm(
               n_neighbors=self.n_neighbors,
               contamination=self.contamination,
               *self.initial_args,
               **self.initial_kwargs
           )
           
           # Train on clean profile data
           model.fit(data)
           
           # Store for later prediction
           self.model_per_source[source] = model


**Step 3: Implement predict()**

Score test data using trained models:

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
       """
       Score all test samples against trained model for source.
       
       Args:
           target_data: Test features (rows=samples, cols=features)
           source: Source identifier to look up trained model
           event_data: Event log (optional reference)
       
       Returns:
           List of anomaly scores (higher = more anomalous)
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model trained for source '{source}'")
       
       model = self.model_per_source[source]
       
       # Get anomaly scores (method-specific)
       # Most sklearn detectors: score_samples() returns decision scores
       scores = model.score_samples(target_data)
       
       # Ensure higher = more anomalous (negate if needed)
       if hasattr(model, '_invert_scores'):
           scores = -scores
       
       return scores.tolist()


**Step 4: Implement predict_one()**

Score single samples (for online/incremental experiments):

.. code-block:: python

   def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
       """
       Score a single new sample.
       
       Args:
           new_sample: Single sample as pandas Series
           source: Source identifier
           is_event: Whether marked as event (context only)
       
       Returns:
           Anomaly score for the sample
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model trained for source '{source}'")
       
       model = self.model_per_source[source]
       
       # Convert to expected shape
       sample_array = new_sample.to_numpy().reshape(1, -1)
       
       # Score using same logic as batch predict
       score = model.score_samples(sample_array)[0]
       
       if hasattr(model, '_invert_scores'):
           score = -score
       
       return float(score)


**Step 5: Implement get_params()**

Return configuration for reproducibility:

.. code-block:: python

   def get_params(self) -> dict:
       """Return all hyperparameters."""
       # Get params from first trained model as reference
       first_source = list(self.model_per_source.keys())[0]
       model = self.model_per_source[first_source]
       
       return {
           **model.get_params(),  # Include underlying algorithm params
           'n_neighbors': self.n_neighbors,
           'contamination': self.contamination,
       }


**Step 6: Implement __str__(), get_library(), get_all_models()**

.. code-block:: python

   def __str__(self) -> str:
       """Human-readable method name."""
       return 'MySemiSupervisedMethod'

   def get_library(self) -> str:
       """Return library for serialization."""
       return 'no_save'  # or your library name

   def get_all_models(self):
       """Return trained models for export (optional)."""
       return self.model_per_source


Testing Your Implementation
----------------------------

With your dataset prepared, test your custom semi-supervised method using ``run_experiment``:

.. code-block:: python

   from pdmlabs.utils.dataset import Dataset
   from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import AutoProfileSemiSupervisedPdMExperiment
   from pdmlabs.RunExperiment import run_experiment
   from my_semi_supervised_method import MySemiSupervisedMethod
   from pdmlabs.pdm_evaluation_types.types import EventPreferences
   
   # 1. Load data (clean profiles for training)
   df = pd.read_csv('your_data.csv')
   dataset_handler = Dataset(
       data=df,
       datetime_column="timestamp",
       source_column="source",
       train_sources=0.6,
       val_sources=0.2,
       test_sources=0.2
   )
   ds_semi, _ = dataset_handler.get_semi_dataset()
   
   # 2. Define hyperparameters for your method
   method_param_space = {
       'n_neighbors': [15, 20, 25],
       'contamination': ['auto'],
   }
   
   # 3. Run experiment with run_experiment
   best_params = run_experiment(
       dataset=ds_semi,
       methods=[MySemiSupervisedMethod],
       param_space_dict_per_method=[method_param_space],
       method_names=['MySemiSupervisedMethod'],
       experiments=[AutoProfileSemiSupervisedPdMExperiment],
       experiment_names=['AutoProfile'],
       MAX_RUNS=10,
       MAX_JOBS=2,
       INITIAL_RANDOM=2,
       profile_size=[5, 10],
       fit_size=5,
       optimization_param='AD1_AUC'
   )
   
   # 5. Check results
   print(f"Best parameters: {best_params[0]}")


Next Steps
----------

- Review ``LocalOutlierFactor`` source code in ``pdmlabs/method/lof_semi.py``
- Explore parameter generation in ``pdmlabs/utils/automatic_parameter_generation.py::online_technique()``
- Check dataset structure in ``pdmlabs/utils/dataset.py::get_semi_dataset()``
