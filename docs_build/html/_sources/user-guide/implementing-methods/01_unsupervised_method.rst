🔍 Implementing Unsupervised Methods
====================================

Unsupervised methods detect anomalies without any labeled training data. They're ideal for discovering novel or unexpected patterns in sensor streams.

**When to Use:** Early exploratory analysis, systems where labeling is expensive, or when normal behavior is hard to define.


Interface Overview
------------------

Unsupervised methods inherit from ``UnsupervisedMethodInterface``:

.. code-block:: python

   from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences

   class MyUnsupervisedMethod(UnsupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, **kwargs):
           super().__init__(event_preferences=event_preferences)
           # Your initialization


**Key Characteristics:**
- No ``fit()`` method (no training phase)
- Models train online on a sliding window of data
- No labeled training data required
- Works with ``UnsupervisedPdMExperiment`` only


Required Methods
----------------

All unsupervised methods must implement:

1. ``predict(target_data, source, event_data)`` — Score entire dataset
2. ``predict_one(new_sample, source, is_event)`` — Score single sample (online)
3. ``get_params()`` — Return configuration dictionary
4. ``__str__()`` — Return method name
5. ``get_library()`` — Return library name for serialization (usually ``'no_save'``)


Example: Isolation Forest Unsupervised
----------------------------------------

The ``IsolationForestUnsupervised`` is a reference implementation of unsupervised anomaly detection.

**File Location:** ``pdmlabs/method/isolation_forest_uns.py``

**What It Does:**
- Trains Isolation Forest models on sliding windows of data
- Each window produces scores based on isolation depth
- Combines overlapping window scores using configurable policies (or/and/first/last)

**Implementation Details:**

.. code-block:: python

   class IsolationForestUnsupervised(UnsupervisedMethodInterface):
       def __init__(self, 
                    event_preferences: EventPreferences,
                    window=40,           # Sliding window size
                    slide=0.5,           # Step size as fraction of window
                    policy="or",         # Score aggregation: or/and/first/last
                    *args, 
                    **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.window = window
           self.slide = int(slide * window)
           self.policy = policy
           self.clf_class = isolation_forest  # sklearn's IsolationForest
           self.model_per_source = {}  # Not used for unsupervised, but kept for consistency

**Key Components:**

- ``window`` — Size of sliding window for training (default: 40 samples)
- ``slide`` — Step size between windows as fraction (default: 0.5 = 50% overlap)
- ``policy`` — How to combine scores from overlapping windows:
  - ``"or"`` — Take maximum score (most sensitive)
  - ``"and"`` — Take minimum score (most conservative)
  - ``"first"`` — Keep first score seen
  - ``"last"`` — Keep last score seen

**Prediction Flow:**

1. Initialize empty score dictionary
2. Slide window over data in steps of ``slide`` samples
3. For each window position:
   - Extract window data
   - Train new Isolation Forest on window
   - Score all samples in window
   - Combine with existing scores using policy
4. Return final scores for all samples

**Scoring:**
- ``-tempmodel.score_samples(window_data)`` — Negative because sklearn returns decision scores (negative = anomaly)
- Higher score = more anomalous


Creating Your Own Unsupervised Method
--------------------------------------

Follow this template:

**Step 1: Create File**

Create ``pdmlabs/method/my_unsupervised_method.py``:

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import SomeDetector  # Your chosen algorithm

   from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences


   class MyUnsupervisedMethod(UnsupervisedMethodInterface):
       """Describe your unsupervised anomaly detection method."""

       def __init__(self, 
                    event_preferences: EventPreferences,
                    window: int = 50,
                    threshold: float = 0.5,
                    *args,
                    **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.window = window
           self.threshold = threshold
           self.initial_args = args
           self.initial_kwargs = kwargs
           self.model_per_source = {}  # Keep for consistency even if not used


**Step 2: Implement predict()**

Score entire dataset:

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
       """
       Score all samples in target_data.
       
       Args:
           target_data: Features (rows=samples, cols=features)
           source: Source identifier (not needed for pure unsupervised)
           event_data: Event log (not needed for unsupervised)
       
       Returns:
           List of anomaly scores (one per sample)
       """
       scores = self._compute_scores(target_data)
       return scores


**Step 3: Implement predict_one()**

Score single sample (for online settings):

.. code-block:: python

   def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
       """
       Score a single new sample.
       
       For fully streaming methods, you might:
       - Update a model buffer
       - Recompute on-demand
       - Use approximate scores
       
       Args:
           new_sample: Single sample as Series
           source: Source identifier
           is_event: Whether this sample is marked as event
       
       Returns:
           Anomaly score for the sample
       """
       # For algorithms that need recent context:
       # - Keep a sliding buffer
       # - Update statistics incrementally
       # - Approximate scores efficiently
       
       score = self._compute_score_single(new_sample)
       return score


**Step 4: Implement get_params()**

Return configuration:

.. code-block:: python

   def get_params(self) -> dict:
       """Return all hyperparameters of this method instance."""
       return {
           'window': self.window,
           'threshold': self.threshold,
           # Include model-specific params if applicable
       }


**Step 5: Implement __str__() and get_library()**

.. code-block:: python

   def __str__(self) -> str:
       """Human-readable method name."""
       return 'MyUnsupervisedMethod'

   def get_library(self) -> str:
       """Return 'no_save' for non-serializable methods."""
       return 'no_save'


**Step 6: (Optional) Implement get_all_models()**

Some methods need to export models:

.. code-block:: python

   def get_all_models(self):
       """Return any trained models (for persistence)."""
       # Return None if not applicable
       return None


Testing Your Implementation
----------------------------

With your dataset prepared, test your custom unsupervised method using ``run_experiment``:

.. code-block:: python

   from pdmlabs.utils.dataset import Dataset
   from pdmlabs.experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
   from pdmlabs.RunExperiment import run_experiment
   from my_unsupervised_method import MyUnsupervisedMethod
   
   # 1. Load data
   df = pd.read_csv('your_data.csv')
   dataset_handler = Dataset(
       data=df,
       datetime_column="timestamp",
       source_column="source",
       train_sources=0.6,
       val_sources=0.2,
       test_sources=0.2
   )
   ds_unsup, _ = dataset_handler.get_unsupervised_dataset()
   
   # 2. Define hyperparameters for your method
   method_param_space = {
       'window': [30, 40, 50],
       'slide': [0.5],
   }
   
   # 3. Run experiment with run_experiment
   best_params = run_experiment(
       dataset=ds_unsup,
       methods=[MyUnsupervisedMethod],
       param_space_dict_per_method=[method_param_space],
       method_names=['MyUnsupervisedMethod'],
       experiments=[UnsupervisedPdMExperiment],
       experiment_names=['Unsupervised Detection'],
       MAX_RUNS=10,
       MAX_JOBS=2,
       INITIAL_RANDOM=2,
       profile_size=10,
       optimization_param='AD1_AUC'
   )
   
   # 4. Check results
   print(f"Best parameters: {best_params[0]}")


Next Steps
----------

- Review ``IsolationForestUnsupervised`` source code for a working reference
- Explore other unsupervised methods in ``pdmlabs/method/`` (e.g., PCA, autoencoder-based)
- Implement parameter generation function in ``pdmlabs/utils/automatic_parameter_generation.py`` if needed
