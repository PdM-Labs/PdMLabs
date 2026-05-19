🎯 Implementing Classification Methods
======================================

Classification methods distinguish between **normal** (healthy) and **anomalous** (degraded) states as a binary classification problem. This is the standard supervised learning approach for anomaly detection.

**When to Use:** When you have labeled training data annotated as normal vs anomalous. Works only with ``SupervisedPdMExperiment``.

**Key Requirement:** Training data must include binary labels (0=normal, 1=anomalous).


Interface Overview
------------------

Classification methods inherit from ``SupervisedMethodInterface``:

.. code-block:: python

   from pdmlabs.method.supervised_method import SupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences

   class MyClassificationMethod(SupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, **kwargs):
           super().__init__(event_preferences=event_preferences)
           # Your initialization


**Key Characteristics:**
- Has ``fit()`` method (training on labeled data)
- Binary classification: normal vs anomaly
- Returns anomaly probabilities as scores
- Must predict probability of positive class (anomalous)
- Works with ``SupervisedPdMExperiment`` only


Required Methods
----------------

All classification methods must implement:

1. ``fit(historic_data, historic_sources, event_data, anomaly_ranges)`` — Train on labeled data
2. ``predict(target_data, source, event_data)`` — Score test data
3. ``predict_one(new_sample, source, is_event)`` — Score single sample
4. ``get_params()`` — Return configuration dictionary
5. ``__str__()`` — Return method name
6. ``get_library()`` — Return library name (usually ``'no_save'``)
7. ``get_all_models()`` — Return trained models (for export)


Example: XGBoost Classification
-------------------------------

The ``XGBoost`` is a reference implementation of supervised binary classification.

**File Location:** ``pdmlabs/method/xgboost.py``

**What It Does:**
- Trains XGBoost classifier to distinguish normal vs anomalous samples
- Maintains separate classifiers per data source
- Returns probability of anomaly as score

**Implementation Details:**

.. code-block:: python

   class XGBoost(SupervisedMethodInterface):
       def __init__(self, event_preferences: EventPreferences, *args, **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.model_per_source = {}
           self.initial_args = args
           self.initial_kwargs = kwargs

**Training Phase:**

.. code-block:: python

   def fit(self, historic_data: list[pd.DataFrame], 
           historic_sources: list[str], 
           event_data: pd.DataFrame,
           anomaly_ranges: list[list]) -> None:
       """Train XGBoost classifier on labeled data.
       
       Args:
           historic_data: Training features (one DataFrame per source)
           historic_sources: Source identifiers
           event_data: Event log (for reference)
           anomaly_ranges: Binary labels (0=normal, 1=anomaly)
       """
       for data, source, labels in zip(historic_data, historic_sources, anomaly_ranges):
           model = xgb.XGBClassifier(*self.initial_args, **self.initial_kwargs)
           model.fit(data, labels)
           self.model_per_source[source] = model

**Key Parameters:**
- ``learning_rate`` — Speed of learning (default: 0.1)
- ``max_depth`` — Tree depth (default: 5)
- ``n_estimators`` — Number of boosting rounds (default: 100)
- ``subsample`` — Fraction of samples for training (default: 1.0)
- ``colsample_bytree`` — Fraction of features per tree (default: 1.0)

**Prediction Phase:**

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
       """Score test data as probability of anomaly."""
       model = self.model_per_source[source]
       # predict_proba returns [[prob_normal, prob_anomaly], ...]
       scores = model.predict_proba(target_data)[:, 1]  # Get anomaly probability
       return scores.tolist()


Creating Your Own Classification Method
----------------------------------------

Follow this template:

**Step 1: Create File**

Create ``pdmlabs/method/my_classifier.py``:

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier  # Your chosen algorithm

   from pdmlabs.method.supervised_method import SupervisedMethodInterface
   from pdmlabs.pdm_evaluation_types.types import EventPreferences


   class MyClassifier(SupervisedMethodInterface):
       """Binary anomaly classifier for predictive maintenance.
       
       This method learns to distinguish normal operation from degraded/anomalous
       states using supervised binary classification.
       """

       def __init__(self, 
                    event_preferences: EventPreferences,
                    n_estimators: int = 100,
                    max_depth: int = 10,
                    *args,
                    **kwargs):
           super().__init__(event_preferences=event_preferences)
           self.n_estimators = n_estimators
           self.max_depth = max_depth
           self.initial_args = args
           self.initial_kwargs = kwargs
           self.model_per_source = {}


**Step 2: Implement fit()**

Train separate classifiers per source:

.. code-block:: python

   def fit(self, historic_data: list[pd.DataFrame], 
           historic_sources: list[str], 
           event_data: pd.DataFrame,
           anomaly_ranges: list[list]) -> None:
       """
       Train a binary classifier for each data source.
       
       Args:
           historic_data: Training features (one per source)
           historic_sources: Source names (e.g., ['bearing_1', 'bearing_2'])
           event_data: Event log (optional reference)
           anomaly_ranges: Binary labels (list per source, values 0 or 1)
       
       The labels should be:
       - 0 for samples representing normal/healthy operation
       - 1 for samples showing degradation or failure conditions
       """
       for data, source, labels in zip(historic_data, historic_sources, anomaly_ranges):
           # Create classifier
           classifier = RandomForestClassifier(
               n_estimators=self.n_estimators,
               max_depth=self.max_depth,
               *self.initial_args,
               **self.initial_kwargs
           )
           
           # Train on labeled data
           classifier.fit(data, labels)
           
           # Store model for this source
           self.model_per_source[source] = classifier


**Step 3: Implement predict()**

Score using trained classifiers:

.. code-block:: python

   def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list[float]:
       """
       Score test data using trained classifier for source.
       
       Scores represent probability of anomaly (higher = more anomalous).
       
       Args:
           target_data: Test features (rows=samples, cols=features)
           source: Source identifier to look up trained model
           event_data: Event log (optional reference)
       
       Returns:
           List of anomaly probabilities [0.0, 1.0]
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model trained for source '{source}'")
       
       classifier = self.model_per_source[source]
       
       # Get probability estimates
       # Most sklearn classifiers: predict_proba returns [[prob_class_0, prob_class_1], ...]
       probabilities = classifier.predict_proba(target_data)
       
       # Extract probability of class 1 (anomaly)
       anomaly_scores = probabilities[:, 1]
       
       return anomaly_scores.tolist()


**Step 4: Implement predict_one()**

Score individual samples:

.. code-block:: python

   def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
       """
       Score a single new sample.
       
       Args:
           new_sample: Single sample as pandas Series
           source: Source identifier
           is_event: Whether marked as event (context only)
       
       Returns:
           Anomaly probability for this sample
       """
       if source not in self.model_per_source:
           raise ValueError(f"No model trained for source '{source}'")
       
       classifier = self.model_per_source[source]
       
       # Reshape to 2D: (1, num_features)
       sample_array = new_sample.to_numpy().reshape(1, -1)
       
       # Get probabilities
       probabilities = classifier.predict_proba(sample_array)
       
       # Return probability of class 1 (anomaly)
       return float(probabilities[0, 1])


**Step 5: Implement get_params()**

Return hyperparameters:

.. code-block:: python

   def get_params(self) -> dict:
       """Return all hyperparameters for reproducibility."""
       # Get params from first trained model
       first_source = list(self.model_per_source.keys())[0]
       model = self.model_per_source[first_source]
       
       return {
           **model.get_params(),
           'n_estimators': self.n_estimators,
           'max_depth': self.max_depth,
       }


**Step 6: Implement remaining methods**

.. code-block:: python

   def __str__(self) -> str:
       """Human-readable method name."""
       return 'MyClassifier'

   def get_library(self) -> str:
       """Library for serialization."""
       return 'no_save'

   def get_all_models(self):
       """Export trained models."""
       return self.model_per_source


Data Labeling Guidelines
------------------------

Classification requires accurate labeling of training data:

**Normal (0) Samples:**
- Healthy sensor readings
- Within expected parameter ranges
- Stable baseline measurements
- Pre-failure or early-stage operation

**Anomalous (1) Samples:**
- Degraded behavior indicators
- Fault condition signatures
- Early warning signs of failure
- Anomalous sensor patterns


**Labeling Strategies:**

1. **Time-window based:**
   - Mark samples 0-N days before failure as anomaly (1)
   - Mark remaining as normal (0)
   - Adjustable lookback window affects class balance

2. **Threshold-based:**
   - Mark samples exceeding thresholds as anomalies
   - Use domain expertise to set thresholds

3. **Human annotation:**
   - Domain experts manually review and label
   - Most accurate but time-consuming

4. **Event-driven:**
   - Use maintenance/failure events as boundaries
   - Samples after event = anomalous, before = normal


Testing Your Implementation
----------------------------

With your labeled dataset prepared, test your custom classifier using ``run_experiment``:

.. code-block:: python

   from pdmlabs.utils.dataset import Dataset
   from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
   from pdmlabs.RunExperiment import run_experiment
   from my_classifier import MyClassifier
   from pdmlabs.pdm_evaluation_types.types import EventPreferences
   
   # 1. Load data (must have binary labels: 0=normal, 1=anomaly)
   df = pd.read_csv('your_labeled_data.csv')
   dataset_handler = Dataset(
       data=df,
       datetime_column="timestamp",
       source_column="source",
       train_sources=0.6,
       val_sources=0.2,
       test_sources=0.2
   )
   ds_class, _ = dataset_handler.get_Classification_dataset()
   
   # 2. Define hyperparameters for your classifier
   method_param_space = {
       'n_estimators': [50, 100, 150],
       'max_depth': [8, 10, 12],
   }
   
   # 3. Run experiment with run_experiment
   best_params = run_experiment(
       dataset=ds_class,
       methods=[MyClassifier],
       param_space_dict_per_method=[method_param_space],
       method_names=['MyClassifier'],
       experiments=[SupervisedPdMExperiment],
       experiment_names=['Classification'],
       MAX_RUNS=15,
       MAX_JOBS=2,
       INITIAL_RANDOM=2,
       profile_size=10,
       optimization_param='AD1_AUC'
   )
   
   # 5. Check results
   print(f"Best parameters: {best_params[0]}")


Next Steps
----------

- Review ``XGBoost`` implementation in ``pdmlabs/method/xgboost.py``
- Check classification metrics in ``pdmlabs/evaluation/vus/``
- Explore ``SupervisedPdMExperiment`` in ``pdmlabs/experiment/batch/supervised_experiment.py``
