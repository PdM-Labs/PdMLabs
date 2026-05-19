🛠️ Implementing Custom Methods
==============================

This section explains how to implement custom anomaly detection methods for each of the five PdM experiment flavors.

The framework provides base interfaces that your custom method must implement. Each flavor requires different components, but they follow the same general pattern:

1. **Inherit from the appropriate interface** (Unsupervised, Semi-supervised, Supervised)
2. **Implement required methods** (fit, predict, predict_one, get_params, etc.)
3. **Handle per-source models** (most methods maintain separate models for each data source)
4. **Return scores** in the expected format (float or list of floats)


Key Concepts
------------

**Method Interfaces:**
Each method must inherit from one of five base interfaces:

- :doc:`01_unsupervised_method` — For unsupervised anomaly detection (online learning)
- :doc:`02_semi_supervised_method` — For semi-supervised detection (profile-based learning)
- :doc:`03_classification_method` — For supervised binary classification
- :doc:`04_rul_regression_method` — For Remaining Useful Life regression
- :doc:`05_survival_analysis_method` — For survival analysis and censoring

**Method Requirements:**

Every method must:

- Accept ``event_preferences`` in ``__init__``
- Maintain ``model_per_source`` dictionary to store models per data source
- Implement ``predict()`` to score entire datasets
- Implement ``predict_one()`` to score individual samples (for online settings)
- Implement ``get_params()`` to return configuration dictionary
- Implement ``__str__()`` to return human-readable method name
- Return scores as float values (or specific structured formats for survival analysis)


Which Flavor Should I Implement For?
-------------------------------------

Choose based on your use case:

.. container:: grid-container

   .. grid-item-card:: 🔍 Unsupervised Learning
      :link: 01_unsupervised_method
      
      Detect anomalies without labeled data. Good for discovering novel patterns.
      
      Example: Isolation Forest with sliding windows

   .. grid-item-card:: 📊 Semi-Supervised Learning
      :link: 02_semi_supervised_method
      
      Learn from clean profiles, adapt with events. Good for typical operational data.
      
      Example: Local Outlier Factor trained on profile data

   .. grid-item-card:: 🎯 Classification
      :link: 03_classification_method
      
      Binary classifier: normal vs anomalous. Requires labeled training data.
      
      Example: XGBoost trained on annotated sensor readings

   .. grid-item-card:: ⏱️ RUL Regression
      :link: 04_rul_regression_method
      
      Predict remaining useful life as a regression target.
      
      Example: XGBoost regressor trained on time-to-failure labels

   .. grid-item-card:: 🧬 Survival Analysis
      :link: 05_survival_analysis_method
      
      Model time-to-event with censoring and survival curves.
      
      Example: Cox Proportional Hazards model


Implementation Workflow
----------------------

For each flavor, follow these steps:

1. **Create a new class file** in ``pdmlabs/method/`` (e.g., ``my_detector.py``)
2. **Import the interface** (e.g., ``from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface``)
3. **Inherit from interface** and implement all required abstract methods
4. **Test with a simple dataset** to verify prediction format
5. **Integrate into experiments** using the framework's pipeline utilities
6. **Register in** ``pdmlabs/method/__init__.py`` if desired


Common Patterns
---------------

**Storing Models Per Source:**

.. code-block:: python

   def fit(self, historic_data, historic_sources, event_data, anomaly_ranges):
       for data, source, labels in zip(historic_data, historic_sources, anomaly_ranges):
           # Train one model per source
           model = SomeAlgorithm()
           model.fit(data, labels)
           self.model_per_source[source] = model

**Predicting with Source Lookup:**

.. code-block:: python

   def predict(self, target_data, source, event_data):
       model = self.model_per_source[source]
       scores = model.predict_proba(target_data)
       return scores[:, 1].tolist()  # Return as list of floats

**Getting Hyperparameters:**

.. code-block:: python

   def get_params(self):
       return self.model_per_source[list(self.model_per_source.keys())[0]].get_params()


See Detailed Guides
-------------------

Select your experiment flavor below to learn how to implement a custom method:

.. toctree::
   :maxdepth: 2
   :caption: Implementation Guides

   01_unsupervised_method
   02_semi_supervised_method
   03_classification_method
   04_rul_regression_method
   05_survival_analysis_method
