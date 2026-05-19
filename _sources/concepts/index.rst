📚 Concepts
==========

This page explains the core ideas behind PdMLabs so you can decide how to model
your use case before writing code.


Mental Model
------------

PdMLabs is an experimentation framework for predictive maintenance.

At a high level, each run follows the same pattern:

1. Define a dataset in the expected PdMLabs format.
2. Select an experiment flavor (online, incremental, unsupervised, supervised, etc.).
3. Choose one or more methods.
4. Compose a pipeline with preprocessor, method, postprocessor, and thresholder.
5. Run parameter search and evaluate with PdM-oriented metrics.

This shared structure enables fair comparison across methods and modeling flavors.


Core Building Blocks
--------------------

PdMLabs is built around four pluggable components:

- **Preprocessor**: transforms raw records before scoring.
- **Method**: produces anomaly/probability/survival-like scores.
- **Postprocessor**: smooths or transforms scores.
- **Thresholder**: converts scores to decision thresholds or target values.

In the codebase, this composition is represented by ``PdMPipeline``.


Experiment Flavors
------------------

PdMLabs supports multiple experiment strategies to match different data assumptions.

For an architectural breakdown of how these flavors execute internally, see :doc:`experiment_flavors_analysis`.

**Anomaly Detection**

- ``AutoProfileSemiSupervisedPdMExperiment``
	Builds profile windows and can re-fit after reset events.
- ``IncrementalSemiSupervisedPdMExperiment``
	Trains and predicts over rolling windows.
- ``SemiSupervisedPdMExperiment``
	Fits once on historic data, then scores target data.
- ``UnsupervisedPdMExperiment``
	Scores without a fitting phase.

**Supervised / Time-to-Event**

- ``SupervisedPdMExperiment``
	Classification-style workflow using labels.
- ``SupervisedRULPdMExperiment``
	Remaining useful life (RUL) workflow.
- ``Supervised_SA_PdMExperiment``
	Survival-analysis-oriented workflow.


Data Contract
-------------

Most framework behavior depends on a dataset dictionary with standard keys.
Typical keys include:

- ``event_data`` and ``event_preferences``
- ``historic_data``, ``historic_sources``
- ``target_data``, ``target_sources``
- ``dates``
- ``predictive_horizon``, ``lead``, ``slide``, ``beta``
- ``max_wait_time``

For supervised workflows, labels such as ``anomaly_labels`` (and in some cases
``target_labels``) are required.

The helper module ``pdmlabs.loadAnomalyDetectionDataset`` provides utility
functions to build and enrich dataset dictionaries.


Events, Failures, and Resets
----------------------------

PdMLabs uses event metadata to determine where failures and resets happen and
which sources are affected.

The ``event_preferences`` object defines how to interpret event rows by:

- ``description``
- ``type``
- ``source``
- ``target_sources``

This is important because evaluation and some experiment flavors depend on
episode boundaries and reset logic.


Evaluation Philosophy
---------------------

PdMLabs evaluates results in a predictive-maintenance context, not only with
generic binary metrics.

Main ideas include:

- Episode-aware splitting around failure timestamps.
- Predictive horizon and lead-time semantics.
- Multiple AD recall variants (e.g. AD1/AD2/AD3 style behavior).
- AUC-PR style summaries.
- Optional range/VUS/affiliation metrics.

This helps teams evaluate whether a method gives useful early warnings in practice,
not just good aggregate classification scores.

For a full list of supported metrics and instructions on how to add your own, see :doc:`evaluation`.


Optimization, Reproducibility, and MLflow
-----------------------------------------

Hyperparameter search is integrated into experiments via Mango
(Bayesian or random search) and can use constraint functions to avoid invalid
parameter combinations.

MLflow logging is deeply integrated in the run lifecycle. For every successful experiment, PdMLabs logs:

- All tested parameter configurations and resulting metrics.
- The **best, fully-fitted pipeline** as an MLflow ``pyfunc`` model.

This means the entire processing chain—preprocessor, method, postprocessor, and thresholder—is saved as a single object. You can later load it directly via MLflow and start making predictions:

.. code-block:: python

    import mlflow
    
    pipeline = mlflow.pyfunc.load_model("runs:/<RUN_ID>/best_pdm_pipeline")
    predictions = pipeline.predict({
        'target_data': new_data_df, 
        'source': 'asset_1',
        'event_data': new_event_df
    })

This enables seamless transition from experimentation to production deployment.
For more details on deploying and inference, check the User Guide!


Extensibility
-------------

You can add custom components by implementing the framework interfaces:

- ``MethodInterface`` and specialized method interfaces
- ``RecordLevelPreProcessorInterface``
- ``PostProcessorInterface``
- ``ThresholderInterface``

Once implemented, they can be used with ``run_experiment`` like built-in components.


How To Use This Page
--------------------

- Read :doc:`../introduction` for the high-level motivation.
- Use :doc:`../getting-started/quickstart` to run your first experiment.
- Use :doc:`../api-reference` for detailed API signatures and module docs.

.. toctree::
   :hidden:

   experiment_flavors_analysis
   evaluation