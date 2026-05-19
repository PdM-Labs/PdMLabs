📜 Our Manifesto
================

PdMLabs is an open-source Python framework for predictive maintenance (PdM) experimentation.
It is designed to help teams detect early warning signs and model time-to-event behavior,
while keeping experiments reproducible, comparable, and extensible.

This project is not just a collection of isolated models. It provides a full experiment
orchestration layer around data preparation, method execution, post-processing, thresholding,
hyperparameter optimization, and evaluation.


What Problem It Solves
----------------------

In real industrial settings, teams usually face one or more of these challenges:

- It is unclear which modeling flavor is appropriate for a given use case.
- Data arrives from multiple sources/assets with different event semantics.
- Evaluation must reflect predictive-maintenance reality, not only generic metrics.
- Reproducibility and fair comparison across methods are difficult.

PdMLabs addresses these by standardizing the experiment workflow and exposing clear interfaces
for each stage of the pipeline.


Design Philosophy
-----------------

PdMLabs follows a few core principles:

1. **Pipeline-first design**
   Experiments are built from explicit steps:
   ``preprocessor -> method -> postprocessor -> thresholder``.

2. **Interface-based extensibility**
   New methods and processing components can be added by implementing lightweight interfaces,
   without rewriting the experiment engine.

3. **Evaluation aligned with PdM goals**
   Metrics emphasize practical early warning behavior (predictive horizon, lead time, episode-aware
   evaluation), with additional range/VUS-style metrics available.

4. **Optimization and tracking by default**
   Parameter search (Bayesian/Random via Mango) and MLflow logging are integrated into the
   experiment lifecycle.

5. **Multi-source realism**
   The dataset contract and event preference mechanism support multiple assets and source mappings,
   matching real deployment scenarios.


How The Framework Is Organized
------------------------------

At a high level, the framework is organized around these packages:

- ``pdmlabs.pipeline``: Pipeline composition and event-aware utilities.
- ``pdmlabs.experiment``: Experiment flavors and execution orchestration.
- ``pdmlabs.method``: Detection/prediction techniques (semi-supervised, unsupervised, supervised).
- ``pdmlabs.preprocessing``: Input transformations before model scoring.
- ``pdmlabs.postprocessing``: Score transformations after model scoring.
- ``pdmlabs.thresholding``: Threshold inference or score-to-target conversion.
- ``pdmlabs.evaluation``: PdM-focused metrics and evaluation utilities.
- ``pdmlabs.utils``: Helper utilities, including automatic parameter generation.


Supported Experiment Flavors
----------------------------

PdMLabs supports multiple experiment modes, including:

- Online auto-profile semi-supervised anomaly detection.
- Incremental semi-supervised anomaly detection.
- Semi-supervised anomaly detection.
- Unsupervised anomaly detection.
- Supervised classification.
- Supervised RUL / survival-oriented workflows.

This allows users to compare fundamentally different strategies under a common execution and
evaluation framework.


Extending PdMLabs
-----------------

You can extend PdMLabs by implementing one of the provided interfaces:

- ``MethodInterface`` and its supervised/semi-supervised/unsupervised variants.
- ``RecordLevelPreProcessorInterface``.
- ``PostProcessorInterface``.
- ``ThresholderInterface``.

After implementation, your custom component can be plugged into ``run_experiment`` just like
built-in components.


Current Scope
-------------

PdMLabs is strong in batch experimentation and comparative evaluation.
Streaming-related modules are present but currently more limited than batch workflows.

The recommended path for new users is:

1. Read :doc:`concepts/index`.
2. Follow :doc:`getting-started/quickstart`.
3. Explore :doc:`examples/index`.
4. Use :doc:`api-reference` for implementation details.