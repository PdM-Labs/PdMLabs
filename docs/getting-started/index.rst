🚀 Getting Started
===============================

Welcome to PdMLabs (pdmlabs)! This section guides you through installing the framework, understanding core concepts, and running your first experiment.

-----------
Quick Paths
-----------

**Choose your learning style:**

.. grid:: 2

    .. grid-item-card:: ⚡ **5-Minute Quickstart**
       :link: quickstart
       
       For the impatient: Prepare data with ``utils.dataset.Dataset`` and run each flavor with one common experiment handler.

    .. grid-item-card:: 📚 **Understand Concepts**
       :link: ../concepts/index

       Before experiments: Learn pipelines, experiment flavors, datasets, and evaluation semantics. Mental models first.

    .. grid-item-card:: 📖 **Step-by-Step Guide**
       :link: ../user-guide/index

       Going deeper: Experiment selection, parameter tuning, component choices, troubleshooting, and best practices.

    .. grid-item-card:: 📜 **Project Philosophy**
       :link: ../introduction

       Why PdMLabs exists: Problem statement, design principles, scope, and extensibility points.

-----------
Installation
-----------

::

    pip install pdmlabs

Or in development mode from source:

::

    git clone <repo-url>
    cd PdM-Evaluation
    pip install -e .

**Requirements:**
- Python 3.8+
- scikit-learn, numpy, pandas
- MLflow (optional, for experiment tracking)
- PyTorch (optional, for neural network methods like TranAD, USAD)

-----------
Next Steps
-----------

1. **Already familiar with ML experimentation?**
   → Jump to :doc:`quickstart` for a hands-on 5-minute walk-through.

2. **New to predictive maintenance or anomaly detection?**
   → Start with :doc:`../introduction` for context, then :doc:`../concepts/index` for mental models.

3. **Ready to build your first experiment?**
   → Head to :doc:`../user-guide/index` for decision trees and real-world guidance.

4. **Need full API documentation?**
   → See :doc:`../api-reference`.

---------
Workflow
---------

At a high level, every experiment follows this pipeline:

.. code-block:: text

    Raw Data
       ↓
    Preprocess (optional: scaling, feature engineering, windowing)
       ↓
    AD Method (isolation forest, neural net, statistical, etc.)
       ↓
    Postprocess (optional: smoothing, aggregation, source fusion)
       ↓
    Threshold (fixed, adaptive, or auto-tuned)
       ↓
    Evaluate (PdM-aware metrics: lead time, episode-aware recall)

The framework handles this pipeline for you via the ``PdMPipeline`` and ``PdMExperiment`` abstractions. You provide:

- A dataset (dict with events, sources, timesteps, preferences)
- An experiment flavor (e.g., ``AutoProfileSemiSupervisedExperiment``)
- Pipeline component choices (preprocessor, method, postprocessor, thresholder)

The framework runs cross-validation, logs results to MLflow, and returns performance metrics.

-----------
Key Concepts
-----------

**Dataset:** A Python dict with time-indexed events, labeled failures, sources (sensors/subsystems), and optional preferences for event interpretation.

**Experiment Flavor:** Batch experiments designed for different settings (multiple methods, semi-supervised, RUL prediction, survival analysis). Streaming experiments are also available but less mature.

**Pipeline:** Sequence of transformers (preprocessor → method → postprocessor → thresholder) applied to each fold.

**Evaluation:** Predictive maintenance–focused metrics (lead time, episode-aware recall, VUS) that measure practical usefulness, not just statistical performance.

**Reproducibility:** Seed management, MLflow logging, and manual parameter overrides ensure deterministic results.

--------
Glossary
--------

.. glossary::

    Method
        Statistical or ML model that detects anomalies (e.g., Isolation Forest, LOF, neural nets).
    
    Preprocessor
        Optional pipeline stage that transforms raw features (e.g., scaling, feature engineering, windowing).
    
    Postprocessor
        Optional pipeline stage that refines predictions (e.g., smoothing, source fusion, aggregation).
    
    Thresholder
        Converts anomaly scores into binary predictions via fixed/adaptive/learned decision boundaries.
    
    Episode
        Contiguous time window bounded by reset events; used for episode-aware evaluation.
    
    Lead Time
        Time from anomaly detection to actual failure; core objective for PdM systems.
    
    AD1_AUC
        Area under curve for recall-at-false-positive-rate trade-off, accounting for lead time penalty.
    
    Fold
        Train/test split used in cross-validation; PdMLabs uses temporal folding strategies.

---------
Next Up
---------

→ :doc:`quickstart` to run your first experiment in 5 minutes.