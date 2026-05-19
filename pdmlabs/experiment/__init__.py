"""Experiment classes for automated PdM model evaluation and hyperparameter tuning.

This module provides the experiment framework that orchestrates:
- Parameter space exploration via Mango (Bayesian/random optimization)
- Cross-validation and temporal evaluation
- MLflow tracking and artifact management
- PdM-aware metric computation

Core Abstractions:

    PdMExperiment (experiment.py)
        Abstract base class for all experiment flavors.
        Defines common interface: execute() -> dict

    Batch Experiments (batch/)
        Designed for offline/retrospective evaluation.
        Best for production validation and performance reporting.
        - AutoProfileSemiSupervisedPdMExperiment: Auto-tuned profile size
        - SemiSupervisedPdMExperiment: Per-scenario invariance
        - SupervisedPdMExperiment: Labeled data training
        - UnsupervisedPdMExperiment: No labels, pattern-based detection
        - IncrementalSemiSupervisedPdMExperiment: Online-style incremental fitting
        - SupervisedRULPdMExperiment: Remaining useful life regression
        - Supervised_SA_PdMExperiment: Survival analysis

    Streaming Experiments (streaming/)
        Early-stage stubs for real-time scenarios.
        Currently not production-ready; use batch for now.
        - StreamingSemiSupervisedPdMExperiment
        - StreamingUnsupervisedPdMExperiment

Typical Workflow:

    1. Prepare dataset (dict with features, labels, events, sources)
    2. Create PdMPipeline specifying method, preprocessor, postprocessor, thresholder
    3. Define param_space for Mango optimization
    4. Choose experiment flavor (e.g., AutoProfileSemiSupervisedPdMExperiment)
    5. Call experiment.execute() to run optimization
    6. Access best_params and metrics from result dict
    7. View runs in MLflow UI

Example:

    >>> from pdmlabs.pipeline.pipeline import PdMPipeline
    >>> from pdmlabs.experiment.batch import AutoProfileSemiSupervisedPdMExperiment
    >>>
    >>> pipeline = PdMPipeline(
    ...     dataset=my_dataset,
    ...     method=IsolationForest,
    ...     preprocessor=StandardScaler,
    ...     postprocessor=NoPostprocessor,
    ...     thresholder=StaticThreshold
    ... )
    >>> param_space = {'profile_size': [10, 20, 50], 'method_contamination': [0.01, 0.05]}
    >>> experiment = AutoProfileSemiSupervisedPdMExperiment(
    ...     experiment_name='demo-auto-profile',
    ...     pipeline=pipeline,
    ...     param_space=param_space,
    ...     num_iteration=30,
    ...     n_jobs=4
    ... )
    >>> results = experiment.execute()
    >>> print(f"Best profile size: {results['best_params']['profile_size']}")
    Best profile size: 20

See Also:
    - pdmlabs.pipeline: PdMPipeline and data contract definition
    - pdmlabs.method: Available anomaly detection methods
    - pdmlabs.preprocessing, postprocessing, thresholding: Pipeline components
    - pdmlabs.evaluation: PdM-aware evaluation metrics
    - pdmlabs.mango: Mango tuner configuration
"""

from pdmlabs.experiment.experiment import PdMExperiment

__all__ = [
    'PdMExperiment',
]
