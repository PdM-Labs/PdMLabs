"""Batch experiment classes for offline/retrospective PdM evaluation.

Batch experiments are designed for comprehensive, offline evaluation of anomaly
detection and remaining useful life (RUL) prediction models. They perform:

- Complete parameter space exploration (Mango optimization)
- Temporal cross-validation (train on past, test on future)
- Full dataset evaluation before deployment

Available Experiment Flavors:

    AutoProfileSemiSupervisedPdMExperiment
        Auto-tunes the size of the "normal profile" (initial N timesteps).
        Best for: Scenarios with clear startup transients where initial behavior
        characterizes normal operation.

    SemiSupervisedPdMExperiment
        Fits method independently on each target scenario.
        Best for: Multiple independent test scenarios; adapts to local patterns.

    SupervisedPdMExperiment
        Uses labeled anomaly windows to train; train-once, test-many.
        Best for: Well-labeled historic data; consistent training across tests.

    UnsupervisedPdMExperiment
        No labels; learns patterns from data alone.
        Best for: Early-stage PdM without failure labels; baseline comparisons.

    IncrementalSemiSupervisedPdMExperiment
        Processes data incrementally with optional model retraining.
        Best for: Simulating online behavior; testing concept drift.

    SupervisedRULPdMExperiment
        Predicts remaining useful life (continuous regression).
        Best for: RUL-aware maintenance scheduling; time-to-failure estimation.

    Supervised_SA_PdMExperiment
        Survival analysis for failure time prediction.
        Best for: Complex failure dynamics; competing risks scenarios.

Choosing an Experiment Flavor:

    1. Do you have labeled anomaly/failure data?
       - Yes → SupervisedPdMExperiment
       - No → Go to step 2

    2. Is the initial portion of data clearly "normal"?
       - Yes → AutoProfileSemiSupervisedPdMExperiment
       - No → SemiSupervisedPdMExperiment

    3. Are you predicting discrete anomalies or continuous RUL?
       - RUL → SupervisedRULPdMExperiment (with labeled RUL data)
       - Anomalies → Choose from above

Typical Pattern:

    >>> from pdmlabs.experiment.batch import AutoProfileSemiSupervisedPdMExperiment
    >>> experiment = AutoProfileSemiSupervisedPdMExperiment(
    ...     experiment_name='my-battery-pd',
    ...     pipeline=pipeline,
    ...     param_space={...},
    ...     num_iteration=30,
    ...     n_jobs=4,
    ...     debug=False
    ... )
    >>> results = experiment.execute()
    >>> best_params = results['best_params']
    >>> best_metric = results['best_objective']

MLflow Integration:

All batch experiments automatically log to MLflow:
- Experiment grouped by name
- Each parameter combination = one MLflow run
- Parameters, metrics, artifacts, and models logged
- Browse results in MLflow UI: mlflow ui

See Also:
    - pdmlabs.experiment.experiment: PdMExperiment base class
    - pdmlabs.pipeline: Define dataset and pipeline
    - pdmlabs.mango: Mango tuner configuration
"""

from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import (
    AutoProfileSemiSupervisedPdMExperiment,
)
from pdmlabs.experiment.batch.incremental_semi_supervised_experiment import (
    IncrementalSemiSupervisedPdMExperiment,
)
from pdmlabs.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment
from pdmlabs.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment
from pdmlabs.experiment.batch.semi_supervised_experiment import (
    SemiSupervisedPdMExperiment,
)
from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
from pdmlabs.experiment.batch.unsupervised_experiment import (
    UnsupervisedPdMExperiment,
)

__all__ = [
    'AutoProfileSemiSupervisedPdMExperiment',
    'IncrementalSemiSupervisedPdMExperiment',
    'SemiSupervisedPdMExperiment',
    'SupervisedPdMExperiment',
    'UnsupervisedPdMExperiment',
    'SupervisedRULPdMExperiment',
    'Supervised_SA_PdMExperiment',
]
