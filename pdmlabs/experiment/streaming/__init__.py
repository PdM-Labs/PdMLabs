"""Streaming experiment classes for online/real-time PdM evaluation (experimental).

**Status: Early-Stage / Stubs**

Streaming experiments are designed for real-time, online scenarios where:
- Data arrives continuously (not all available upfront)
- Models must adapt or update as new data is seen
- Predictions are needed immediately (not retrospectively)

Current State:
This module contains placeholder implementations. Streaming support is planned
for future versions. For production use, prefer batch experiments.

Available Classes:

    StreamingSemiSupervisedPdMExperiment
        Placeholder for online semi-supervised anomaly detection.
        Status: Stub (not implemented)

    StreamingUnsupervisedPdMExperiment
        Placeholder for online unsupervised anomaly detection.
        Status: Stub (not implemented)

Future Roadmap:

    Phase 1 (Future)
        - Per-sample prediction interface
        - Streaming parameter tuning
        - Automated concept drift detection

    Phase 2 (Future)
        - Online model adaptation (no retraining needed)
        - Memory-efficient windoring strategies
        - Real-time MLflow integration

    Phase 3 (Future)
        - Ensemble methods for streaming
        - Anomaly score confidence intervals
        - Multi-source fusion

Recommendation:

For now, use batch experiments (pdmlabs.experiment.batch) for all
production PdM applications. Revisit streaming when fully implemented.

Alternative:
Use temporal cross-validation in batch experiments to simulate streaming
performance (train on early data, test on later data).

See Also:
    - pdmlabs.experiment.batch: Production-ready batch experiments
    - pdmlabs.experiment.experiment: PdMExperiment base class
"""

from pdmlabs.experiment.streaming.semi_supervised_experiment import (
    StreamingSemiSupervisedPdMExperiment,
)
from pdmlabs.experiment.streaming.unsupervised_experiment import (
    StreamingUnsupervisedPdMExperiment,
)

__all__ = [
    'StreamingSemiSupervisedPdMExperiment',
    'StreamingUnsupervisedPdMExperiment',
]
