import pandas as pd
import mlflow

from pdmlabs.experiment import PdMExperiment
from pdmlabs.evaluation.evaluation import myeval as pdm_evaluate


class StreamingSemiSupervisedPdMExperiment(PdMExperiment):
    """Streaming (online) semi-supervised anomaly detection.

    **Status: Experimental/Stub Implementation**

    This experiment flavor is designed for streaming data scenarios:
    - Processes data continuously as it arrives (row-by-row or in small batches)
    - Adapts models online without batch retraining
    - Produces predictions in real-time

    Current Implementation:
    This is an early-stage stub that iterates over target data but does not yet
    implement full streaming evaluation logic. Use batch experiments for production.

    Future Work:
    - Streaming parameter tuning
    - Online model adaptation
    - Concept drift detection
    - Memory-efficient processing

    Raises:
        NotImplementedError: Full streaming functionality not yet implemented.

    Examples:
        >>> experiment = StreamingSemiSupervisedPdMExperiment(...)
        >>> # Note: streaming experiments are currently stubs
        >>> # Use batch experiments instead for now
    """
    def execute(self) -> None:
        """Execute placeholder streaming experiment (not fully implemented).

        Returns:
            None: Streaming experiments are currently stubs.
        """
        super()._register_experiment()

        with mlflow.start_run(experiment_id=self.experiment_id) as parent_run:
            for current_row_index, current_row in self.target_data.iterrows():
                print(current_row_index)

            super()._finish_run(parent_run=parent_run)