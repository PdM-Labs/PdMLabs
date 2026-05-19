from pdmlabs.experiment import PdMExperiment


class StreamingUnsupervisedPdMExperiment(PdMExperiment):
    """Streaming (online) unsupervised anomaly detection.

    **Status: Stub Implementation**

    This experiment flavor is designed for unsupervised streaming data:
    - Processes continuous data streams without labels
    - Adapts models in real-time
    - Produces anomaly scores online

    Current Implementation:
    This is a placeholder stub with no execution logic. Use batch experiments
    for full functionality. Streaming support is planned for future versions.

    Design Goals:
    - Minimal memory footprint for long-running applications
    - Per-sample or mini-batch prediction
    - Automatic concept drift handling
    - No offline/batch retraining required

    Raises:
        NotImplementedError: Streaming functionality not yet implemented.

    Examples:
        >>> # Streaming experiments are not yet implemented
        >>> # Use UnsupervisedPdMExperiment (batch) instead
    """
    def execute(self) -> None:
        """Execute placeholder unsupervised streaming experiment.

        Returns:
            None: Not implemented.
        """
        pass