import abc

class EvaluatorInterface(abc.ABC):
    """
    Interface for implementing custom evaluation metrics.
    
    Users can implement this interface to inject their own metrics into
    the PdMExperiment evaluation pipeline. Evaluators must compute metrics
    and return them as a dictionary. They can also optionally log artifacts
    (e.g., plots or CSV files) to the active MLflow run.
    """
    
    @abc.abstractmethod
    def evaluate(self, experiment, **kwargs) -> dict:
        """
        Compute evaluation metrics and return them.
        
        Args:
            experiment: The PdMExperiment instance running the evaluation.
            **kwargs: Additional data required for evaluation depending on the experiment flavor.
                      Common kwargs include:
                      - result_scores: The predicted scores/values.
                      - result_dates: The timestamps corresponding to the predictions.
                      - results_isfailure / result_labels: Ground-truth labels or failure indicators.
                      - plot_dictionary: A dictionary to store plotting data if debug=True.
                      - rtfs: Run-to-failure indicators (for RUL/SA).
                      
        Returns:
            dict: A dictionary mapping metric names (str) to their computed values.
                  Example: {'my_custom_f1': 0.85, 'my_custom_auc': 0.92}
        """
        pass
