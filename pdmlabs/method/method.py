"""Abstract base interface for anomaly detection methods.

All anomaly detection methods (supervised, unsupervised, semi-supervised, etc.)
must implement the MethodInterface to integrate with the framework.

The interface supports two prediction modes:
1. Batch mode: predict() processes entire datasets at once
2. Online/streaming mode: predict_one() processes single samples

All methods must return anomaly scores (higher = more anomalous).
Post-processors can optionally convert scores to binary labels.

Typical usage pipeline:
  1. Create method instance with event_preferences
  2. Call fit() (if supervised/semi-supervised)
  3. Call predict() or predict_one() to get anomaly scores
  4. Optionally apply post-processing for binary labels
"""

import abc

import pandas as pd

from pdmlabs.pdm_evaluation_types.types import EventPreferences


class MethodInterface(abc.ABC):
    """Abstract base class for all anomaly detection methods.
    
    Defines the contract that all anomaly detection implementations must follow.
    Supports both batch and streaming prediction modes with source-specific logic.
    
    Attributes:
        event_preferences (EventPreferences): Event configuration dict.
    """
    def __init__(self, event_preferences: EventPreferences):
        """Initialize anomaly detection method.
        
        Args:
            event_preferences (EventPreferences): Event configuration dictionary.
        """
        self.event_preferences = event_preferences
        

    @abc.abstractmethod
    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> list:
        """Predict anomaly scores for batch of samples (offline mode).

        Computes anomaly score for each row in target_data independently.
        Higher scores indicate more anomalous behavior.

        Args:
            target_data (pd.DataFrame): Feature matrix with dates in index,
                features in columns. Must have same features as training data.
            source (str): Source identifier (e.g., 'bearing_1'). Used to select
                source-specific model if method maintains multiple models.
            event_data (pd.DataFrame): Event log with columns 'date', 'type',
                'source', 'description'. Can be used for context-aware scoring.

        Returns:
            list: Anomaly scores (float) with length = target_data.shape[0].
                Score range and semantics depend on method:
                - Distance-based: typically [0, ∞) where higher = more anomalous
                - Probability-based: typically [0, 1] or (-∞, 0] log-likelihood
                - Reconstruction-based: typically [0, ∞) reconstruction error
                
        Examples:
            >>> method = SomeAnomalyDetector(event_preferences={...})
            >>> method.fit([df_train], ['bearing_1'], events_df, labels)
            >>> df_test = pd.DataFrame([feature values], index=[dates])
            >>> scores = method.predict(df_test, 'bearing_1', events_df)
            >>> print(len(scores), scores[0])  # (100, 0.45)
            
        Raises:
            NotImplementedError: If method hasn't been fit (for supervised methods).
        """
        pass


    @abc.abstractmethod
    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:
        """Predict anomaly score for single sample (online/streaming mode).

        Computes anomaly score for one observation at a time. Useful for:
        - Real-time anomaly detection
        - Online learning scenarios
        - Memory-efficient processing
        
        May maintain internal state (buffers, windows) for context-aware scoring.

        Args:
            new_sample (pd.Series): Single observation with feature values.
                Index should contain feature names matching training data.
            source (str): Source identifier for source-specific models.
            is_event (bool): Whether this sample is from an event timestamp.
                Can affect how method processes the sample (e.g., special handling
                for known events vs normal operation).

        Returns:
            float: Anomaly score for this single sample (same scale as predict()).

        Examples:
            >>> method = SomeAnomalyDetector(event_preferences={...})
            >>> method.fit([df_train], ['bearing_1'], events_df, labels)
            >>> 
            >>> # Online scoring
            >>> for idx, row in df_test.iterrows():
            ...     is_event = idx in event_timestamps
            ...     score = method.predict_one(row, 'bearing_1', is_event)
            ...     print(f'{idx}: {score}')
        """
        pass

    @abc.abstractmethod
    def get_library(self) -> str:
        """Return the underlying library/framework name.
        
        Returns:
            str: Name of library used (e.g., 'sklearn', 'torch', 'custom').
                Used for dependency tracking and method categorization.
        """
        pass


    @abc.abstractmethod
    def get_params(self) -> dict:
        """Return hyperparameters and configuration.
        
        Returns:
            dict: Dictionary of hyperparameters (e.g.,
                {'n_neighbors': 5, 'contamination': 0.1}).
                Useful for logging, model comparison, and reproducibility.
        """
        pass


    @abc.abstractmethod
    def __str__(self) -> str:
        """Return human-readable method name.

        Returns:
            str: Method name (e.g., 'Isolation_Forest', 'LOF', 'AutoEncoder').
                Used for logging, result files, and UI display.
        """
        pass


    @abc.abstractmethod
    def get_all_models(self):
        """Return reference to internal model(s).
        
        Returns model instances for inspection, export, or further processing.
        Structure depends on method - may return single model or dict of models.
        
        Returns:
            model or dict: Underlying model object(s). Examples:
                - Single model: sklLearn model instance
                - Multiple models: {'bearing_1': model1, 'bearing_2': model2}
                - None: If model is not accessible/applicable
        """
        pass

    
    def destruct(self):
        """Clean up resources (optional).
        
        Called when method is no longer needed. Implementations can override
        to clean up GPU memory, temporary files, etc.
        """
        pass