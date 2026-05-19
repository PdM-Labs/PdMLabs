"""Supervised anomaly detection method interface.

SupervisedMethodInterface extends MethodInterface to add the fit() method that
enables training on labeled data. All supervised anomaly detectors must implement
this interface.

Supervised learning requires:
- Training data (features)
- Anomaly labels (0=normal, 1=anomaly OR time ranges marking anomalous periods)
- Event information (optional but recommended)

This contrasts with unsupervised methods that require no labels.
"""

import abc

import pandas as pd

from pdmlabs.method.method import MethodInterface

class SupervisedMethodInterface(MethodInterface):
    """Base class for supervised anomaly detection methods.
    
    Supervised methods learn from labeled training data to distinguish normal
    vs anomalous behavior. Requires fit() before predict() can be called.
    
    Training data preparation:
    - Features: normalized, preprocessed sensor readings
    - Labels: binary (0/1) or time ranges marking anomalous periods
    - Multiple sources: separate training data per sensor/device
    
    Examples of supervised methods:
    - Neural networks (autoencoder, LSTM)
    - XGBoost/gradient boosting
    - SVM (with RBF kernel for score calibration)
    - Decision trees (with confidence normalization)
    
    Examples:
        >>> from pdmlabs.method.supervised_method import SupervisedMethodInterface
        >>> 
        >>> # Training phase
        >>> method = SomeNeuralNetworkDetector(event_preferences={...})
        >>> method.fit(
        ...     historic_data=[df_train_bearing1, df_train_bearing2],
        ...     historic_sources=['bearing_1', 'bearing_2'],
        ...     event_data=events_df,
        ...     anomaly_ranges=[[idx1:idx2], [idx3:idx4]]  # Labeled anomalies
        ... )
        >>> 
        >>> # Prediction phase
        >>> scores = method.predict(df_test, 'bearing_1', events_df)
    """
    
    @abc.abstractmethod
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges: list[list]) -> None:
        """Train anomaly detection model on labeled data.

        Fits the model to distinguish normal from anomalous behavior using
        provided training data and labels.

        Args:
            historic_data (list[pd.DataFrame]): List of training DataFrames,
                one per source. Each DataFrame has:
                - Columns: feature names (sensor readings, computed metrics)
                - Index: datetime (must be sorted)
                - Shape: (num_samples, num_features)
                All DataFrames should have same features.

            historic_sources (list[str]): Source identifiers corresponding to
                historic_data. Example: ['bearing_1', 'bearing_2', 'pump_1'].
                Length must match len(historic_data).

            event_data (pd.DataFrame): Event log for context. Columns should
                include 'date', 'type' ('failure', 'reset', etc.), 'source',
                and optional 'description'. Can help training identify event
                patterns or validate training data selection.

            anomaly_ranges (list[list]): Labels marking anomalous time periods.
                Structure: list of lists where element i corresponds to
                historic_sources[i]. Each element is list of time indices or
                ranges marking anomalies. Examples:
                - Index-based: [[10:50, 100:120], [5:30]]  (indices start:end)
                - Boolean: [pd.Series([0,1,0,...]), pd.Series([0,0,1,...])]
                
                Use case: If bearing_1 has anomalies from idx 10-50,
                anomaly_ranges[0] includes that range.

        Returns:
            None: Modifies internal state to store fitted model.

        Raises:
            ValueError: If data shapes don't match or labels invalid.
            NotImplementedError: Implementation not complete (abstract).

        Notes:
            - Implementations may copy historic_data if they need to store it
            - Training may take significant time for large datasets
            - After fit(), predict() can be called
            - Multiple fit() calls should retrain (not append)

        Examples:
            >>> # Binary labels (0/1 for each sample)
            >>> labels = [np.array([0, 0, 1, 1, 0, ...]), np.array([0, 0, 0, ...])]
            >>> method.fit(data, sources, events, labels)
            >>> 
            >>> # Time ranges (list of anomalous periods)
            >>> anomaly_periods = [
            ...     [(start_idx1, end_idx1), (start_idx2, end_idx2)],
            ...     [(start_idx3, end_idx3)],
            ... ]
            >>> method.fit(data, sources, events, anomaly_periods)
        """
        pass