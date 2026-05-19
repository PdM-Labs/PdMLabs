"""Unsupervised anomaly detection method interface.

UnsupervisedMethodInterface extends MethodInterface for methods that don't require
labeled training data. No fit() method is needed - methods learn directly from
unlabeled feature distributions or patterns.

Unsupervised learning assumes:
- Normal behavior dominates the data (majority of samples are normal)
- Anomalies are rare and deviate from learned patterns
- No explicit labels are required

This contrasts with supervised methods that require labeled time periods.

Typical unsupervised methods:
- Isolation Forest: isolates anomalies through random partitioning
- Local Outlier Factor (LOF): density-based anomaly detection
- One-Class SVM: learns single-class boundary
- Autoencoders: learns reconstruction of normal data
- k-Nearest Neighbors: distance to k-nearest neighbors
"""

from pdmlabs.method.method import MethodInterface

class UnsupervisedMethodInterface(MethodInterface):
    """Base class for unsupervised anomaly detection methods.
    
    Unsupervised methods don't require labeled training data. They can be called
    directly with predict() without fit(), though many implementations may use
    predict() data to learn or calibrate internally.
    
    Common patterns for unsupervised methods:
    
    Pattern 1 - Distribution learning:
        Learn normal data distribution from initial samples, then detect deviations.
        
    Pattern 2 - Reference-based:
        Use provided test data as reference, detect outliers vs reference.
        
    Pattern 3 - Parameter-driven:
        Use fixed hyperparameters without learning from data (e.g., k-NN with k=5).
    
    Examples of unsupervised methods:
    - Isolation Forest (tree-based)
    - One-Class SVM (margin-based)
    - LOF (density-based)
    - Autoencoder (reconstruction-based)
    - Statistical (z-score, Mahalanobis)
    
    Examples:
        >>> from pdmlabs.method.unsupervised_method import UnsupervisedMethodInterface
        >>> 
        >>> # Direct prediction (no fit needed)
        >>> method = IsolationForest(event_preferences={...})
        >>> scores = method.predict(df_test, 'bearing_1', events_df)
        >>> 
        >>> # Or with internal calibration
        >>> method = SomeAutoencoder(event_preferences={...})
        >>> # Internal learning from df_test happens in predict()
        >>> scores = method.predict(df_test, 'bearing_1', events_df)
    """
    pass