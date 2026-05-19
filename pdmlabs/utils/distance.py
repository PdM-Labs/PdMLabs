"""Distance measure classes for time-series anomaly detection.

This module provides advanced distance metrics designed for time-series anomaly detection,
particularly for reconstruction-based and distance-based methods. Each class measures
the dissimilarity between observed and predicted/expected subsequences.

Classes:
    Euclidean: Lp norm-based distance with optional normalization
    Mahalanobis: Covariance-based distance for multivariate time-series
    Garch: Volatility-adjusted distance using ARCH/GARCH models
    SSA_DISTANCE: Singular Spectrum Analysis distance for contextual anomalies

Usage:
    The distance classes are typically used with time-series anomaly detectors
    that have train data (X_train_), a window size, and estimation/prediction errors.
    
    Steps:
    1. Create distance object: dist = Euclidean(power=2, neighborhood=100)
    2. Set detector: dist.set_param() after assigning detector
    3. Measure: score = dist.measure(X_real, X_predicted, index)

Example:
    >>> from pdmlabs.utils.distance import Euclidean
    >>> euclidean_dist = Euclidean(power=2, norm=True)
    >>> euclidean_dist.detector = my_detector  # Set detector first
    >>> euclidean_dist.set_param()
    >>> score = euclidean_dist.measure(X_obs, X_pred, sample_index)
"""

# Author: Yinchen Wu <yinchen@uchicago.edu>

import numpy as np
from arch import arch_model
import math


class Euclidean:
    """Lp norm distance measure for reconstruction-based anomaly detection.
    
    Computes the Lp norm (L1, L2, etc.) between observed and predicted time-series
    subsequences. Can optionally normalize by local neighborhood statistics to
    account for varying baseline behavior.
    
    Parameters
    ----------
    power : int, default=1
        The power p for the Lp norm. power=1 is Manhattan distance, power=2 is Euclidean.
    neighborhood : int, default=100
        Size of neighborhood window (±samples around each point) used to compute
        normalization factor D (difference between max and min in neighborhood).
        If None, skips normalization.
    window : int, default=20
        Length of each subsequence being compared.
    norm : bool, default=False
        If True, normalizes distance by local neighborhood statistics (dividing by D).
        If False, returns raw Lp norm.
    
    Attributes
    ----------
    decision_scores_ : list
        List of (index, score) tuples generated during measure() calls.
    detector : object
        Reference to the anomaly detector providing X_train_, window, neighborhood.
    
    Examples
    --------
    >>> # Unnormalized L2 distance
    >>> euclidean = Euclidean(power=2, norm=False)
    >>> score = euclidean.measure(observed_X, predicted_X, start_index)
    
    >>> # Normalized distance with neighborhood statistics
    >>> euclidean_norm = Euclidean(power=2, neighborhood=50, norm=True)
    >>> euclidean_norm.detector = detector  # Must set detector first
    >>> euclidean_norm.set_param()
    >>> score = euclidean_norm.measure(X_real, X_pred, idx)
    
    Notes
    -----
    Without normalization, the metric treats all points equally regardless of their
    local context. Normalization accounts for natural variations in the data by
    dividing by the range observed in nearby samples.
    """
    def __init__(self, power = 1, neighborhood = 100, window = 20, norm = False):
        self.power = power
        self.window = window
        self.neighborhood = neighborhood
        self.detector = None
        self.decision_scores_  = []
        self.norm = norm
        self.X_train = 2

    def measure(self, X, Y, index):
        """Calculate distance anomaly score between observed and expected subsequences.
        
        Computes the Lp norm between X and Y, optionally normalized by neighborhood
        statistics. Higher scores indicate greater dissimilarity (anomaly).
        
        Parameters
        ----------
        X : np.ndarray
            Observed/actual subsequence values (1D array).
        Y : np.ndarray
            Expected/predicted subsequence values (1D array, same length as X).
        index : int
            Position in the training data where this window starts.
            Used to identify the neighborhood region for normalization.
        
        Returns
        -------
        float
            Anomaly score (distance). Range depends on normalization:
            - Without norm: typically 0 to infinity
            - With norm: typically 0 to 1 (normalized)
        
        Notes
        -----
        - Stores result in self.decision_scores_ list for analysis
        - For normalized mode: D = max(neighborhood) - min(neighborhood)
        - Handles edge cases: empty arrays, single dimensions, boundary indices
        
        Examples
        --------
        >>> dist = Euclidean(power=2)
        >>> dist.detector = detector
        >>> dist.set_param()
        >>> X_real = np.array([1.0, 2.0, 3.0])
        >>> X_pred = np.array([1.1, 2.1, 2.9])
        >>> score = dist.measure(X_real, X_pred, 100)  # ~0.14
        """
        X_train = self.X_train
        X_train = self.detector.X_train_
        power = self.power
        
        window = self.window
        neighborhood = self.neighborhood
        norm = self.norm
        data = X_train
        if norm == False:
            if X.shape[0] == 0:
                score = 0
            else:
                score = np.linalg.norm(X-Y, power)/(X.shape[0])
            self.decision_scores_.append((index, score))
            return score
        elif type(X_train) == int:
            print('Error! Detector is not fed to the object and X_train is not known')
        elif neighborhood != 'all':
            neighbor = int(self.neighborhood/2)

            if index + neighbor < self.n_train_ and index - neighbor > 0: 
                region = np.concatenate((data[index - neighbor: index], data[index + window: index + neighbor] ))
                D = np.max(region) - np.min(region)
            elif index + neighbor >= self.n_train_ and index + window < self.n_train_:
                region = np.concatenate((data[self.n_train_ - neighborhood: index], data[index + window: self.n_train_] ))
                D =  np.max(region) - np.min(region)   
            elif index + window >= self.n_train_:
                region = data[self.n_train_ - neighborhood: index]
                D = np.max(region) - np.min(region) 
            else:
                region = np.concatenate((data[0: index], data[index + window: index + neighborhood] ))
                D = np.max(region) - np.min(region) 
            
            score = np.linalg.norm(X-Y, power)/D/(X.shape[0]**power)
            self.decision_scores_.append((index, score))
            return score
    def set_param(self):
        """Initialize parameters from detector object.
        
        Extracts window, neighborhood, training data, and other properties
        from the detector to ensure consistency. Must be called after
        setting the detector attribute.
        
        Returns
        -------
        self
            Returns self for method chaining.
        
        Raises
        ------
        AttributeError
            If detector is None or lacks required attributes.
        """
        if self.detector != None:
            self.window = self.detector.window
            self.neighborhood = self.detector.neighborhood
            self.n_train_ = self.detector.n_train_
            self.X_train = self.detector.X_train_
        else:
            print('Error! Detector is not fed to the object and X_train is not known')
        return self
                

class Mahalanobis:
    """Mahalanobis distance measure accounting for covariance structure.
    
    Computes Mahalanobis distance which measures dissimilarity while accounting
    for correlations between dimensions. Uses covariance matrix computed from
    residuals (X_train - estimation) in a neighborhood window.
    
    Parameters
    ----------
    probability : bool, default=False
        If False: Returns Mahalanobis distance (quadratic form with inverse covariance)
        If True: Returns probability via multivariate/univariate normal PDF
    
    Attributes
    ----------
    detector : object
        Reference to anomaly detector with X_train_, estimation, window, n_initial_.
    decision_scores_ : list
        List of (index, score) tuples.
    cov : np.ndarray
        Covariance matrix of residuals (computed in set_param).
    mu : np.ndarray
        Mean vector (typically zeros).
    
    Examples
    --------
    >>> # Mahalanobis distance
    >>> maha = Mahalanobis(probability=False)
    >>> maha.detector = detector
    >>> maha.set_param()
    >>> score = maha.measure(X_obs, X_pred, idx)
    
    >>> # Probability-based scoring
    >>> maha_prob = Mahalanobis(probability=True)
    >>> maha_prob.detector = detector
    >>> maha_prob.set_param()
    >>> prob_score = maha_prob.measure(X_obs, X_pred, idx)
    
    Notes
    -----
    Mahalanobis distance is particularly useful when features have different scales
    or when correlations between features are important for anomaly detection.
    """
    def __init__(self, probability = False):
        self.probability = probability
        self.detector = None
        self.decision_scores_  = []
        self.mu = 0
        
    def set_param(self):
        '''update the parameters with the detector that is used 
        '''

        self.n_initial_ = self.detector.n_initial_
        self.estimation = self.detector.estimation
        self.X_train = self.detector.X_train_
        self.window = self.detector.window
        window = self.window
        resid = self.X_train - self.estimation
        number = max(100, self.window)
        self.residual = np.zeros((window, number))
        for i in range(number):
            self.residual[:, i] = resid[self.n_initial_+i:self.n_initial_+i+window]
        self.mu = np.zeros(number)
        self.cov = np.cov(self.residual, rowvar=1)
        if self.window == 1:
            self.cov = (np.sum(np.square(self.residual))/(number - 1))**0.5
        return self
    def norm_pdf_multivariate(self, x):
        '''multivarite normal density function
        '''
        try:
            mu = self.mu
        except:
            mu = np.zeros(x.shape[0])
        sigma = self.cov
        size = x.shape[0]
        if size == len(mu) and (size, size) == sigma.shape:
            det = np.linalg.det(sigma)
            if det == 0:
                raise NameError("The covariance matrix can't be singular")

            norm_const = 1.0/ ( math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2) )
            x_mu = np.matrix(x - mu)
            inv = np.linalg.inv(sigma)        
            result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")
    def normpdf(self, x):
        '''univariate normal
        '''
        mean = 0
        sd = np.asscalar(self.cov)
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom 

    def measure(self, X, Y, index):
        """Derive the decision score based on the given distance measure 
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The real input samples subsequence.
        Y : numpy array of shape (n_samples, )
            The estimated input samples subsequence.
        Index : int
        the index of the starting point in the subsequence
        Returns
        -------
        score : float
            dissimiarity score between the two subsquence
        """
        mu = np.zeros(self.detector.window)
        cov = self.cov
        if self.probability == False:

            if X.shape[0] == mu.shape[0]:
                score = np.matmul(np.matmul((X-Y-mu).T, cov), (X-Y-mu))/(X.shape[0])
                self.decision_scores_.append((index, score))
                return score
            else:
                return (X-Y).T.dot(X-Y)

        else:
            if len(X) > 1:
                prob = self.norm_pdf_multivariate(X-Y)
            elif len(X) == 1: 
                X = np.asscalar(X)
                Y = np.asscalar(Y)
                prob = self.normpdf(X-Y)
            else:
                prob = 1
            score = 1 - prob
            score = max(score, 0)
            self.decision_scores_.append((index, score))
            return score


class Garch:
    """GARCH-based distance measure using volatility modeling.
    
    Models residual volatility using ARCH/GARCH (Generalized AutoRegressive Conditional
    Heteroskedasticity) models. Divides reconstruction error by the modeled volatility
    to create volatility-adjusted anomaly scores.
    
    Parameters
    ----------
    p : int, default=1
        ARCH order (number of lagged squared errors in volatility equation).
    q : int, default=1
        GARCH order (number of lagged variance terms in volatility equation).
    mean : str, default='zero'
        Mean model: 'Zero', 'Constant', 'AR', 'ARX', 'ARMAX', etc.
    vol : str, default='garch'
        Volatility model: 'GARCH', 'ARCH', 'ConstantMean', 'ZeroMean', etc.
    
    Attributes
    ----------
    decision_scores_ : list
        List of (index, score) tuples.
    volatility : np.ndarray
        Estimated volatility at each timepoint.
    
    Examples
    --------
    >>> garch_dist = Garch(p=1, q=1, mean='zero', vol='garch')
    >>> garch_dist.detector = detector
    >>> garch_dist.set_param()  # Fits GARCH model on training residuals
    >>> score = garch_dist.measure(X_real, X_pred, idx)
    
    Notes
    -----
    - GARCH models are effective for time-series with changing volatility
    - Residuals are scaled by 10 during fitting for numerical stability
    - Volatility-adjusted errors better capture anomalies in volatile regimes
    - More computationally expensive than simpler distance measures
    """
    def __init__(self, p = 1, q = 1, mean = 'zero', vol = 'garch'):
        self.p = p
        self.q = q
        self.vol = vol
        self.mean = mean
        self.decision_scores_  = []
        
    def set_param(self):
        '''update the parameters with the detector that is used 
        '''
        q = self.q
        p=self.p
        mean = self.mean
        vol = self.vol
        if self.detector != None:
            self.n_initial_ = self.detector.n_initial_
            self.estimation = self.detector.estimation
            self.X_train = self.detector.X_train_
            self.window = self.detector.window
            resid = 10 * (self.X_train - self.estimation)
            model = arch_model(resid, mean=mean, vol=vol, p=p, q=q)
            model_fit = model.fit(disp='off')
            self.votility = model_fit.conditional_volatility/10
        else:
            print('Error! Detector not fed to the measure')
        return self

    def measure(self, X, Y, index):
        """Derive the decision score based on the given distance measure 
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The real input samples subsequence.
        Y : numpy array of shape (n_samples, )
            The estimated input samples subsequence.
        Index : int
        the index of the starting point in the subsequence
        Returns
        -------
        score : float
            dissimiarity score between the two subsquences
        """
        X = np.array(X)
        Y = np.array(Y)
        length = len(X)
        score = 0
        if length != 0:
            for i in range(length):
                sigma = self.votility[index + i]
                if sigma != 0:
                    score += abs(X[i]-Y[i])/sigma
                    
            score = score/length       
        return score


class SSA_DISTANCE:
    """ The function class for SSA measure
    good for contextual anomolies
    ----------
    method : string, optional (default='linear' )
        The method to fit the line and derives the SSA score
    e: float, optional (default = 1)
        The upper bound to start new line search for linear method
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    detector: Object classifier
        the anomaly detector that is used
    """
    def __init__(self, method ='linear', e = 1):
        self.method = method
        self.decision_scores_  = []
        self.e = e
    def Linearization(self, X2):
        """Obtain the linearized curve.
        Parameters
        ----------
        X2 : numpy array of shape (n, )
            the time series curve to be fitted
        e: float, integer, or numpy array 
        weights to obtain the 
        Returns
        -------
        fit: parameters for the fitted linear curve
        """
        e = self.e
        i = 0
        fit = {}
        fit['index'] = []
        fit['rep'] = []
        while i < len(X2):
            fit['index'].append(i)
            try:
                fit['Y'+str(i)]= X2[i]
            except:
                print(X2.shape, X2)
            fit['rep'].append(np.array([i, X2[i]]))
            if i+1 >= len(X2):
                    break
            k = X2[i+1]-X2[i]
            b = -i*(X2[i+1]-X2[i])+X2[i]
            fit['reg' +str(i)]= np.array([k, b])
            i += 2
            if i >= len(X2):
                break
            d = np.abs(X2[i]- (k * i+b))
            while d < e:
                i +=1 
                if i >= len(X2):
                    break
                d = np.abs(X2[i]- (k * i+b)) 
        return fit   
    def set_param(self):
        '''update the parameters with the detector that is used. 
        Since the SSA measure doens't need the attributes of detector
        or characteristics of X_train, the process is omitted. 
        '''

        return self

    def measure(self, X2, X3, start_index):
        """Obtain the SSA similarity score.
        Parameters
        ----------
        X2 : numpy array of shape (n, )
            the reference timeseries
        X3 : numpy array of shape (n, )
            the tested timeseries
        e: float, integer, or numpy array 
        weights to obtain the 
        Returns
        -------
        score: float, the higher the more dissimilar are the two curves 
        """       
        #linearization of data X2 and X3
        X2 = np.array(X2)
        X3 = np.array(X3)

        fit = self.Linearization(X2)
        fit2 = self.Linearization(X3)
    
        #line alinement 
        Index = []
        test_list = fit['index'] + fit2['index']
        [Index.append(x) for x in test_list if x not in Index]
        Y = 0
    
        #Similarity Computation
        for i in Index:
            if i in fit['index'] and i in fit2['index']:
                Y += abs(fit['Y'+str(i)]-fit2['Y'+str(i)])

            elif i in fit['index']:
                J = np.max(np.where(np.array(fit2['index']) < i ))
                index = fit2['index'][J]
                k = fit2['reg'+str(index)][0]
                b = fit2['reg'+str(index)][1]
                value = abs(k * i + b - fit['Y'+str(i)])
                Y += value
            elif i in fit2['index']:
                J = np.max(np.where(np.array(fit['index']) < i ))
                index = fit['index'][J]
                k = fit['reg'+str(index)][0]
                b = fit['reg'+str(index)][1]
                value = abs(k * i + b - fit2['Y'+str(i)])
                Y += value
        if len(Index) != 0: 
            score = Y/len(Index)
        else:
            score = 0
        self.decision_scores_.append((start_index, score))
        if len(X2) == 1:
            print('Error! SSA measure doesn\'t apply to singleton' )
        else:
            return score  


class Fourier:
    """ The function class for Fourier measure
    good for contextual anomolies
    ----------
    power: int, optional (default = 2)
        Lp norm for dissimiarlity measure considered
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    detector: Object classifier
        the anomaly detector that is used
    """
    def __init__(self, power = 2):
        self.decision_scores_  = []
        self.power = power
    def set_param(self):
        '''update the parameters with the detector that is used 
        since the FFT measure doens't need the attributes of detector
        or characteristics of X_train, the process is omitted. 
        '''

        return self

    def measure(self, X2, X3, start_index):
        """Obtain the SSA similarity score.
        Parameters
        ----------
        X2 : numpy array of shape (n, )
            the reference timeseries
        X3 : numpy array of shape (n, )
            the tested timeseries
        index: int, 
        current index for the subseqeuence that is being measured 
        Returns
        -------
        score: float, the higher the more dissimilar are the two curves 
        """       
 
        power = self.power
        X2 = np.array(X2)
        X3 = np.array(X3)
        if len(X2) == 0:
            score = 0
        else:
            X2 = np.fft.fft(X2);
            X3 = np.fft.fft(X3)
            score = np.linalg.norm(X2 - X3, ord = power)/len(X3)
        self.decision_scores_.append((start_index, score))
        return score


class DTW:
    """ The function class for dynamic time warping measure

    ----------
    method : string, optional (default='L2' )
        The distance measure to derive DTW.
        Avaliable "L2", "L1", and custom
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    detector: Object classifier
        the anomaly detector that is used
    """
    def __init__(self, method = 'L2'):
        self.decision_scores_  = []
        if type(method) == str:
            if method == 'L1':
                distance = lambda x, y: abs(x-y)
            elif method == 'L2':
                distance = lambda x, y: (x-y)**2
        else:
            distance = method
        self.distance = distance
    def set_param(self):
        '''update the parameters with the detector that is used 
        since the FFT measure doens't need the attributes of detector
        or characteristics of X_train, the process is omitted. 
        '''

        return self

    def measure(self, X1, X2, start_index):
        """Obtain the SSA similarity score.
        Parameters
        ----------
        X1 : numpy array of shape (n, )
            the reference timeseries
        X2 : numpy array of shape (n, )
            the tested timeseries
        index: int, 
        current index for the subseqeuence that is being measured 
        Returns
        -------
        score: float, the higher the more dissimilar are the two curves 
        """       
        distance = self.distance
        X1 = np.array(X1)
        X2 = np.array(X2)
        
        value = 1
        if len(X1)==0:
            value =0
            X1= np.zeros(5)
            X2 = X1
        M = np.zeros((len(X1), len(X2)))
        for index_i in range(len(X1)):
            for index_j in range(len(X1) - index_i):
                L = []
                i = index_i
                j = index_i + index_j
                D = distance(X1[i], X2[j])
                try:
                    L.append(M[i-1, j-1])
                except:
                    L.append(np.inf)
                try:
                    L.append(M[i, j-1])
                except:
                    L.append(np.inf)
                try:
                    L.append(M[i-1, j])
                except:
                    L.append(np.inf)
                D += min(L)
                M[i,j] = D
                if i !=j:
                    L = []
                    j = index_i
                    i = index_i + index_j
                    D = distance(X1[i], X2[j])
                    try:
                        L.append(M[i-1, j-1])
                    except:
                        L.append(np.inf)
                    try:
                        L.append(M[i, j-1])
                    except:
                        L.append(np.inf)
                    try:
                        L.append(M[i-1, j])
                    except:
                        L.append(np.inf)
                    D += min(L)
                    M[i,j] = D
        
        score = M[len(X1)-1, len(X1)-1]/len(X1)
        if value == 0:
            score = 0
        self.decision_scores_.append((start_index, score))
        return score


class EDRS:
    """ The function class for edit distance on real sequences 

    ----------
    method : string, optional (default='L2' )
        The distance measure to derive DTW.
        Avaliable "L2", "L1", and custom
    ep: float, optiona (default = 0.1)
        the threshold value to decide Di_j
    vot : boolean, optional (default = False)
        whether to adapt a chaging votilities estimaed by garch
        for ep at different windows. 
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    detector: Object classifier
        the anomaly detector that is used
    """
    def __init__(self, method = 'L1', ep = False, vol = False):
        self.decision_scores_  = []
        if type(method) == str:
            if method == 'L1':
                distance = lambda x, y: abs(x-y)
        else:
            distance = method
        self.distance = distance
        self.ep = ep
        self.vot = vol
    def set_param(self):
        '''update the ep based on the votalitiy of the model 
        '''
        estimation = np.array(self.detector.estimation )
        initial = self.detector.n_initial_
        X = np.array(self.detector.X_train_)
        self.initial = initial
        residual = estimation[initial:] - X[initial:]
        # number = len(residual)
        # var = (np.sum(np.square(residual))/(number - 1))**0.5
        vot = self.vot
        if vot == False:
            var = np.var(residual)
        else:
            model = arch_model(10 * residual, mean='Constant', vol='garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            var = model_fit.conditional_volatility/10
            
        if self.ep == False:
            self.ep =  3 * (np.sum(np.square(residual))/(len(residual) - 1))**0.5
        else: 
            self.ep = self.ep
        
        return self

    def measure(self, X1, X2, start_index):
        """Obtain the SSA similarity score.
        Parameters
        ----------
        X1 : numpy array of shape (n, )
            the reference timeseries
        X2 : numpy array of shape (n, )
            the tested timeseries
        index: int, 
        current index for the subseqeuence that is being measured 
        Returns
        -------
        score: float, the higher the more dissimilar are the two curves 
        """       
        distance = self.distance
        X1 = np.array(X1)
        X2 = np.array(X2)
        vot = self.vot

        if vot == False:
            ep = self.ep
        else:
            try:
                ep = self.ep[start_index - self.initial]
            except:
                #sometime start_index is the length of the number 
                ep = 0
        value = 1
        if len(X1)==0:
            value =0
            X1= np.zeros(5)
            X2 = X1
        M = np.zeros((len(X1), len(X2)))
        M[:, 0] = np.arange(len(X1))
        M[0, :] = np.arange(len(X1))
        for index_i in range(1, len(X1)):
            for index_j in range(len(X1) - index_i):

                L = []
                i = index_i
                j = index_i + index_j
                D = distance(X1[i], X2[j])
                if D < ep:
                    M[i, j]= M[i-1, j-1]
                else:
                    try:
                        L.append(M[i-1, j-1])
                    except:
                        L.append(np.inf)
                    try:
                        L.append(M[i, j-1])
                    except:
                        L.append(np.inf)
                    try:
                        L.append(M[i-1, j])
                    except:
                        L.append(np.inf)
                    M[i,j] = 1 + min(L)
                if i !=j:
                    L = []
                    j = index_i
                    i = index_i + index_j
                    D = distance(X1[i], X2[j])
                    if D < ep:
                        M[i, j]= M[i-1, j-1]
                    else: 
                        try:
                            L.append(M[i-1, j-1])
                        except:
                            L.append(np.inf)
                        try:
                            L.append(M[i, j-1])
                        except:
                            L.append(np.inf)
                        try:
                            L.append(M[i-1, j])
                        except:
                            L.append(np.inf)
                        M[i,j] = 1 + min(L)

        score = M[len(X1)-1, len(X1)-1]/len(X1)
        if value == 0:
            score = 0
        self.decision_scores_.append((start_index, score))
        return score

class TWED:
    """ Function class for Time-warped edit distance(TWED) measure

    ----------
    method : string, optional (default='L2' )
        The distance measure to derive DTW.
        Avaliable "L2", "L1", and custom
    gamma: float, optiona (default = 0.1)
        mismatch penalty
    v : float, optional (default = False)
        stifness parameter
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    detector: Object classifier
        the anomaly detector that is used
    """
    def __init__(self, gamma = 0.1, v = 0.1):
        self.decision_scores_  = []

        self.gamma = gamma
        self.v = v
    def set_param(self):
        '''No need'''     
        return self
    
    def measure(self, A, B, start_index):
        """Obtain the SSA similarity score.
        Parameters
        ----------
        X1 : numpy array of shape (n, )
            the reference timeseries
        X2 : numpy array of shape (n, )
            the tested timeseries
        index: int, 
        current index for the subseqeuence that is being measured 
        Returns
        -------
        score: float, the higher the more dissimilar are the two curves 
        """    
        #code modifed from wikipedia
        Dlp = lambda x,y: abs(x-y)
        timeSB = np.arange(1,len(B)+1)
        timeSA = np.arange(1,len(A)+1)
        nu = self.v
        _lambda = self.gamma
        # Reference :
        #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
        #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
        #    http://people.irisa.fr/Pierre-Francois.Marteau/

        # Check if input arguments
        if len(A) != len(timeSA):
            print("The length of A is not equal length of timeSA")
            return None, None
    
        if len(B) != len(timeSB):
            print("The length of B is not equal length of timeSB")
            return None, None

        if nu < 0:
            print("nu is negative")
            return None, None

        # Add padding
        A = np.array([0] + list(A))
        timeSA = np.array([0] + list(timeSA))
        B = np.array([0] + list(B))
        timeSB = np.array([0] + list(timeSB))

        n = len(A)
        m = len(B)
        # Dynamical programming
        DP = np.zeros((n, m))

        # Initialize DP Matrix and set first row and column to infinity
        DP[0, :] = np.inf
        DP[:, 0] = np.inf
        DP[0, 0] = 0

        # Compute minimal cost
        for i in range(1, n):
            for j in range(1, m):
                # Calculate and save cost of various operations
                C = np.ones((3, 1)) * np.inf
                # Deletion in A
                C[0] = (
                    DP[i - 1, j]
                    + Dlp(A[i - 1], A[i])
                    + nu * (timeSA[i] - timeSA[i - 1])
                    + _lambda
                )
                # Deletion in B
                C[1] = (
                    DP[i, j - 1]
                    + Dlp(B[j - 1], B[j])
                    + nu * (timeSB[j] - timeSB[j - 1])
                    + _lambda
                )
                # Keep data points in both time series
                C[2] = (
                    DP[i - 1, j - 1]
                    + Dlp(A[i], B[j])
                    + Dlp(A[i - 1], B[j - 1])
                    + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
                )
                # Choose the operation with the minimal cost and update DP Matrix
                DP[i, j] = np.min(C)
        distance = DP[n - 1, m - 1]
        self.M = DP
        self.decision_scores_.append((start_index, distance))
        return distance