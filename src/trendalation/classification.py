import numpy as np
from sklearn import metrics
from scipy.spatial import procrustes

# For interpolation purposes
# import scipy.interpolate as interp # Or use resampy

class ProcClassifier:
    def __init__(self) -> None:
        self.ref_curve, self.width, self.thresh = None, None, None
        self.is_fitted = False
        
    def _proc_error(self, trace):
        '''
        Returns the procrustes error against the fitted reference curve for a given trace.

                Parameters
                ----------
                trace : {array-like} of shape (n_features,)
                    The training input samples. Internally, it will be converted to
                    ``dtype=np.float32``.

                Returns
                -------
                error : float 
                    Procrustes error for the trace against the reference curve.
        '''
        # if not self.is_fitted:
        #     raise Exception("Error: Fit model before making predictions")
    
        _, _, disparity = procrustes(self.ref_curve.reshape(-1, 1), trace.reshape(-1, 1))
        return disparity
    
    def _normalize_trace(self, trace):
        '''
        Normalize a trace with respect to itself => (trace - mean(trace)) / std(trace)

                Parameters
                ----------
                trace : {array-like} of shape (n_features,)
                    The training input samples. Internally, it will be converted to
                    ``dtype=np.float32``.

                Returns
                -------
                trace : array-like of shape (n_sampn_featuresles,)
                    Normalized trace.
        '''
        mean = np.mean(trace)
        std = np.std(trace)
        return (trace - mean) / std

    def fit(self, X, y=None, threshold=None, normalize=True, ci_width=3.0):
        '''
        Build a Procrustes Analysis powered classifier from the training set (X, y).

                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    The training input samples. Internally, it will be converted to
                    ``dtype=np.float32``.

                y : array-like of shape (n_samples,), default=None
                    The target values (binary class labels) as integers with 1 representing
                    an anomaly. If None, the classifier is fitted on the entire dataset. Else,
                    classifier is fitted only on "good" samples in an attempt to get higher
                    errors with "bad" samples against the reference curve.

                threshold : float, default=None [expected to be between 0 and 1]
                    Provide a percentile value to determine the threshold value on the
                    procrustes error distribution.

                normalize : bool, default=True
                    Option to normalize traces with respect to themselves. Useful when traces
                    have highly variables ranges. For example, stock prices.

                ci_width : float, default=3.0
                    Confidence interval width parameter. Defaults to 3.0 which corresponds to
                    a 99% confidence interval.

                Returns
                -------
                None
        '''
        
        # Convert to 2d numpy array
        X_train = np.array(X, dtype=np.float64, copy=True)
        if X_train.ndim == 1:
            X_train = X.reshape(1, -1)
            
        # Normalize each trace with respect to itself
        if normalize:
            X_train = np.array([self._normalize_trace(x) for x in X_train])
        
        # Only train on good data points if labels are provided
        if y is not None:
            X_train = X_train[y == 0]
        
        # Setup procrustes parameters
        self.width = ci_width
        self.ref_curve = np.mean(X_train, axis=0)
        
        # Error threshold for classification
        errors = np.array([self._proc_error(x) for x in X])
        if threshold is not None:
            if threshold > 0 and threshold < 1:
                self.thresh = np.percentile(errors, threshold)
            else:
                print("Error: Threshold value should be a percentile values expressed as a number between 0 and 1, using default value")
        elif self.thresh is None and y is not None:
            # Optimal cut off would be where tpr is high and fpr is low => tpr - (1-fpr) ~ 0
            fpr, tpr, thresholds = metrics.roc_curve(y, errors, pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            self.thresh = thresholds[optimal_idx]
        elif self.thresh is None:
            self.thresh = np.percentile(errors, 0.75)

        # Mark estimator as fitted 
        self.is_fitted = True

    def predict(self, X):
        '''
        Classify traces based on the trained estimator (X).

                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    The training input samples. Internally, it will be converted to
                    ``dtype=np.float32``.

                Returns
                -------
                y : array-like of shape (n_samples,)
                    The predicted label values (binary class labels) as integers with 1 
                    representing an anomaly.
        '''
        if not self.is_fitted:
            raise Exception("Error: Fit model before making predictions")
        
        # Convert to 2d numpy array
        X = np.array(X, dtype=np.float64, copy=True)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Compute procrustes error for each curve
        errors = np.array([self._proc_error(x) for x in X])
        preds = np.where(errors > self.thresh, 1.0, 0.0)
        
        return preds
