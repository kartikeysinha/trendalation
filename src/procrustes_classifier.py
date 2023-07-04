import numpy as np
from sklearn import metrics
import scipy.interpolate as interp # Or use resampy
from scipy.spatial import procrustes

def proc_error(ref_curve, trace):
    ref_curve, trace = [np.array(x) for x in (ref_curve, trace)]
    _, _, disparity = procrustes(ref_curve.reshape(-1, 1), trace.reshape(-1, 1))
    return disparity

def normalize_trace(trace, eps=1e-7):
    mean = np.mean(trace)
    std = np.std(trace) + eps # Add small value to prevent division by zero error
    return (trace - mean) / std

class ProcClassifier:
    def __init__(self) -> None:
        self.ref_curve, self.width, self.thresh = None, None, None
        
    def _proc_error(self, ref_curve, trace):
        _, _, disparity = procrustes(ref_curve.reshape(-1, 1), trace.reshape(-1, 1))
        return disparity
    
    def _normalize_trace(self, trace):
        eps = 1e-7 # Add small value to prevent division by zero error
        mean = np.mean(trace)
        std = np.std(trace) + eps
        return (trace - mean) / std

    def fit(self, X, y=None, ci_width=3.0, threshold=None, normalize=True):
        '''
        Expects y to be a binary value array => y[i] == 1 or y[i] == 0
        (Assumes 1 is the label for anomalies)
        '''
        
        # Convert to 2d numpy array
        X_train = np.array(X, copy=True)
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
        errors = np.array([self._proc_error(self.ref_curve, x) for x in X])
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
            
    def predict(self, X):
        # Convert to 2d numpy array
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Compute procrustes error for each curve
        errors = np.array([self._proc_error(self.ref_curve, x) for x in X])
        preds = np.where(errors > self.thresh, 1.0, 0.0)
        
        return preds
