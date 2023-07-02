import numpy as np
from sklearn import metrics
import scipy.interpolate as interp # Or use resampy
from scipy.spatial import procrustes

class ProcClassifier:
    def __init__(self) -> None:
        self.mean, self.width, self.thresh = None, None, None
        
    def _proc_error(self, ref_curve, trace):
        _, _, disparity = procrustes(ref_curve.reshape(1, -1), trace.reshape(1, -1))
        return disparity

    def fit(self, X, y=None, ci_width=3.0, threshold=None):
        '''
        Expects y to be a binary value array => y[i] == 1 or y[i] == 0
        (Assumes 1 is the label for anomalies)
        '''
        
        # Convert to 2d numpy array
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Only train on good data points if labels are provided
        X_train = X.copy() 
        if y is not None:
            X_train = X_train[y == 0]
        
        # Setup procrustes parameters
        self.width = ci_width
        self.mean = np.mean(X, axis=0)
        
        # Error threshold for classification
        errors = np.array([self._proc_error(self.mean, x) for x in X])
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
        errors = np.array([self._proc_error(self.mean, x) for x in X])
        preds = np.where(errors >= self.thresh, 1.0, 0.0)
        
        return preds
