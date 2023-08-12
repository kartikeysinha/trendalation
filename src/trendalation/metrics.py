import numpy as np
from scipy.spatial import procrustes

def proc_error(ref_curve, trace):
    '''
    Returns the procrustes error for the given curve against the specified reference curve.

            Parameters
            ----------
            ref_curve : {array-like} of shape (n_features,)
                The training input samples. Internally, it will be converted to
                ``dtype=np.float32``.

            trace : {array-like} of shape (n_features,)
                The training input samples. Internally, it will be converted to
                ``dtype=np.float32``.

            Returns
            -------
            error : float 
                Procrustes error for the trace against the reference curve.
    '''
    ref_curve, trace = [np.array(x) for x in (ref_curve, trace)]
    _, _, disparity = procrustes(ref_curve.reshape(-1, 1), trace.reshape(-1, 1))
    return disparity

def normalize_trace(trace):
    '''
    Normalize a trace with respect to itself => (trace - mean(trace)) / std(trace)

            Parameters
            ----------
            trace : {array-like} of shape (n_features,)
                The training input samples. Internally, it will be converted to
                ``dtype=np.float32``.

            Returns
            -------
            trace : array-like of shape (n_samples, n_features)
                Normalized trace.
    '''
    mean = np.mean(trace)
    std = np.std(trace)
    return (trace - mean) / std
