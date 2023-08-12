import numpy as np
from scipy.spatial import procrustes

# Import metrics module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from metrics import proc_error, normalize_trace

# To run these tests- python3 -m pytest src/trendalation/tests/metrics_unit_tests.py

def test_proc_error():
    ref_curve = np.array([1, 2, 3, 4, 5])
    trace = np.array([2, 3, 4, 5, 6])
    result = proc_error(ref_curve, trace)
    expected = procrustes(ref_curve.reshape(-1, 1), trace.reshape(-1, 1))[2]
    assert np.allclose(result, expected)

def test_normalize_trace():
    trace = np.array([1, 2, 3, 4, 5])
    result = normalize_trace(trace)
    expected = (trace - np.mean(trace)) / np.std(trace)
    assert np.allclose(result, expected)
