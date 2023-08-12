import numpy as np
import pytest

# Import classification module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from classification import ProcClassifier

# To run these tests- python3 -m pytest src/trendalation/tests/classifier_unit_tests.py

@pytest.fixture
def mock_data():
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_fit_and_predict(mock_data):
    X, y = mock_data
    classifier = ProcClassifier()
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)

def test_predict_score(mock_data):
    X, y = mock_data
    classifier = ProcClassifier()
    classifier.fit(X, y)
    scores = classifier.predict_score(X)
    assert len(scores) == len(X)

def test_predict_preproc(mock_data):
    X, y = mock_data
    classifier = ProcClassifier()
    classifier.fit(X, y)
    preproc_result = classifier._predict_preproc(X)
    assert len(preproc_result) == len(X)

def test_fit_without_labels(mock_data):
    X, _ = mock_data
    classifier = ProcClassifier()
    classifier.fit(X)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)

def test_fit_without_labels_threshold(mock_data):
    X, _ = mock_data
    classifier = ProcClassifier()
    classifier.fit(X, threshold=0.75)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)

def test_fit_with_invalid_threshold(mock_data):
    X, y = mock_data
    classifier = ProcClassifier()
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)

def test_fit_and_predict_normalize_false(mock_data):
    X, y = mock_data
    classifier = ProcClassifier()
    classifier.fit(X, y, normalize=False)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)

def test_predict_before_fit(mock_data):
    X, _ = mock_data
    classifier = ProcClassifier()
    with pytest.raises(Exception):
        classifier.predict(X)
