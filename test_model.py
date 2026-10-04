"""
Unit tests for the A3 classification model

Run from the repo root with:   pytest
GitHub Actions runs this file on every push (see .github/workflows/ci-cd.yml).
"""
import os
import sys

import joblib
import numpy as np
import pytest

# make `logistic_regression` importable (joblib needs it to unpickle the model)
CODE_DIR = os.path.join(os.path.dirname(__file__), "app", "code")
sys.path.insert(0, CODE_DIR)

MODEL_PATH = os.path.join(CODE_DIR, "models", "model_v3.pkl")
N_CLASSES = 4          # price buckets 0, 1, 2, 3


@pytest.fixture(scope="module")
def model():
    """Load the exact saved model the web app uses (also checks the pickle loads)."""
    return joblib.load(MODEL_PATH)


def make_input(model, rows=5):
    """Fake already-preprocessed rows: intercept column of ones + random features."""
    n_features = model.W.shape[0]            # includes the intercept column
    X = np.random.default_rng(0).normal(size=(rows, n_features))
    X[:, 0] = 1.0
    return X


def test_model_takes_expected_input(model):
    """(1) The model accepts input with the expected number of columns,
    and rejects input with the wrong number of columns."""
    X = make_input(model)
    assert X.shape[1] == model.W.shape[0]    # 43 = intercept + 42 features
    model.predict(X)                          # must not raise

    X_wrong = X[:, :-1]                       # one column short
    with pytest.raises(ValueError):
        model.predict(X_wrong)


def test_model_output_has_expected_shape(model):
    """(2) predict() gives one class label per row, each a valid bucket (0-3);
    the probability matrix is (rows, 4) and every row sums to 1."""
    X = make_input(model, rows=5)

    preds = model.predict(X)
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset(set(range(N_CLASSES)))

    probs = model.h_theta(X, model.W)
    assert probs.shape == (5, N_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0)
