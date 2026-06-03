"""Shared utilities for tree-based classifiers."""

import numpy as np


def validate_X_y(X, y):
    """Return validated feature matrix and target vector."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array")
    if y.ndim != 1:
        y = y.ravel()
    if X.shape[0] == 0:
        raise ValueError("X is empty")
    if y.shape[0] == 0:
        raise ValueError("y is empty")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    return X, y


def validate_X(X, n_features=None):
    """Return a validated feature matrix."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array")
    if X.shape[0] == 0:
        raise ValueError("X is empty")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, expected {n_features}")

    return X


def entropy(y):
    """Calculate label entropy."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def gini(y):
    """Calculate Gini impurity."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities ** 2))


def majority_class(y):
    """Return the most frequent class label."""
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]
