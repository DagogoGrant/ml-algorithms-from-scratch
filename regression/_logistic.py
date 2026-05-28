# GenAI Usage Declaration
# This implementation was developed primarily by myself.
# I used support tools for:
#   - clarifying theoretical concepts (e.g., Logistic Regression and SGD methods),
#   - discussing implementation approaches,
#   - debugging and improving code quality.
# All code was written, reviewed, and adapted by me. I understand the full
# implementation and can explain all design decisions and mathematical steps.
# The final solution reflects my own work and understanding.

import numpy as np

# Stability Constants
SIGMOID_LIMIT = 20.0
EPS = 1e-15


def _sigmoid(z):
    """
    Compute the sigmoid activation function with numerical stability.

    Parameters
    ----------
    z : ndarray
        Linear boundary scores.

    Returns
    -------
    prob : ndarray
        Class probabilities.
    """
    z_clipped = np.clip(z, -SIGMOID_LIMIT, SIGMOID_LIMIT)
    return np.where(
        z_clipped >= 0,
        1.0 / (1.0 + np.exp(-z_clipped)),
        np.exp(z_clipped) / (1.0 + np.exp(z_clipped))
    )


class LogisticRegression:
    """
    Logistic Regression with full-batch gradient descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 l2_penalty=0.0001, momentum=0.95,
                 tol=1e-8, random_state=None,
                 decay=1e-6, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_penalty = l2_penalty
        self.momentum = momentum
        self.tol = tol
        self.random_state = random_state
        self.decay = decay
        self.fit_intercept = fit_intercept
        self.weights_ = None
        self.bias_ = 0.0
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = 0
        self.n_iter = 0
        self.loss_history = []

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model using full-batch gradient descent.
        """
        if len(X) == 0:
            raise ValueError("Empty training data")
            
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).flatten()
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        mean, std = np.mean(X_arr, axis=0), np.std(X_arr, axis=0)
        std = np.where(std < EPS, 1.0, std)
        X_scaled = (X_arr - mean) / std

        weights = np.zeros(n_features, dtype=np.float64)
        bias = 0.0

        sw = np.asarray(sample_weight).flatten() if sample_weight is not None else np.ones(n_samples, dtype=np.float64)
        counts = np.bincount(y_arr.astype(int))
        cw = [n_samples / (2.0 * counts[0]), n_samples / (2.0 * counts[1])] if len(counts) > 1 else [1.0, 1.0]
        sw *= np.where(y_arr == 1, cw[1], cw[0])
        sw *= (n_samples / np.sum(sw))

        v_w, v_b, p_w = np.zeros(n_features, dtype=np.float64), 0.0, weights.copy()

        for i in range(1, self.n_iterations + 1):
            self.n_iter = i
            z = X_scaled @ weights + bias
            error = sw * (_sigmoid(z) - y_arr)

            dw = (X_scaled.T @ error + self.l2_penalty * weights) / n_samples
            db = np.sum(error) / n_samples if self.fit_intercept else 0.0

            current_lr = self.learning_rate / (1.0 + self.decay * i)
            v_w = self.momentum * v_w - current_lr * dw
            v_b = self.momentum * v_b - current_lr * db
            weights += v_w
            bias += v_b

            if np.linalg.norm(weights - p_w) < self.tol:
                break
            p_w = weights.copy()

        z_f = X_scaled @ weights + bias
        loss = np.mean(sw * (np.maximum(z_f, 0) - z_f * y_arr + np.log1p(np.exp(-np.abs(z_f)))))
        self.loss_history.append(float(loss + (self.l2_penalty / (2.0 * n_samples)) * np.sum(weights ** 2)))

        self.weights_ = weights / std
        self.bias_ = float(bias - np.sum((weights * mean) / std))
        self.coef_ = self.weights_.reshape(1, -1)
        self.intercept_ = np.array([self.bias_])
        return self

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        """
        return np.asarray(X, dtype=np.float64) @ self.weights_ + self.bias_

    def predict_proba(self, X):
        """
        Probability estimates.
        """
        p1 = _sigmoid(self.decision_function(X)).flatten()
        return np.column_stack((1.0 - p1, p1)).astype(np.float64)

    def predict(self, X):
        """
        Predict binary labels.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SGDClassifier:
    """
    Logistic Regression using mini-batch SGD.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 batch_size=32, random_state=None,
                 l2_penalty=0.0001, momentum=0.95,
                 tol=1e-8, decay=1e-6,
                 class_weight='balanced', fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.l2_penalty = l2_penalty
        self.momentum = momentum
        self.tol = tol
        self.decay = decay
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.weights_ = None
        self.bias_ = 0.0
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = 0
        self.n_iter = 0
        self.loss_history = []

    def fit(self, X, y, sample_weight=None):
        """
        Fit using mini-batch SGD.
        """
        if len(X) == 0:
            raise ValueError("Empty training data")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).flatten()
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        mean, std = np.mean(X_arr, axis=0), np.std(X_arr, axis=0)
        std = np.where(std < EPS, 1.0, std)
        X_scaled = (X_arr - mean) / std

        weights = np.zeros(n_features, dtype=np.float64)
        bias = 0.0

        sw = np.asarray(sample_weight).flatten() if sample_weight is not None else np.ones(n_samples, dtype=np.float64)
        if self.class_weight == 'balanced':
            counts = np.bincount(y_arr.astype(int))
            cw = [n_samples / (2.0 * counts[0]), n_samples / (2.0 * counts[1])] if len(counts) > 1 else [1.0, 1.0]
            sw *= np.where(y_arr == 1, cw[1], cw[0])
        sw *= (n_samples / np.sum(sw))

        rng = np.random.default_rng(self.random_state)
        v_w, v_b, p_w = np.zeros(n_features, dtype=np.float64), 0.0, weights.copy()

        for i in range(1, self.n_iterations + 1):
            self.n_iter = i
            idx = rng.permutation(n_samples)
            current_lr = self.learning_rate / (1.0 + self.decay * i)
            for j in range(0, n_samples, self.batch_size):
                bi = idx[j:j + self.batch_size]
                bm = len(bi)
                Xb = X_scaled[bi]
                z = Xb @ weights + bias
                error = sw[bi] * (_sigmoid(z) - y_arr[bi])

                dw = (Xb.T @ error + self.l2_penalty * weights) / bm
                db = np.sum(error) / bm if self.fit_intercept else 0.0

                v_w = self.momentum * v_w - current_lr * dw
                v_b = self.momentum * v_b - current_lr * db
                weights += v_w
                bias += v_b

            if np.linalg.norm(weights - p_w) < self.tol:
                break
            p_w = weights.copy()

        z_f = X_scaled @ weights + bias
        loss = np.mean(sw * (np.maximum(z_f, 0) - z_f * y_arr + np.log1p(np.exp(-np.abs(z_f)))))
        self.loss_history.append(float(loss + (self.l2_penalty / (2.0 * n_samples)) * np.sum(weights ** 2)))

        self.weights_ = weights / std
        self.bias_ = float(bias - np.sum((weights * mean) / std))
        self.coef_, self.intercept_ = self.weights_.reshape(1, -1), np.array([self.bias_])
        return self

    def decision_function(self, X):
        """
        Confidence scores.
        """
        return np.asarray(X, dtype=np.float64) @ self.weights_ + self.bias_

    def predict_proba(self, X):
        """
        Probabilities.
        """
        p1 = _sigmoid(self.decision_function(X)).flatten()
        return np.column_stack((1.0 - p1, p1)).astype(np.float64)

    def predict(self, X):
        """
        Labels.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
