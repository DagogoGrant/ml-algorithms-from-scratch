# GenAI Usage Declaration
# This implementation was developed primarily by myself.
# I used support tools for:
#   - clarifying theoretical concepts (e.g., Gaussian/Multinomial Naive Bayes),
#   - discussing implementation approaches,
#   - debugging and improving code quality.
# The final solution reflects my own work and understanding.

"""
Naive Bayes classifiers implementation: Gaussian and Multinomial.
"""

# pylint: disable=invalid-name

import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes (GaussianNB) classifier.

    Assumes continuous features follow a normal distribution.
    """
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None    # unique class labels (set after fit)
        self.priors_ = None     # prior probabilities per class
        self.mean_ = None       # mean of each feature per class
        self.variance_ = None   # variance of each feature per class
        self.class_count_ = None # number of training samples per class
        self.n_features_in_ = None # number of features seen during fit

    def fit(self, X, y):
        """
        Train the model by computing class priors, means, and variances.

        Args:
            X: numpy array of shape (n_samples, n_features) - continuous features
            y: numpy array of shape (n_samples,) - class labels
        """
        # Convert sparse or alternative array-like inputs to dense NumPy array
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "todense"):
            X = np.asarray(X.todense())

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.shape[0] == 0:
            raise ValueError("Empty training data X")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.priors_ = np.zeros(n_classes, dtype=np.float64)
        self.mean_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.n_features_in_ = n_features

        # To prevent division by zero or underflow in variance,
        # we add var_smoothing * max variance of all features in X.
        global_variance = np.var(X, axis=0)
        epsilon = self.var_smoothing * np.max(global_variance)
        if epsilon == 0.0:
            # Fallback if global variance is zero
            epsilon = self.var_smoothing

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = len(X_c) / n_samples
            self.mean_[i, :] = np.mean(X_c, axis=0)
            self.variance_[i, :] = np.var(X_c, axis=0) + epsilon
            self.class_count_[i] = len(X_c)

        return self

    def _joint_log_likelihood(self, X):
        """
        Calculate the joint log likelihood for each class.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "todense"):
            X = np.asarray(X.todense())

        X = np.asarray(X, dtype=np.float64)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        joint_log_lik = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_classes):
            prior = self.priors_[i]
            mean = self.mean_[i]
            variance = self.variance_[i]

            # log(P(x_j | c)) = -0.5 * log(2 * pi * var) - 0.5 * (x - mean)^2 / var
            # Sum over features:
            log_prior = np.log(prior)
            log_pdf = (
                -0.5 * np.sum(np.log(2.0 * np.pi * variance))
                - 0.5 * np.sum(((X - mean) ** 2) / variance, axis=1)
            )
            joint_log_lik[:, i] = log_prior + log_pdf

        return joint_log_lik

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        joint_log_lik = self._joint_log_likelihood(X)
        best_class_idx = np.argmax(joint_log_lik, axis=1)
        return self.classes_[best_class_idx]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        joint_log_lik = self._joint_log_likelihood(X)
        # Log-sum-exp trick for numerical stability
        max_log = np.max(joint_log_lik, axis=1, keepdims=True)
        exp_likelihood = np.exp(joint_log_lik - max_log)
        return exp_likelihood / np.sum(exp_likelihood, axis=1, keepdims=True)

    def predict_log_proba(self, X):
        """
        Predict class log probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        joint_log_lik = self._joint_log_likelihood(X)
        max_log = np.max(joint_log_lik, axis=1, keepdims=True)
        log_sum = max_log + np.log(np.sum(np.exp(joint_log_lik - max_log), axis=1, keepdims=True))
        return joint_log_lik - log_sum

    def predict_log_prob(self, X):
        """Alias for predict_log_proba."""
        return self.predict_log_proba(X)

    def predict_probability(self, X):
        """Alias for predict_proba."""
        return self.predict_proba(X)

    @property
    def class_prior_(self):
        """Prior probabilities of each class."""
        return self.priors_

    @property
    def theta_(self):
        """Mean of each feature per class."""
        return self.mean_

    @property
    def var_(self):
        """Variance of each feature per class."""
        return self.variance_

    @property
    def sigma_(self):
        """Variance of each feature per class (alias for var_)."""
        return self.variance_


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes (MultinomialNB) classifier.

    Designed for discrete count/frequency features.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha                  # Laplace smoothing parameter
        self.classes_ = None                # unique class labels
        self.class_log_prior_ = None        # empirical log prior per class
        self.feature_log_prob_ = None       # log probability of features given class
        self.class_count_ = None            # number of training samples per class
        self.feature_count_ = None          # number of feature counts observed
        self.n_features_in_ = None          # number of features seen during fit

    def fit(self, X, y):
        """
        Train the model by computing class priors and feature likelihoods.

        Args:
            X: numpy array of shape (n_samples, n_features) - non-negative count features
            y: numpy array of shape (n_samples,) - class labels
        """
        # Convert sparse or alternative array-like inputs to dense NumPy array
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "todense"):
            X = np.asarray(X.todense())

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.shape[0] == 0:
            raise ValueError("Empty training data X")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative count features")

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.n_features_in_ = n_features

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Prior probability in log space
            self.class_log_prior_[i] = np.log(len(X_c) / n_samples)
            self.class_count_[i] = len(X_c)

            # Feature counts for class c
            feature_counts = np.sum(X_c, axis=0)
            self.feature_count_[i, :] = feature_counts
            total_count = np.sum(feature_counts)

            # Laplace smoothing: (count + alpha) / (total_count + alpha * n_features)
            self.feature_log_prob_[i, :] = (
                np.log(feature_counts + self.alpha)
                - np.log(total_count + self.alpha * n_features)
            )

        return self

    def _joint_log_likelihood(self, X):
        """
        Calculate joint log likelihood of features and classes.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "todense"):
            X = np.asarray(X.todense())

        X = np.asarray(X, dtype=np.float64)

        # log likelihood = X @ feature_log_prob_.T + class_log_prior_
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        joint_log_lik = self._joint_log_likelihood(X)
        best_class_idx = np.argmax(joint_log_lik, axis=1)
        return self.classes_[best_class_idx]

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        joint_log_lik = self._joint_log_likelihood(X)
        # Log-sum-exp trick
        max_log = np.max(joint_log_lik, axis=1, keepdims=True)
        exp_likelihood = np.exp(joint_log_lik - max_log)
        return exp_likelihood / np.sum(exp_likelihood, axis=1, keepdims=True)

    def predict_log_proba(self, X):
        """
        Predict class log probabilities.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes)
        """
        joint_log_lik = self._joint_log_likelihood(X)
        max_log = np.max(joint_log_lik, axis=1, keepdims=True)
        log_sum = max_log + np.log(np.sum(np.exp(joint_log_lik - max_log), axis=1, keepdims=True))
        return joint_log_lik - log_sum

    def predict_log_prob(self, X):
        """Alias for predict_log_proba."""
        return self.predict_log_proba(X)

    def predict_probability(self, X):
        """Alias for predict_proba."""
        return self.predict_proba(X)

    @property
    def class_prior_(self):
        """Empirical prior class probabilities."""
        if self.class_log_prior_ is None:
            return None
        return np.exp(self.class_log_prior_)

    @property
    def coef_(self):
        """Expose feature log probabilities as coefficients for linear model interpretation."""
        if self.feature_log_prob_ is None:
            return None
        n_classes = len(self.classes_)
        if n_classes == 2:
            return self.feature_log_prob_[1:]
        return self.feature_log_prob_

    @property
    def intercept_(self):
        """Expose class log priors as intercepts for linear model interpretation."""
        if self.class_log_prior_ is None:
            return None
        n_classes = len(self.classes_)
        if n_classes == 2:
            return self.class_log_prior_[1:]
        return self.class_log_prior_



# Aliases
GaussianNB = GaussianNaiveBayes
MultinomialNB = MultinomialNaiveBayes
