# GenAI Usage Declaration
# This implementation was developed primarily by myself.
# I used support tools for:
#   - clarifying theoretical concepts (e.g., KMeans clustering and KMeans++ initialization),
#   - discussing implementation approaches,
#   - debugging and improving code quality.
# All code was written, reviewed, and adapted by me. I understand the full
# implementation and can explain all design decisions and mathematical steps.
# The final solution reflects my own work and understanding.

import numpy as np


def _compute_distances(X, centers):
    """Vectorized squared Euclidean distances using ||x-c||^2 = ||x||^2 - 2xc + ||c||^2."""
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    C_sq = np.sum(centers**2, axis=1, keepdims=True).T
    dist = X_sq - 2.0 * np.dot(X, centers.T) + C_sq
    return np.maximum(dist, 0.0)


def _k_means_plusplus(X, n_clusters, rng):
    """KMeans++ initialization with explicit distance caching."""
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features))
    centers[0] = X[rng.integers(n_samples)]
    for k in range(1, n_clusters):
        dists = _compute_distances(X, centers[:k])
        min_dists = np.min(dists, axis=1) + 1e-15
        probs = min_dists / np.sum(min_dists)
        centers[k] = X[rng.choice(n_samples, p=probs)]
    return centers


def _kmeans_single_run(X, n_clusters, rng, init_func, config):
    """Perform a single run with early stopping and history tracking."""
    centers = init_func(X, n_clusters, rng)
    labels = np.full(X.shape[0], -1)
    prev_inertia = np.inf

    for _ in range(config[0]): # max_iter
        dist_matrix = _compute_distances(X, centers)
        new_labels = np.argmin(dist_matrix, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = np.mean(X[mask], axis=0)
            else:
                new_centers[k] = X[rng.integers(X.shape[0])]

        inertia = np.sum(np.min(dist_matrix, axis=1))
        # Early stopping: inertia delta and norm
        if np.abs(prev_inertia - inertia) < config[1]: # tol
            break
        if np.linalg.norm(new_centers - centers) < config[1]: # tol
            break
        prev_inertia, centers = inertia, new_centers
    return centers, labels, prev_inertia


class KMeans:
    """K-Means clustering implementation."""
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        if n_clusters <= 0:
            raise ValueError("n_clusters > 0")
        if max_iter <= 0:
            raise ValueError("max_iter > 0")
        if tol < 0:
            raise ValueError("tol >= 0")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """Fit with data validation and multi-restart."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.size == 0:
            raise ValueError("Empty X")
        if np.any(np.isnan(X_arr)):
            raise ValueError("NaN X")
        if X_arr.shape[0] < self.n_clusters:
            raise ValueError("N < K")

        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        config = (self.max_iter, self.tol)
        for _ in range(15):
            c, l, h = _kmeans_single_run(X_arr, self.n_clusters, rng,
                                        lambda x, k, r: x[r.choice(x.shape[0], k, False)],
                                        config)
            if h < best_inertia:
                best_inertia, self.cluster_centers_, self.labels_, self.inertia_ = h, c, l, h
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.argmin(_compute_distances(X_arr, self.cluster_centers_), axis=1)

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.sqrt(_compute_distances(X_arr, self.cluster_centers_))

    def score(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return -np.sum(np.min(_compute_distances(X_arr, self.cluster_centers_), axis=1))

    def fit_predict(self, X):
        return self.fit(X).labels_


class KMeansPlusPlus:
    """K-Means++ clustering implementation."""
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        if n_clusters <= 0:
            raise ValueError("n_clusters > 0")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.size == 0 or X_arr.shape[0] < self.n_clusters:
            raise ValueError("Size")
        if np.any(np.isnan(X_arr)):
            raise ValueError("NaN")

        rng = np.random.default_rng(self.random_state)
        c, l, h = _kmeans_single_run(X_arr, self.n_clusters, rng, _k_means_plusplus,
                                    (self.max_iter, self.tol))
        self.cluster_centers_, self.labels_, self.inertia_ = c, l, h
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.argmin(_compute_distances(X_arr, self.cluster_centers_), axis=1)

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.sqrt(_compute_distances(X_arr, self.cluster_centers_))

    def score(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return -np.sum(np.min(_compute_distances(X_arr, self.cluster_centers_), axis=1))

    def fit_predict(self, X):
        return self.fit(X).labels_
