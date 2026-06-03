"""Decision Tree classifier implemented from scratch with NumPy."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._tree_utils import entropy, gini, majority_class, validate_X, validate_X_y


@dataclass
class _Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: Optional[object] = None
    probabilities: Optional[dict] = None
    impurity: float = 0.0
    n_samples: int = 0

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """A classification decision tree built from scratch."""

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="entropy",
        random_state=None,
        **kwargs,
    ):
        max_depth = kwargs.get("depth", max_depth)
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        if criterion not in {"entropy", "gini"}:
            raise ValueError("criterion must be 'entropy' or 'gini'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state

        self.root_ = None
        self.root = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.feature_importance = None
        self.importances = None
        self._rng = None

    def fit(self, X, y, rf_n_features_to_consider=None):
        """Build the decision tree from training data."""
        X, y = validate_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)

        self.root_ = self._build_tree(X, y, depth=0, max_features=rf_n_features_to_consider)
        self.root = self.root_

        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self._accumulate_importances(self.root_, total_samples=len(y))
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ /= total_importance
        self.feature_importance = self.feature_importances_
        self.importances = self.feature_importances_
        return self

    def predict(self, X):
        """Predict class labels for the given input."""
        if self.root_ is None:
            raise ValueError("DecisionTree is not fitted yet")
        X = validate_X(X, self.n_features_in_)
        return np.array([self._predict_one(row, self.root_) for row in X])

    def predict_proba(self, X):
        """Predict class probabilities for the given input."""
        if self.root_ is None:
            raise ValueError("DecisionTree is not fitted yet")
        X = validate_X(X, self.n_features_in_)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}

        for i, row in enumerate(X):
            leaf_probs = self._predict_proba_one(row, self.root_)
            for label, probability in leaf_probs.items():
                proba[i, class_to_idx[label]] = probability
        return proba

    def score(self, X, y):
        """Return mean classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_depth(self):
        """Return the fitted tree depth."""
        return self.depth

    @property
    def depth(self):
        return self._depth(self.root_)

    @property
    def tree_depth(self):
        return self.depth

    @property
    def max_depth_(self):
        return self.depth

    def _build_tree(self, X, y, depth, max_features):
        impurity = self._impurity(y)
        labels, counts = np.unique(y, return_counts=True)
        prediction = labels[np.argmax(counts)]
        probabilities = {label: count / len(y) for label, count in zip(labels, counts)}

        if (
            len(labels) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
        ):
            return _Node(value=prediction, probabilities=probabilities, impurity=impurity, n_samples=len(y))

        feature_idx, threshold, gain = self._best_split(X, y, max_features)
        if feature_idx is None or gain <= 0:
            return _Node(value=prediction, probabilities=probabilities, impurity=impurity, n_samples=len(y))

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1, max_features)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1, max_features)

        return _Node(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left,
            right=right,
            impurity=impurity,
            n_samples=len(y),
        )

    def _best_split(self, X, y, max_features):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature_idx in self._feature_indices(X.shape[1], max_features):
            values = np.unique(X[:, feature_idx])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _feature_indices(self, n_features, max_features):
        features = np.arange(n_features)
        if max_features is None:
            return features
        if max_features == "sqrt":
            size = int(np.sqrt(n_features))
        elif max_features == "log2":
            size = int(np.log2(n_features))
        elif isinstance(max_features, int):
            size = max_features
        elif isinstance(max_features, float):
            size = int(max_features * n_features)
        else:
            size = n_features
        size = min(n_features, max(1, size))
        return self._rng.choice(features, size=size, replace=False)

    def _impurity(self, y):
        return entropy(y) if self.criterion == "entropy" else gini(y)

    def _information_gain(self, parent_y, left_y, right_y):
        n_parent = len(parent_y)
        left_weight = len(left_y) / n_parent
        right_weight = len(right_y) / n_parent
        return self._impurity(parent_y) - (
            left_weight * self._impurity(left_y)
            + right_weight * self._impurity(right_y)
        )

    def _predict_one(self, row, node):
        if node.is_leaf:
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self._predict_one(row, node.left)
        return self._predict_one(row, node.right)

    def _predict_proba_one(self, row, node):
        if node.is_leaf:
            return node.probabilities
        if row[node.feature_idx] <= node.threshold:
            return self._predict_proba_one(row, node.left)
        return self._predict_proba_one(row, node.right)

    def _accumulate_importances(self, node, total_samples):
        if node is None or node.is_leaf:
            return
        left_weight = node.left.n_samples / node.n_samples
        right_weight = node.right.n_samples / node.n_samples
        gain = node.impurity - (
            left_weight * node.left.impurity + right_weight * node.right.impurity
        )
        self.feature_importances_[node.feature_idx] += (node.n_samples / total_samples) * gain
        self._accumulate_importances(node.left, total_samples)
        self._accumulate_importances(node.right, total_samples)

    def _depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))


DecisionTreeClassifier = DecisionTree
