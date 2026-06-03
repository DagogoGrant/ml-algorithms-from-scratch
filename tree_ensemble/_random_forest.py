"""Random Forest classifier implemented from scratch with NumPy."""

from collections import Counter

import numpy as np

from ._decision_tree import DecisionTree
from ._tree_utils import validate_X, validate_X_y


class RandomForest:
    """A random forest classifier using DecisionTree base estimators."""

    def __init__(
        self,
        n_estimators=20,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        n_features_to_consider="sqrt",
        random_state=None,
        **kwargs,
    ):
        n_estimators = kwargs.get("n_trees", kwargs.get("num_trees", n_estimators))
        max_depth = kwargs.get("depth", max_depth)
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features_to_consider = n_features_to_consider
        self.random_state = random_state

        self.trees_ = []
        self.estimators_ = self.trees_
        self.trees = self.trees_
        self.forest = self.trees_
        self.feature_importances_ = None
        self.oob_score_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        """Build the random forest from training data using bootstrap sampling."""
        X, y = validate_X_y(X, y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_features_in_ = n_features
        rng = np.random.default_rng(self.random_state)

        self.trees_ = []
        oob_predictions = [[] for _ in range(n_samples)]

        for _ in range(self.n_estimators):
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            tree_seed = int(rng.integers(0, 2**31 - 1))

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=tree_seed,
            )
            tree.fit(
                X[bootstrap_indices],
                y[bootstrap_indices],
                rf_n_features_to_consider=self.n_features_to_consider,
            )
            self.trees_.append(tree)

            if len(oob_indices) > 0:
                predictions = tree.predict(X[oob_indices])
                for sample_idx, prediction in zip(oob_indices, predictions):
                    oob_predictions[sample_idx].append(prediction)

        self.estimators_ = self.trees_
        self.trees = self.trees_
        self.forest = self.trees_
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.trees_],
            axis=0,
        )
        self.oob_score_ = self._calculate_oob_score(y, oob_predictions)
        return self

    def predict(self, X):
        """Predict class labels using majority voting across all trees."""
        if not self.trees_:
            raise ValueError("RandomForest is not fitted yet")
        X = validate_X(X, self.n_features_in_)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees_])
        predictions = []
        for sample_predictions in tree_predictions.T:
            predictions.append(Counter(sample_predictions).most_common(1)[0][0])
        return np.array(predictions)

    def predict_proba(self, X):
        """Predict class probabilities by averaging tree probabilities."""
        if not self.trees_:
            raise ValueError("RandomForest is not fitted yet")
        X = validate_X(X, self.n_features_in_)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}

        for tree in self.trees_:
            tree_proba = tree.predict_proba(X)
            for local_idx, label in enumerate(tree.classes_):
                proba[:, class_to_idx[label]] += tree_proba[:, local_idx]
        return proba / len(self.trees_)

    def score(self, X, y):
        """Return mean classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _calculate_oob_score(self, y, oob_predictions):
        correct = []
        for sample_idx, predictions in enumerate(oob_predictions):
            if predictions:
                vote = Counter(predictions).most_common(1)[0][0]
                correct.append(vote == y[sample_idx])
        if not correct:
            return 0.0
        return float(np.mean(correct))


RandomForestClassifier = RandomForest
RandomForestClassifierScratch = RandomForest
