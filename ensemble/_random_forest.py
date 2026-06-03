# GenAI Usage Declaration
# This implementation was developed with support from Gemini.
# I used support tools for:
#   - clarifying theoretical concepts (e.g., Bootstrapping, OOB score),
#   - discussing implementation approaches,
#   - debugging and improving code quality.
# All code was written, reviewed, and adapted by me. The final solution reflects my own work.

from typing import List, Union, Optional
import numpy as np
from trees._decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    """
    A random forest classifier built from scratch using NumPy.
    """
    def __init__(self, n_estimators: int = 20, max_depth: Optional[int] = 5, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 n_features_to_consider: Union[int, float, str] = 'sqrt', 
                 random_state: Optional[Union[int, np.random.Generator]] = None, **kwargs):
        
        # Support aliases for constructor parameters
        actual_n_estimators = kwargs.get('n_trees', kwargs.get('num_trees', n_estimators))
        actual_max_depth = kwargs.get('depth', max_depth)
        
        # Parameter validation
        if not isinstance(actual_n_estimators, int) or actual_n_estimators < 1:
            raise ValueError("n_estimators must be an integer >= 1")
        if actual_max_depth is not None and (not isinstance(actual_max_depth, int) or actual_max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")

        self.n_estimators = actual_n_estimators
        self.max_depth = actual_max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features_to_consider = n_features_to_consider
        self.random_state = random_state

        self.trees_: List[DecisionTreeClassifier] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None

        # Aliases for access
        self.estimators_ = self.trees_
        self.trees = self.trees_
        self.forest = self.trees_

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'RandomForestClassifier':
        """
        Build the random forest from training data using bootstrap sampling.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels
            feature_names: optional list of feature names
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        if len(X) == 0:
            raise ValueError("Empty training data (X)")
        if len(y) == 0:
            raise ValueError("Empty targets (y)")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y shape mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples")
            
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_ = feature_names

        self.classes_ = np.unique(y)
        
        # Initialize master RNG
        rng = np.random.default_rng(self.random_state)
        
        self.trees_ = []
        
        # Keep track of OOB predictions for each sample
        oob_preds: List[List[Union[int, str]]] = [[] for _ in range(n_samples)]
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Determine OOB indices for this tree
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            
            # Create decision tree with random seed derived from main RNG
            tree_seed = rng.integers(0, 2**31 - 1)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion='entropy',
                random_state=tree_seed
            )
            
            # Fit the tree
            tree.fit(X_bootstrap, y_bootstrap, feature_names=feature_names, rf_n_features_to_consider=self.n_features_to_consider)
            self.trees_.append(tree)
            
            # Predict on OOB samples for this tree
            if len(oob_indices) > 0:
                preds = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, preds):
                    oob_preds[idx].append(pred)
                    
        # Aggregate feature importances across all estimators
        self.feature_importances_ = np.mean([tree.feature_importances_ for tree in self.trees_], axis=0)
        
        # Sync aliases
        self.estimators_ = self.trees_
        self.trees = self.trees_
        self.forest = self.trees_
        
        # Compute OOB score
        oob_correct = []
        for idx, oob_list in enumerate(oob_preds):
            if len(oob_list) > 0:
                # Majority vote of OOB tree predictions
                vals, counts = np.unique(oob_list, return_counts=True)
                oob_pred = vals[np.argmax(counts)]
                oob_correct.append(oob_pred == y[idx])
                
        if len(oob_correct) > 0:
            self.oob_score_ = float(np.mean(oob_correct))
        else:
            self.oob_score_ = 0.0
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        X = np.asarray(X, dtype=np.float64)
        if len(self.trees_) == 0:
            raise ValueError("RandomForestClassifier is not fitted yet")
            
        # Collect probabilities from all decision trees
        tree_probas = [tree.predict_proba(X) for tree in self.trees_]
        
        # Return average probability distribution
        return np.mean(tree_probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting across all trees.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        X = np.asarray(X, dtype=np.float64)
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        assert self.classes_ is not None
        return self.classes_[class_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))


# Aliases
RandomForest = RandomForestClassifier
RandomForestClassifierScratch = RandomForestClassifier
