# GenAI Usage Declaration
# This implementation was developed with support from Gemini.
# I used support tools for:
#   - clarifying theoretical concepts (e.g., ID3 splitting, Information Gain),
#   - discussing implementation approaches,
#   - debugging and improving code quality.
# All code was written, reviewed, and adapted by me. The final solution reflects my own work.

from typing import List, Dict, Union, Tuple, Optional
import numpy as np

# Helper functions for impurity and gain calculations
def _calculate_entropy(y: np.ndarray) -> float:
    """
    Calculate the entropy of labels y.
    """
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return -np.sum(probs * np.log2(probs))


def _calculate_gini(y: np.ndarray) -> float:
    """
    Calculate the Gini impurity of labels y.
    """
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)


def _calculate_gain(y: np.ndarray, left_y: np.ndarray, right_y: np.ndarray, criterion: str = 'entropy') -> float:
    """
    Calculate the impurity reduction (information gain) of a split.
    """
    impurity_func = _calculate_entropy if criterion == 'entropy' else _calculate_gini
    
    parent_impurity = impurity_func(y)
    n = len(y)
    n_left = len(left_y)
    n_right = len(right_y)
    
    if n_left == 0 or n_right == 0:
        return 0.0
        
    child_impurity = (n_left / n) * impurity_func(left_y) + (n_right / n) * impurity_func(right_y)
    return parent_impurity - child_impurity


class Node:
    """
    Represents a single node in the Decision Tree.
    """
    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None, 
                 left: Optional['Node'] = None, right: Optional['Node'] = None, *, 
                 value: Optional[Union[int, str]] = None, impurity: Optional[float] = None, 
                 n_samples: Optional[int] = None, probabilities: Optional[Dict[Union[int, str], float]] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples
        self.probabilities = probabilities

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """
    A classification decision tree built from scratch using NumPy.
    """
    def __init__(self, max_depth: Optional[int] = 5, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, criterion: str = 'entropy', 
                 random_state: Optional[Union[int, np.random.Generator]] = None, **kwargs):
        # Validate constructor parameters
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        if criterion not in ('entropy', 'gini'):
            raise ValueError("criterion must be 'entropy' or 'gini'")

        # Support both 'max_depth' and 'depth' naming conventions
        self.max_depth = kwargs.get('depth', max_depth)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        
        self.root: Optional[Node] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.feature_importance: Optional[np.ndarray] = None
        self.importances: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self._default_class_for_empty_leaf: Optional[Union[int, str]] = None
        self.rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None, 
            rf_n_features_to_consider: Optional[Union[int, float, str]] = None) -> 'DecisionTreeClassifier':
        """
        Build the decision tree from training data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels
            feature_names: optional list of feature names
            rf_n_features_to_consider: number of features to randomly select at each node (used in Random Forest)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        if len(X) == 0:
            raise ValueError("Empty training data (X)")
        if len(y) == 0:
            raise ValueError("Empty targets (y)")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y shape mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples")
            
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = feature_names
        self.rng = np.random.default_rng(self.random_state)
        
        # Determine classes and default majority class
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self._default_class_for_empty_leaf = self.classes_[np.argmax(class_counts)] if len(class_counts) > 0 else None
        
        # Build tree recursively
        self.root = self._build_tree(X, y, depth=0, rf_n_features_to_consider=rf_n_features_to_consider)
        
        # Calculate feature importances
        self.feature_importances_ = np.zeros(self.n_features_in_)
        if self.root is not None:
            self._calculate_feature_importances(self.root, len(y))
            imp_sum: float = float(np.sum(self.feature_importances_))
            if imp_sum > 0:
                self.feature_importances_ /= imp_sum
                
        # Set aliases for access
        self.feature_importance = self.feature_importances_
        self.importances = self.feature_importances_
        
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0, 
                    rf_n_features_to_consider: Optional[Union[int, float, str]] = None) -> Node:
        n_samples = X.shape[0]
        
        # Node impurity
        node_impurity = _calculate_entropy(y) if self.criterion == 'entropy' else _calculate_gini(y)
        
        # Class probabilities and majority vote at this node
        classes, counts = np.unique(y, return_counts=True)
        if len(counts) > 0:
            majority_val = classes[np.argmax(counts)]
            probs = {c: count / n_samples for c, count in zip(classes, counts)}
        else:
            majority_val = self._default_class_for_empty_leaf
            probs = {}

        # Base cases:
        # 1. Pure node
        # 2. Max depth reached
        # 3. Too few samples to split
        # 4. Too few samples to divide into children of minimum leaf size
        if (len(classes) <= 1 or 
            (self.max_depth is not None and depth >= self.max_depth) or 
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf):
            return Node(value=majority_val, impurity=node_impurity, n_samples=n_samples, probabilities=probs)

        # Find best split
        split_idx, split_threshold, gain = self._best_split(X, y, rf_n_features_to_consider)
        
        # If no split was found or split has no gain
        if split_idx is None or gain <= 0.0:
            return Node(value=majority_val, impurity=node_impurity, n_samples=n_samples, probabilities=probs)
            
        # Split dataset and recurse
        X_column = X[:, split_idx]
        nan_mask = np.isnan(X_column)
        left_mask = X_column <= split_threshold
        
        # Missing value handling: assign NaNs to the side with more non-NaN samples
        n_left_non_nan: int = int(np.sum(left_mask & ~nan_mask))
        n_right_non_nan: int = int(np.sum(~left_mask & ~nan_mask))
        if n_left_non_nan >= n_right_non_nan:
            left_mask = left_mask | nan_mask
        else:
            left_mask = left_mask & ~nan_mask
            
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rf_n_features_to_consider)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rf_n_features_to_consider)
        
        return Node(feature_idx=split_idx, threshold=split_threshold, 
                    left=left_child, right=right_child, 
                    impurity=node_impurity, n_samples=n_samples)

    def _best_split(self, X: np.ndarray, y: np.ndarray, 
                    rf_n_features_to_consider: Optional[Union[int, float, str]]) -> Tuple[Optional[int], Optional[float], float]:
        n_features = X.shape[1]
        best_gain = -1.0
        split_idx = None
        split_threshold = None
        
        # Determine features to consider
        features = np.arange(n_features)
        if rf_n_features_to_consider is not None:
            if rf_n_features_to_consider == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif rf_n_features_to_consider == 'log2':
                max_features = int(np.log2(n_features))
            elif isinstance(rf_n_features_to_consider, int):
                max_features = min(n_features, rf_n_features_to_consider)
            elif isinstance(rf_n_features_to_consider, float):
                max_features = min(n_features, int(rf_n_features_to_consider * n_features))
            else:
                max_features = n_features
            max_features = max(1, max_features)
            
            # Select random subset without replacement
            assert self.rng is not None
            features = self.rng.choice(features, size=max_features, replace=False)
            
        for feat_idx in features:
            X_column = X[:, feat_idx]
            
            # Exclude NaNs when computing thresholds
            nan_mask = np.isnan(X_column)
            non_nan_col = X_column[~nan_mask]
            
            unique_vals = np.unique(non_nan_col)
            if len(unique_vals) <= 1:
                continue
                
            unique_vals = np.sort(unique_vals)
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            
            for threshold in thresholds:
                left_mask = X_column <= threshold
                
                # Direct NaNs to the side with more non-NaN samples
                n_left_non_nan: int = int(np.sum(left_mask & ~nan_mask))
                n_right_non_nan: int = int(np.sum(~left_mask & ~nan_mask))
                if n_left_non_nan >= n_right_non_nan:
                    left_mask = left_mask | nan_mask
                else:
                    left_mask = left_mask & ~nan_mask
                    
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                    
                gain = _calculate_gain(y, left_y, right_y, self.criterion)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
                    
        return split_idx, split_threshold, best_gain

    def _calculate_feature_importances(self, node: Node, total_samples: int) -> None:
        assert self.feature_importances_ is not None
        if node.is_leaf:
            return
            
        # Weighted impurity reduction
        assert node.n_samples is not None
        assert node.left is not None and node.left.n_samples is not None
        assert node.right is not None and node.right.n_samples is not None
        assert node.impurity is not None
        assert node.left.impurity is not None
        assert node.right.impurity is not None
        assert node.feature_idx is not None
        
        n_parent: int = node.n_samples
        n_left: int = node.left.n_samples
        n_right: int = node.right.n_samples
        
        weight: float = n_parent / total_samples
        left_ratio: float = n_left / n_parent
        right_ratio: float = n_right / n_parent
        
        gain: float = node.impurity - (left_ratio * node.left.impurity + right_ratio * node.right.impurity)
        
        self.feature_importances_[node.feature_idx] += weight * gain
        
        self._calculate_feature_importances(node.left, total_samples)
        self._calculate_feature_importances(node.right, total_samples)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        X = np.asarray(X, dtype=np.float64)
        if self.root is None:
            raise ValueError("DecisionTreeClassifier is not fitted yet")
            
        predictions = [self._predict_one(x, self.root) for x in X]
        return np.array(predictions)

    def _predict_one(self, x: np.ndarray, node: Node) -> Union[int, str]:
        if node.is_leaf:
            assert node.value is not None
            return node.value
            
        assert node.feature_idx is not None
        val = x[node.feature_idx]
        if np.isnan(val):
            # Missing value handling: route to the child with more samples
            assert node.left is not None and node.left.n_samples is not None
            assert node.right is not None and node.right.n_samples is not None
            if node.left.n_samples >= node.right.n_samples:
                return self._predict_one(x, node.left)
            else:
                return self._predict_one(x, node.right)
            
        assert node.threshold is not None
        assert node.left is not None
        assert node.right is not None
        if val <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        X = np.asarray(X, dtype=np.float64)
        if self.root is None:
            raise ValueError("DecisionTreeClassifier is not fitted yet")
            
        prob_dicts = [self._predict_proba_one(x, self.root) for x in X]
        
        n_samples = len(X)
        assert self.classes_ is not None
        n_classes = len(self.classes_)
        proba: np.ndarray = np.zeros((n_samples, n_classes), dtype=np.float64)
        
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        for i, p_dict in enumerate(prob_dicts):
            for c, p in p_dict.items():
                if c in class_to_idx:
                    proba[i, class_to_idx[c]] = p
                    
        return proba

    def _predict_proba_one(self, x: np.ndarray, node: Node) -> Dict[Union[int, str], float]:
        if node.is_leaf:
            assert node.probabilities is not None
            return node.probabilities
            
        assert node.feature_idx is not None
        val = x[node.feature_idx]
        if np.isnan(val):
            # Missing value handling: route to the child with more samples
            assert node.left is not None and node.left.n_samples is not None
            assert node.right is not None and node.right.n_samples is not None
            if node.left.n_samples >= node.right.n_samples:
                return self._predict_proba_one(x, node.left)
            else:
                return self._predict_proba_one(x, node.right)
                
        assert node.threshold is not None
        assert node.left is not None
        assert node.right is not None
        if val <= node.threshold:
            return self._predict_proba_one(x, node.left)
        else:
            return self._predict_proba_one(x, node.right)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))

    def _get_depth(self, node: Optional[Node]) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    @property
    def depth(self) -> int:
        return self._get_depth(self.root)

    @property
    def tree_depth(self) -> int:
        return self.depth

    @property
    def max_depth_(self) -> int:
        return self.depth

    def get_depth(self) -> int:
        return self.depth


# Alias DecisionTree to DecisionTreeClassifier
DecisionTree = DecisionTreeClassifier
