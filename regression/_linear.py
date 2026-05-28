# GenAI Usage Declaration
# This implementation was developed primarily by myself.
# I used ChatGPT (OpenAI) as a support tool for:
#   - clarifying theoretical concepts (e.g., linear regression, gradient descent),
#   - discussing implementation approaches,
#   - debugging and improving code quality.
# All code was written, reviewed, and adapted by me. I understand the full
# implementation and can explain all design decisions and mathematical steps.
# The final solution reflects my own work and understanding.
import numpy as np

def _validate_inputs(X, y=None, n_features=None):
    """
    Validates input matrices X and targets y for regression tasks.
    Ensures safe numeric types, correct dimensionality, checks for empty datasets, 
    and guarantees matching sample sizes.
    """
    # 1. Enforce Numeric Types
    try:
        X = np.asarray(X, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Input X cannot be safely converted to a numeric array.") from exc
        
    if X.size == 0:
        raise ValueError("Input X is empty.")

    # 2. Defensively ensure X is a 2D matrix
    # Convention: 1D arrays are explicitly reshaped as (n_samples, 1) representing a single feature
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        raise ValueError(f"Expected 1D or 2D array for X, got {X.ndim}D array.")

    # Feature consistency check for prediction vs training
    if n_features is not None:
        if X.shape[1] != n_features:
            raise ValueError(f"X has {X.shape[1]} features, but model is expecting {n_features} features.")

    # 3. Target shape validation
    if y is not None:
        try:
            y = np.asarray(y, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Input y cannot be safely converted to a numeric array.") from exc

        if y.size == 0:
            raise ValueError("Input y is empty.")
            
        # Safely ensure y is a 1D vector (n_samples,)
        if y.ndim != 1:
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ValueError(f"Target y has invalid shape {y.shape}. "
                                 "Expected (n_samples,) or (n_samples, 1).")

        # 4. Consistency check
        if len(X) != len(y):
            raise ValueError(f"Inconsistent samples: X({len(X)}), y({len(y)}).")
            
        return X, y
        
    return X


class LinearRegressor:
    """
    Linear Regression model implementing the closed-form Normal Equation.
    """
    def __init__(self):
        self.weights_ = None
        self.bias_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressor':
        """
        Train the model using the normal equation closed-form solution.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,) or (n_samples, 1).
            
        Returns:
            self: The fitted regressor instance.
        """
        # Strictly validate and sanitize numeric inputs
        X, y = _validate_inputs(X, y)
        # Efficiently build the augmented matrix [1, X] without unnecessary copies
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        X_b = np.empty((n_samples, n_features + 1))
        X_b[:, 0] = 1
        X_b[:, 1:] = X
        
        # Perform Least Squares solution using lstsq for maximum robustness
        # This handles rank-deficient / singular matrices gracefully.
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

        # Map coefficients accurately to specialized attributes
        self.bias_ = float(theta[0])
        self.weights_ = np.asarray(theta[1:]).ravel()

        # Sklearn-compatible aliases
        self.coef_ = self.weights_
        self.intercept_ = self.bias_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            y_pred: Predicted target values as a 1D array of shape (n_samples,).
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        # Validate inputs natively matching the training expected feature dimension
        X = _validate_inputs(X, n_features=self.n_features_in_)
        
        # Vectorized linear combination prediction
        return X @ self.weights_ + self.bias_


class SGDRegression:
    """
    Linear Regression using mini-batch stochastic gradient descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, 
                 random_state=None, l2_penalty=0.0001):
        if learning_rate <= 0 or n_iterations <= 0 or batch_size < 1:
            raise ValueError("learning_rate/n_iterations must be > 0 and batch_size >= 1.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.l2_penalty = l2_penalty

        self.weights_ = None
        self.bias_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SGDRegression':
        """
        Fit the model using mini-batch Stochastic Gradient Descent on MSE loss.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self: The fitted regressor instance.
        """
        X, y = _validate_inputs(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Internal feature standardisation: zero-mean, unit-variance per feature.
        # This normalises the loss surface so gradient steps are equally scaled
        # across all features, improving convergence speed and coefficient accuracy.
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0.0] = 1.0          # guard against constant features
        X_scaled = (X - X_mean) / X_std

        # Temporary weights in the scaled space
        weights_scaled = np.zeros(n_features)
        bias_scaled = 0.0

        rng = np.random.default_rng(self.random_state)
        tol = 1e-6
        prev_weights = weights_scaled.copy()
        self.n_iter_ = 0

        for _ in range(self.n_iterations):
            self.n_iter_ += 1
            # Optimization: use precomputed shuffled indices to avoid dataset copies
            indices = rng.permutation(n_samples)

            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                X_batch = X_scaled[batch_idx]
                y_batch = y[batch_idx]

                m_batch = len(y_batch)

                y_pred = X_batch @ weights_scaled + bias_scaled
                error = y_pred - y_batch

                # Gradients: MSE term + L2 regularization (penalty)
                dw = (2 / m_batch) * (X_batch.T @ error) + self.l2_penalty * weights_scaled
                db = (2 / m_batch) * np.sum(error)

                weights_scaled -= self.learning_rate * dw
                bias_scaled -= self.learning_rate * db

            # Convergence check (Squared distance is faster than norm)
            if np.sum((weights_scaled - prev_weights)**2) < tol**2:
                break
            prev_weights = weights_scaled.copy()

        # Convert trained parameters back to original (unscaled) feature space
        self.weights_ = weights_scaled / X_std
        self.bias_ = float(bias_scaled - np.sum((weights_scaled * X_mean) / X_std))

        # Sklearn-compatible aliases
        self.coef_ = self.weights_
        self.intercept_ = float(self.bias_)

        return self



    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input.
        
        Args:
            X: Input data of shape (n_samples, n_features).
            
        Returns:
            y_pred: Predicted target values as a 1D array of shape (n_samples,).
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        X = _validate_inputs(X, n_features=self.n_features_in_)
        
        return X @ self.weights_ + self.bias_


# Added as a compatibility alias for the notebook import, 
# while preserving the required LinearRegressor class.
LinearRegression = LinearRegressor
