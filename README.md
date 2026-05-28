# Machine Learning Algorithms From Scratch

This repository contains Python implementations of machine learning and optimization algorithms built from scratch using numerical libraries, mainly NumPy. The goal is to translate mathematical formulations and pseudo-code into concrete, readable implementations without relying on high-level machine learning libraries such as scikit-learn, TensorFlow, or PyTorch.

The project follows the Machine Learning Lab topics for SoSe 26 and will continue to grow as more algorithms are covered.

## Goals

- Implement core machine learning algorithms independently.
- Use only numerical Python libraries for the algorithm logic.
- Understand each algorithm from the mathematical and implementation side.
- Build evaluation tools such as accuracy, precision, recall, F1 score, and other metrics.
- Test implementations on standardized datasets and identify possible implementation errors.

## Current Status

| Category | Algorithm | Status | Location |
|---|---|---:|---|
| Regression | Linear Regression | Implemented | `regression/_linear.py` |
| Optimization | Stochastic Gradient Descent for Regression | Implemented | `regression/_linear.py` |
| Classification | Logistic Regression | Implemented | `regression/_logistic.py` |
| Optimization | SGD Classifier | Implemented | `regression/_logistic.py` |
| Bayesian Learning | Gaussian Naive Bayes | Implemented | `naive_bayes/_naive_bayes.py` |
| Bayesian Learning | Multinomial Naive Bayes | Implemented | `naive_bayes/_naive_bayes.py` |
| Clustering | K-Means | Implemented | `clustering/_kmeans.py` |
| Clustering | K-Means++ | Implemented | `clustering/_kmeans.py` |

## Course Roadmap

These are the main topics from the course schedule and the planned direction for this repository.

| Topic | Planned Work |
|---|---|
| Evaluation | Accuracy, precision, recall, F1 score, confusion matrix, average precision |
| Linear Regression | Closed-form solution, gradient descent variants, regression metrics |
| Logistic Regression | Binary classification, probability prediction, classification metrics |
| K-Means | Standard K-Means, K-Means++ initialization, clustering evaluation |
| Naive Bayes | Gaussian and Multinomial Naive Bayes |
| Decision Trees | Classification trees, splitting criteria, prediction traversal |
| Ensembles | Random Forests and bagging-based methods |
| Support Vector Machines | SVM classifier and possibly Support Vector Regression |
| Artificial Neural Networks | Feedforward neural networks and backpropagation |
| Advanced Deep Learning | Deeper neural network training methods |
| Convolutional Neural Networks | Basic CNN layers and image-classification experiments |

## Repository Structure

```text
mlab/
├── clustering/
│   ├── __init__.py
│   └── _kmeans.py
├── naive_bayes/
│   ├── __init__.py
│   └── _naive_bayes.py
├── regression/
│   ├── __init__.py
│   ├── _linear.py
│   └── _logistic.py
├── __init__.py
├── .gitignore
└── README.md
```

## Example Usage

```python
import numpy as np

from regression import LinearRegressor, LogisticRegression, SGDRegression
from naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes
from clustering import KMeans, KMeansPlusPlus


# Linear regression
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.0, 6.0, 8.0])

model = LinearRegressor()
model.fit(X, y)
print(model.predict([[5.0]]))


# K-Means clustering
X_cluster = np.array([
    [1.0, 1.0],
    [1.2, 0.9],
    [8.0, 8.0],
    [8.3, 7.7],
])

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_cluster)
print(labels)
```

## Design Principles

- Algorithms should be implemented in a clear and educational style.
- NumPy vectorization should be used where it improves clarity and performance.
- Models should expose familiar methods such as `fit`, `predict`, and `predict_proba` where appropriate.
- Implementations should include validation for common input errors.
- Evaluation should be implemented manually instead of delegated to high-level ML libraries.

## Planned Additions

- Evaluation metrics package
- Train/test split helpers
- Decision Tree classifier
- Random Forest classifier
- DBSCAN or Hierarchical Agglomerative Clustering
- Support Vector Machine
- Feedforward Neural Network
- AdaGrad optimizer
- More examples and tests

## Academic Integrity Note

This repository is intended as a learning project. The implementations are written to demonstrate understanding of the algorithms, their mathematical foundations, and their practical behavior in Python.
