"""
Baseline models for imbalanced classification.

Implements Logistic Regression and Random Forest classifiers
with optional class weighting support.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Union
from src.models.classical.legacy.config import config


class LogisticRegressionBaseline(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression baseline classifier.

    Supports class_weight parameter for handling imbalanced data.
    """

    def __init__(
        self,
        class_weight: Optional[Union[str, dict]] = None,
        random_state: int = None,
    ):
        """
        Initialize Logistic Regression baseline.

        Args:
            class_weight: Class weight strategy ('balanced' or dict)
            random_state: Random seed for reproducibility
        """
        if random_state is None:
            random_state = config.RANDOM_STATE

        params = config.LOGISTIC_REGRESSION_PARAMS.copy()
        params["random_state"] = random_state

        if class_weight is not None:
            params["class_weight"] = class_weight

        self.model = LogisticRegression(**params)
        self.class_weight = class_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        """
        Fit the model to training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Test features

        Returns:
            Predicted binary labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates.

        Args:
            X: Test features

        Returns:
            Probability estimates for both classes (n_samples, 2)
        """
        return self.model.predict_proba(X)


class RandomForestBaseline(BaseEstimator, ClassifierMixin):
    """
    Random Forest baseline classifier.

    Supports class_weight parameter for handling imbalanced data.
    """

    def __init__(
        self,
        class_weight: Optional[Union[str, dict]] = None,
        random_state: int = None,
    ):
        """
        Initialize Random Forest baseline.

        Args:
            class_weight: Class weight strategy ('balanced' or dict)
            random_state: Random seed for reproducibility
        """
        if random_state is None:
            random_state = config.RANDOM_STATE

        params = config.RANDOM_FOREST_PARAMS.copy()
        params["random_state"] = random_state

        if class_weight is not None:
            params["class_weight"] = class_weight

        self.model = RandomForestClassifier(**params)
        self.class_weight = class_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        """
        Fit the model to training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Test features

        Returns:
            Predicted binary labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates.

        Args:
            X: Test features

        Returns:
            Probability estimates for both classes (n_samples, 2)
        """
        return self.model.predict_proba(X)
