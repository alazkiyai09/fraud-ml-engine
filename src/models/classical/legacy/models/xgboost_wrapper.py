"""
XGBoost wrapper for consistent API with scikit-learn models.

Provides a wrapper around XGBoost classifier that matches
the scikit-learn interface used throughout the project.
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional
from src.models.classical.legacy.config import config


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with scikit-learn compatible API.

    Supports class weights via sample_weight parameter and
    uses scale_pos_weight for imbalanced binary classification.
    """

    def __init__(
        self,
        use_class_weight: bool = False,
        random_state: int = None,
    ):
        """
        Initialize XGBoost wrapper.

        Args:
            use_class_weight: Whether to apply class weighting
            random_state: Random seed for reproducibility
        """
        if random_state is None:
            random_state = config.RANDOM_STATE

        params = config.XGBOOST_PARAMS.copy()
        params["random_state"] = random_state

        self.model = XGBClassifier(**params)
        self.use_class_weight = use_class_weight
        self.random_state = random_state

    def _calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Calculate scale_pos_weight for imbalanced binary classification.

        XGBoost recommends: scale_pos_weight = total_negative_samples / total_positive_samples

        Args:
            y: Training labels

        Returns:
            Scale position weight
        """
        n_negative = np.sum(y == 0)
        n_positive = np.sum(y == 1)

        if n_positive == 0:
            return 1.0

        return n_negative / n_positive

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostWrapper":
        """
        Fit the model to training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        if self.use_class_weight:
            # Use scale_pos_weight for class imbalance
            scale_pos_weight = self._calculate_scale_pos_weight(y)
            self.model.set_params(scale_pos_weight=scale_pos_weight)

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
