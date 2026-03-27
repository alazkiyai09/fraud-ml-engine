"""Isolation Forest implementation for anomaly detection."""

import numpy as np
from sklearn.ensemble import IsolationForest
from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest based anomaly detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples = "auto",
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of outliers.
            n_estimators: Number of trees in the forest.
            max_samples: Number of samples to draw for each tree.
            random_state: Random seed.
        """
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit Isolation Forest on training data.

        IMPORTANT: Should only be trained on Class=0 (legitimate) data.

        Args:
            X: Training data of shape (n_samples, n_features).
        """
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X)
        self.is_fitted = True

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Isolation Forest returns negative scores (lower = more anomalous).
        We convert to positive scores (higher = more anomalous).

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
            Higher scores indicate higher likelihood of being anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # Get raw scores (negative, with outliers having lower values)
        raw_scores = self.model.score_samples(X)
        # Convert to positive scores (higher = more anomalous)
        return -raw_scores
