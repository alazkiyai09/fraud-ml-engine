"""Local Outlier Factor (LOF) implementation for anomaly detection."""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .base import AnomalyDetector


class LOFDetector(AnomalyDetector):
    """Local Outlier Factor based anomaly detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        algorithm: str = "auto",
        metric: str = "minkowski"
    ):
        """
        Initialize LOF detector.

        Args:
            contamination: Expected proportion of outliers.
            n_neighbors: Number of neighbors to use.
            algorithm: Algorithm for computing nearest neighbors
                      ('auto', 'ball_tree', 'kd_tree', 'brute').
            metric: Distance metric.
        """
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit LOF on training data.

        IMPORTANT: Should only be trained on Class=0 (legitimate) data.
        Note: LOF is fit using novelty=True for prediction on new data.

        Args:
            X: Training data of shape (n_samples, n_features).
        """
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            algorithm=self.algorithm,
            metric=self.metric,
            novelty=True  # Required for predict on new data
        )
        self.model.fit(X)
        self.is_fitted = True

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        LOF returns negative outlier factor (lower = more anomalous).
        We convert to positive scores.

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
