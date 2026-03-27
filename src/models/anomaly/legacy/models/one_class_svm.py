"""One-Class SVM implementation for anomaly detection."""

import numpy as np
from sklearn.svm import OneClassSVM
from .base import AnomalyDetector


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM based anomaly detector."""

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str = "scale"
    ):
        """
        Initialize One-Class SVM detector.

        Args:
            nu: Upper bound on fraction of training errors (similar to contamination).
                Must be in (0, 1].
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
            gamma: Kernel coefficient ('scale', 'auto', or float).
        """
        super().__init__(contamination=nu)
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit One-Class SVM on training data.

        IMPORTANT: Should only be trained on Class=0 (legitimate) data.

        Args:
            X: Training data of shape (n_samples, n_features).
        """
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.model.fit(X)
        self.is_fitted = True

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        One-Class SVM returns signed distances to separating hyperplane.
        Negative values indicate anomalies. We convert to positive scores.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
            Higher scores indicate higher likelihood of being anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # Get raw scores (negative for anomalies)
        raw_scores = self.model.score_samples(X)
        # Convert to positive scores (higher = more anomalous)
        return -raw_scores
