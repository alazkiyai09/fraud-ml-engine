"""Abstract base class for anomaly detection models."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AnomalyDetector(ABC):
    """Abstract base class for all anomaly detection models."""

    def __init__(self, contamination: float = 0.1):
        """
        Initialize the anomaly detector.

        Args:
            contamination: The expected proportion of outliers in the dataset.
                          Used for threshold determination.
        """
        self.contamination = contamination
        self.threshold = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:  # type: ignore[valid-type]
        """
        Fit the anomaly detector on training data.

        IMPORTANT: Should only be trained on Class=0 (legitimate) data.

        Args:
            X: Training data of shape (n_samples, n_features).
        """
        pass

    @abstractmethod
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:  # type: ignore[valid-type]
        """
        Compute anomaly scores for samples.

        Higher scores indicate higher likelihood of being anomalous.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
        """
        pass

    def set_threshold(self, X_val: np.ndarray, target_fpr: float = 0.01) -> float:  # type: ignore[valid-type]
        """
        Set decision threshold based on validation data to achieve target FPR.

        Args:
            X_val: Validation data (should be mostly legitimate).
            target_fpr: Target false positive rate (default: 0.01 = 1%).

        Returns:
            The computed threshold value.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before setting threshold.")

        scores = self.predict_anomaly_score(X_val)
        # Find threshold that achieves target FPR
        self.threshold = np.quantile(scores, 1 - target_fpr)
        return self.threshold

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:  # type: ignore[valid-type]
        """
        Predict binary labels (0=normal, 1=anomaly) based on threshold.

        Args:
            X: Input data of shape (n_samples, n_features).
            threshold: Decision threshold. If None, uses self.threshold.

        Returns:
            Binary predictions of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if threshold is None:
            if self.threshold is None:
                raise RuntimeError("Threshold not set. Call set_threshold() or provide threshold.")
            threshold = self.threshold

        scores = self.predict_anomaly_score(X)
        return (scores >= threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[valid-type]
        """
        Fit the model and return predictions on the same data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            Binary predictions of shape (n_samples,).
        """
        self.fit(X)
        # Use contamination to set threshold
        scores = self.predict_anomaly_score(X)
        self.threshold = np.quantile(scores, 1 - self.contamination)
        return self.predict(X)
