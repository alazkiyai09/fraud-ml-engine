"""Abstract base class for model explainers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np


class BaseExplainer(ABC):
    """
    Abstract base class defining the interface for model explainers.

    All explainers must implement local and global explanation methods.
    Ensures consistency across different explanation techniques.
    """

    def __init__(self, model: Any, model_type: str):
        """
        Initialize the explainer with a model.

        Args:
            model: The trained model to explain
            model_type: Type of model ('xgboost', 'random_forest', 'neural_network', etc.)
        """
        self.model = model
        self.model_type = model_type

    @abstractmethod
    def explain_local(
        self,
        X: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Generate local explanation for a single prediction.

        Args:
            X: Single instance to explain (shape: [1, n_features] or [n_features])
            feature_names: List of feature names
            **kwargs: Additional explainer-specific parameters

        Returns:
            Dictionary mapping feature names to importance scores for top 5 features
            Format: {feature_name: importance_score}
        """
        pass

    @abstractmethod
    def explain_global(
        self,
        X: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Generate global feature importance explanations.

        Args:
            X: Dataset to explain (shape: [n_samples, n_features])
            feature_names: List of feature names
            **kwargs: Additional explainer-specific parameters

        Returns:
            Dictionary mapping feature names to global importance scores
            Format: {feature_name: importance_score}
        """
        pass

    def get_top_features(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Extract top N features by importance score.

        Args:
            feature_importance: Dictionary of feature names to scores
            top_n: Number of top features to return

        Returns:
            List of (feature_name, score) tuples sorted by score (descending)
        """
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:top_n]

    def validate_input(self, X: np.ndarray, feature_names: List[str]) -> None:
        """
        Validate input dimensions and data.

        Args:
            X: Input data
            feature_names: List of feature names

        Raises:
            ValueError: If input validation fails
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature mismatch: X has {X.shape[1]} features, "
                f"but {len(feature_names)} names provided"
            )
