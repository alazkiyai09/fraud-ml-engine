"""SHAP-based explainer for fraud detection models."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from .base import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """
    SHAP (SHapley Additive exPlanations) explainer for fraud detection models.

    Supports:
    - TreeExplainer for XGBoost, Random Forest, and other tree-based models
    - KernelExplainer for neural networks and other black-box models
    - DeepExplainer for TensorFlow/Keras models
    """

    def __init__(self, model: Any, model_type: str, training_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model to explain
            model_type: Type of model ('xgboost', 'random_forest', 'neural_network', etc.)
            training_data: Background dataset for Kernel/Deep explainers (required for non-tree models)
        """
        super().__init__(model, model_type)

        if not HAS_SHAP:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        self.explainer = None
        self.training_data = training_data
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer based on model type."""
        if self.model_type in ['xgboost', 'random_forest', 'gradient_boosting']:
            # Use TreeExplainer for tree-based models (faster, more accurate)
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'neural_network':
            # Use DeepExplainer or KernelExplainer for neural networks
            if self.training_data is None:
                raise ValueError(
                    "training_data is required for neural network explainers. "
                    "Provide a background dataset (typically 100-500 random samples)."
                )
            try:
                self.explainer = shap.DeepExplainer(self.model, self.training_data)
            except Exception:
                # Fallback to KernelExplainer if DeepExplainer fails
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    self.training_data
                )
        else:
            # Default to KernelExplainer for other model types
            if self.training_data is None:
                raise ValueError(
                    "training_data is required for this model type. "
                    "Provide a background dataset (typically 100-500 random samples)."
                )
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                self.training_data
            )

    def explain_local(
        self,
        X: np.ndarray,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Generate local SHAP explanation for a single prediction.

        Args:
            X: Single instance to explain (shape: [1, n_features] or [n_features])
            feature_names: List of feature names
            background_data: Optional background data (overrides training_data)

        Returns:
            Dictionary mapping top 5 feature names to SHAP values
        """
        self.validate_input(X, feature_names)

        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For multi-class, take the positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Get SHAP values for the first (and only) instance
        shap_values_single = shap_values[0] if shap_values.ndim > 1 else shap_values

        # Create feature importance dictionary
        feature_importance = {
            feature_names[i]: float(shap_values_single[i])
            for i in range(len(feature_names))
        }

        # Return top 5 features
        top_features = self.get_top_features(feature_importance, top_n=5)
        return dict(top_features)

    def explain_global(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Generate global feature importance using SHAP.

        Args:
            X: Dataset to explain (shape: [n_samples, n_features])
            feature_names: List of feature names
            max_samples: Maximum number of samples to use for efficiency

        Returns:
            Dictionary mapping feature names to mean absolute SHAP values
        """
        self.validate_input(X, feature_names)

        # Sample for efficiency if dataset is large
        if X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sampled = X[indices]
        else:
            X_sampled = X

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sampled)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Create feature importance dictionary
        feature_importance = {
            feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(feature_names))
        }

        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    def generate_waterfall_plot(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 10
    ):
        """
        Generate SHAP waterfall plot for local explanation.

        Args:
            X: Single instance to explain
            feature_names: List of feature names
            max_display: Maximum number of features to display

        Returns:
            matplotlib Figure object
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Create explanation object
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]
        else:
            expected_value = np.mean(self.model.predict_proba(X)[:, 1])

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=X[0],
            feature_names=feature_names
        )

        return shap.plots.waterfall(explanation, max_display=max_display, show=False)

    def generate_summary_plot(
        self,
        X: np.ndarray,
        feature_names: List[str],
        max_display: int = 20
    ):
        """
        Generate SHAP summary plot for global explanation.

        Args:
            X: Dataset to explain
            feature_names: List of feature names
            max_display: Maximum number of features to display

        Returns:
            matplotlib Figure object
        """
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        return shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
