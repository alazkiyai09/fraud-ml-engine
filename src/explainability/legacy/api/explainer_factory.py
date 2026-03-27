"""Factory for creating model explainers."""

from typing import Any, Dict, List, Optional
import numpy as np

from ..explainers.base import BaseExplainer
from ..explainers.shap_explainer import SHAPExplainer
from ..explainers.lime_explainer import LIMEExplainer
from ..explainers.pdp_explainer import PDPExplainer


class ExplainerFactory:
    """
    Factory for creating appropriate explainers based on model type and user preferences.

    Supports:
    - Automatic explainer selection based on model type
    - Manual explainer selection
    - Multiple explainer types (SHAP, LIME, PDP)
    """

    EXPLAINER_TYPES = ['shap', 'lime', 'pdp']

    MODEL_TYPE_MAPPING = {
        'xgboost': {
            'recommended': 'shap',
            'supported': ['shap', 'lime', 'pdp']
        },
        'random_forest': {
            'recommended': 'shap',
            'supported': ['shap', 'lime', 'pdp']
        },
        'gradient_boosting': {
            'recommended': 'shap',
            'supported': ['shap', 'lime', 'pdp']
        },
        'neural_network': {
            'recommended': 'shap',  # DeepExplainer or KernelExplainer
            'supported': ['shap', 'lime']
        },
        'sklearn_generic': {
            'recommended': 'lime',
            'supported': ['shap', 'lime']
        },
        'generic': {
            'recommended': 'lime',
            'supported': ['lime']
        }
    }

    def __init__(
        self,
        default_explainer: str = 'shap',
        fallback_explainer: str = 'lime'
    ):
        """
        Initialize factory.

        Args:
            default_explainer: Default explainer type to use
            fallback_explainer: Fallback if default explainer fails
        """
        if default_explainer not in self.EXPLAINER_TYPES:
            raise ValueError(
                f"Unknown explainer type: {default_explainer}. "
                f"Supported: {self.EXPLAINER_TYPES}"
            )

        self.default_explainer = default_explainer
        self.fallback_explainer = fallback_explainer

    def create(
        self,
        model: Any,
        model_type: str,
        explainer_type: Optional[str] = None,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> BaseExplainer:
        """
        Create an explainer instance.

        Args:
            model: Trained model to explain
            model_type: Type of model ('xgboost', 'random_forest', 'neural_network', etc.)
            explainer_type: Type of explainer to create (None = auto-select)
            training_data: Training data (required for some explainers)
            feature_names: List of feature names (required for LIME)
            **kwargs: Additional explainer-specific parameters

        Returns:
            BaseExplainer instance

        Raises:
            ValueError: If explainer type is not supported
            RuntimeError: If explainer creation fails
        """
        # Auto-select explainer if not specified
        if explainer_type is None:
            explainer_type = self._get_recommended_explainer(model_type)

        # Validate explainer type is supported for this model
        if not self._is_explainer_supported(model_type, explainer_type):
            raise ValueError(
                f"Explainer '{explainer_type}' is not supported for model type '{model_type}'. "
                f"Supported: {self._get_supported_explainers(model_type)}"
            )

        # Create explainer
        try:
            explainer = self._create_explainer(
                model=model,
                model_type=model_type,
                explainer_type=explainer_type,
                training_data=training_data,
                feature_names=feature_names,
                **kwargs
            )
            return explainer

        except Exception as e:
            # Try fallback explainer if available
            if explainer_type != self.fallback_explainer:
                if self._is_explainer_supported(model_type, self.fallback_explainer):
                    print(
                        f"Warning: Failed to create {explainer_type} explainer. "
                        f"Attempting fallback to {self.fallback_explainer}. "
                        f"Error: {str(e)}"
                    )
                    return self.create(
                        model=model,
                        model_type=model_type,
                        explainer_type=self.fallback_explainer,
                        training_data=training_data,
                        feature_names=feature_names,
                        **kwargs
                    )

            raise RuntimeError(
                f"Failed to create {explainer_type} explainer: {str(e)}"
            )

    def _create_explainer(
        self,
        model: Any,
        model_type: str,
        explainer_type: str,
        training_data: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        **kwargs
    ) -> BaseExplainer:
        """Create specific explainer instance."""

        if explainer_type == 'shap':
            return SHAPExplainer(
                model=model,
                model_type=model_type,
                training_data=training_data,
                **kwargs
            )

        elif explainer_type == 'lime':
            if training_data is None:
                raise ValueError("training_data is required for LIME explainer")
            if feature_names is None:
                raise ValueError("feature_names is required for LIME explainer")

            return LIMEExplainer(
                model=model,
                training_data=training_data,
                feature_names=feature_names,
                model_type=model_type,
                **kwargs
            )

        elif explainer_type == 'pdp':
            if training_data is None:
                raise ValueError("training_data is required for PDP explainer")
            if feature_names is None:
                raise ValueError("feature_names is required for PDP explainer")

            return PDPExplainer(
                model=model,
                model_type=model_type,
                X_train=training_data,
                feature_names=feature_names,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def _get_recommended_explainer(self, model_type: str) -> str:
        """Get recommended explainer for model type."""
        model_info = self.MODEL_TYPE_MAPPING.get(
            model_type,
            self.MODEL_TYPE_MAPPING['generic']
        )
        return model_info['recommended']

    def _get_supported_explainers(self, model_type: str) -> List[str]:
        """Get list of supported explainers for model type."""
        model_info = self.MODEL_TYPE_MAPPING.get(
            model_type,
            self.MODEL_TYPE_MAPPING['generic']
        )
        return model_info['supported']

    def _is_explainer_supported(self, model_type: str, explainer_type: str) -> bool:
        """Check if explainer is supported for model type."""
        supported = self._get_supported_explainers(model_type)
        return explainer_type in supported

    def create_multiple(
        self,
        model: Any,
        model_type: str,
        explainer_types: List[str],
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, BaseExplainer]:
        """
        Create multiple explainers at once.

        Useful for comparing different explanation methods.

        Args:
            model: Trained model to explain
            model_type: Type of model
            explainer_types: List of explainer types to create
            training_data: Training data
            feature_names: List of feature names
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping explainer type to explainer instance
        """
        explainers = {}

        for exp_type in explainer_types:
            try:
                explainer = self.create(
                    model=model,
                    model_type=model_type,
                    explainer_type=exp_type,
                    training_data=training_data,
                    feature_names=feature_names,
                    **kwargs
                )
                explainers[exp_type] = explainer
            except Exception as e:
                print(f"Warning: Could not create {exp_type} explainer: {str(e)}")

        return explainers


def create_explainer(
    model: Any,
    model_type: str,
    explainer_type: str = 'shap',
    training_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    **kwargs
) -> BaseExplainer:
    """
    Convenience function to create an explainer.

    Args:
        model: Trained model to explain
        model_type: Type of model ('xgboost', 'random_forest', 'neural_network', etc.)
        explainer_type: Type of explainer to create (default: 'shap')
        training_data: Training data (required for some explainers)
        feature_names: List of feature names (required for LIME, PDP)
        **kwargs: Additional explainer-specific parameters

    Returns:
        BaseExplainer instance
    """
    factory = ExplainerFactory()
    return factory.create(
        model=model,
        model_type=model_type,
        explainer_type=explainer_type,
        training_data=training_data,
        feature_names=feature_names,
        **kwargs
    )
