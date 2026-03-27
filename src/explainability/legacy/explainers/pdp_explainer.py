"""Partial Dependence Plot explainer for global feature insights."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    HAS_SKLEARN_INSPECTION = True
except ImportError:
    HAS_SKLEARN_INSPECTION = False

import matplotlib.pyplot as plt

from .base import BaseExplainer


class PDPExplainer(BaseExplainer):
    """
    Partial Dependence Plot (PDP) explainer for global feature insights.

    PDP shows the marginal effect of one or two features on the predicted outcome.
    Useful for understanding global feature behavior and identifying non-linearities.
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        X_train: np.ndarray,
        feature_names: List[str]
    ):
        """
        Initialize PDP explainer.

        Args:
            model: Trained model to explain
            model_type: Type of model
            X_train: Training data (required for calculating partial dependence)
            feature_names: List of feature names
        """
        super().__init__(model, model_type)

        if not HAS_SKLEARN_INSPECTION:
            raise ImportError(
                "scikit-learn inspection module not available. "
                "Update scikit-learn: pip install scikit-learn>=1.0"
            )

        self.X_train = X_train
        self.feature_names = feature_names
        self.feature_name_to_idx = {
            name: idx for idx, name in enumerate(feature_names)
        }

    def explain_local(
        self,
        X: np.ndarray,
        feature_names: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        PDP is a global method - local explanation not applicable.

        This method returns the ICE (Individual Conditional Expectation) values,
        which are the local counterpart to PDP.

        Args:
            X: Single instance to explain
            feature_names: List of feature names
            **kwargs: Additional parameters

        Returns:
            Dictionary of feature to ICE value deviation from average
        """
        raise NotImplementedError(
            "PDP is a global explanation method. Use explain_global() for "
            "Partial Dependence Plots, or use SHAP/LIME for local explanations."
        )

    def explain_global(
        self,
        X: np.ndarray,
        feature_names: List[str],
        features: Optional[List[str]] = None,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Calculate partial dependence for specified features.

        Args:
            X: Dataset (uses X_train if not provided)
            feature_names: List of feature names
            features: List of specific features to analyze (if None, analyzes all)
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary with PDP results for each feature
        """
        if features is None:
            features = feature_names

        results = {}

        for feature in features:
            if feature not in self.feature_name_to_idx:
                continue

            feature_idx = self.feature_name_to_idx[feature]

            # Calculate partial dependence
            pdp_result = partial_dependence(
                self.model,
                self.X_train,
                features=[feature_idx],
                kind='average',
                n_jobs=n_jobs
            )

            results[feature] = {
                'values': pdp_result['values'][0],  # Feature values
                'average': pdp_result['average'][0],  # Average predictions
                'feature_name': feature,
                'min_value': float(np.min(pdp_result['values'][0])),
                'max_value': float(np.max(pdp_result['values'][0])),
                'min_prediction': float(np.min(pdp_result['average'][0])),
                'max_prediction': float(np.max(pdp_result['average'][0])),
                'range': float(np.max(pdp_result['average'][0]) - np.min(pdp_result['average'][0]))
            }

        return results

    def generate_pd_plot(
        self,
        feature: str,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Generate a Partial Dependence Plot for a single feature.

        Args:
            feature: Feature name to plot
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        if feature not in self.feature_name_to_idx:
            raise ValueError(f"Feature '{feature}' not found in training data")

        feature_idx = self.feature_name_to_idx[feature]

        fig, ax = plt.subplots(figsize=figsize)

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_train,
            features=[feature_idx],
            feature_names=self.feature_names,
            ax=ax,
            kind='average'
        )

        ax.set_title(f'Partial Dependence Plot: {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Average Prediction')

        return fig

    def generate_2way_pd_plot(
        self,
        feature1: str,
        feature2: str,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Generate a 2-way Partial Dependence Plot showing interaction effects.

        Args:
            feature1: First feature name
            feature2: Second feature name
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        if feature1 not in self.feature_name_to_idx:
            raise ValueError(f"Feature '{feature1}' not found")
        if feature2 not in self.feature_name_to_idx:
            raise ValueError(f"Feature '{feature2}' not found")

        feature_idx1 = self.feature_name_to_idx[feature1]
        feature_idx2 = self.feature_name_to_idx[feature2]

        fig, ax = plt.subplots(figsize=figsize)

        display = PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_train,
            features=[(feature_idx1, feature_idx2)],
            feature_names=self.feature_names,
            ax=ax,
            kind='average'
        )

        ax.set_title(f'2-Way Partial Dependence: {feature1} vs {feature2}')

        return fig

    def get_feature_importance_by_pd_range(
        self,
        X: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> Dict[str, float]:
        """
        Rank features by their effect on predictions (using PDP range).

        Features with larger PDP ranges have stronger effects on predictions.

        Args:
            X: Dataset (uses X_train if not provided)
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            Dictionary mapping feature names to their PDP ranges
        """
        if X is None:
            X = self.X_train
        if feature_names is None:
            feature_names = self.feature_names

        pdp_results = self.explain_global(X, feature_names)

        # Extract ranges
        feature_ranges = {
            feature: result['range']
            for feature, result in pdp_results.items()
        }

        # Sort by range
        sorted_ranges = dict(
            sorted(feature_ranges.items(), key=lambda x: x[1], reverse=True)
        )

        # Return top N
        return dict(list(sorted_ranges.items())[:top_n])

    def detect_nonlinear_features(
        self,
        feature_names: Optional[List[str]] = None,
        linearity_threshold: float = 0.95
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect features with non-linear relationships to predictions.

        Args:
            feature_names: List of features to check
            linearity_threshold: Correlation threshold for linearity (0-1)

        Returns:
            Dictionary of feature to analysis results
        """
        if feature_names is None:
            feature_names = self.feature_names

        results = {}

        for feature in feature_names:
            if feature not in self.feature_name_to_idx:
                continue

            # Get PDP results
            pdp_result = self.explain_global(
                self.X_train,
                self.feature_names,
                features=[feature]
            )

            if feature not in pdp_result:
                continue

            feature_data = pdp_result[feature]
            values = feature_data['values']
            averages = feature_data['average']

            # Calculate correlation to detect linearity
            correlation = np.corrcoef(values, averages)[0, 1]

            results[feature] = {
                'is_linear': abs(correlation) >= linearity_threshold,
                'correlation': float(correlation),
                'pdp_range': float(feature_data['range']),
                'recommendation': (
                    "Linear relationship" if abs(correlation) >= linearity_threshold
                    else "Non-linear relationship - may require careful interpretation"
                )
            }

        return results
