"""LIME-based explainer for fraud detection models."""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

from .base import BaseExplainer


class LIMEExplainer(BaseExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) explainer.

    LIME approximates the model locally with an interpretable linear model.
    Uses fixed random seeds to ensure consistency across runs.
    """

    def __init__(
        self,
        model: Any,
        training_data: np.ndarray,
        feature_names: List[str],
        model_type: str = 'generic',
        discretize_continuous: bool = True,
        random_state: int = 42
    ):
        """
        Initialize LIME explainer.

        Args:
            model: Trained model to explain
            training_data: Training data used to train the model (for LIME's sampling)
            feature_names: List of feature names
            model_type: Type of model (for informational purposes)
            discretize_continuous: Whether to discretize continuous features
            random_state: Random seed for reproducibility
        """
        super().__init__(model, model_type)

        if not HAS_LIME:
            raise ImportError("LIME is not installed. Install with: pip install lime")

        self.training_data = training_data
        self.feature_names = feature_names
        self.random_state = random_state

        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=feature_names,
            discretize_continuous=discretize_continuous,
            random_state=random_state,
            mode='classification'  # Fraud detection is typically binary classification
        )

    def explain_local(
        self,
        X: np.ndarray,
        feature_names: List[str],
        num_features: int = 5,
        num_samples: int = 5000
    ) -> Dict[str, float]:
        """
        Generate local LIME explanation for a single prediction.

        Args:
            X: Single instance to explain (shape: [1, n_features] or [n_features])
            feature_names: List of feature names (must match training feature_names)
            num_features: Number of top features to include in explanation
            num_samples: Number of samples to generate for local approximation

        Returns:
            Dictionary mapping top 5 feature names to LIME importance scores
        """
        self.validate_input(X, feature_names)

        # Ensure 1D array for LIME
        if X.ndim == 2 and X.shape[0] == 1:
            X_single = X[0]
        else:
            X_single = X

        # Ensure feature names match
        if feature_names != self.feature_names:
            # Map feature names if order matches
            if len(feature_names) == len(self.feature_names):
                # We'll use the index-based approach
                pass
            else:
                raise ValueError(
                    f"Feature names must match training data. "
                    f"Expected {len(self.feature_names)}, got {len(feature_names)}"
                )

        # Define prediction function
        def predict_fn(x):
            """Prediction function compatible with LIME."""
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x)
            else:
                preds = self.model.predict(x)
                # Convert to probability format
                return np.column_stack([1 - preds, preds])

        # Generate explanation with fixed seed for consistency
        exp = self.explainer.explain_instance(
            data_row=X_single,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        # Extract top features and their importance
        feature_importance = {}
        for feature, importance in exp.as_list():
            # Feature name may be in format "feature_name <= threshold" or "feature_name > threshold"
            # Extract the base feature name
            base_feature = feature.split()[0]

            # If we've seen this feature before, sum the importances
            # (LIME splits continuous features into bins)
            if base_feature in feature_importance:
                feature_importance[base_feature] += abs(importance)
            else:
                feature_importance[base_feature] = abs(importance)

        # Sort and return top 5
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return dict(sorted_features)

    def explain_global(
        self,
        X: np.ndarray,
        feature_names: List[str],
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Generate global feature importance by aggregating local explanations.

        Note: LIME is primarily a local explanation method. This method aggregates
        local explanations across multiple samples to approximate global importance.

        Args:
            X: Dataset to explain (shape: [n_samples, n_features])
            feature_names: List of feature names
            sample_size: Number of random samples to explain

        Returns:
            Dictionary mapping feature names to aggregated importance scores
        """
        self.validate_input(X, feature_names)

        # Sample instances for efficiency
        if X.shape[0] > sample_size:
            indices = np.random.choice(
                X.shape[0],
                sample_size,
                replace=False
            )
            X_sampled = X[indices]
        else:
            X_sampled = X

        # Aggregate feature importance across samples
        aggregated_importance = {}

        for i in range(X_sampled.shape[0]):
            local_exp = self.explain_local(
                X_sampled[i],
                feature_names,
                num_features=len(feature_names),  # Get all features
                num_samples=1000  # Fewer samples for speed
            )

            for feature, importance in local_exp.items():
                if feature in aggregated_importance:
                    aggregated_importance[feature] += importance
                else:
                    aggregated_importance[feature] = importance

        # Average the importances
        for feature in aggregated_importance:
            aggregated_importance[feature] /= X_sampled.shape[0]

        # Sort by importance
        sorted_importance = dict(
            sorted(
                aggregated_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        return sorted_importance

    def generate_as_html(
        self,
        X: np.ndarray,
        feature_names: List[str],
        num_features: int = 5
    ) -> str:
        """
        Generate LIME explanation as HTML.

        Args:
            X: Single instance to explain
            feature_names: List of feature names
            num_features: Number of features to show

        Returns:
            HTML string containing the visualization
        """
        self.validate_input(X, feature_names)

        if X.ndim == 2 and X.shape[0] == 1:
            X_single = X[0]
        else:
            X_single = X

        def predict_fn(x):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x)
            else:
                preds = self.model.predict(x)
                return np.column_stack([1 - preds, preds])

        exp = self.explainer.explain_instance(
            data_row=X_single,
            predict_fn=predict_fn,
            num_features=num_features
        )

        return exp.as_html()
