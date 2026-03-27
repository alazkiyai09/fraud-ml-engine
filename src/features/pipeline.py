"""Fraud feature engineering pipeline."""

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

from src.features.transformers.velocity_features import VelocityFeatures
from src.features.transformers.deviation_features import DeviationFeatures
from src.features.transformers.merchant_features import MerchantRiskFeatures
from src.features.feature_selection.shap_selector import SHAPSelector


class FraudFeaturePipeline:
    """
    Complete fraud detection feature engineering pipeline.

    Combines velocity, deviation, and merchant risk features using FeatureUnion,
    followed by optional SHAP-based feature selection.

    Parameters
    ----------
    user_col : str
        User identifier column name
    merchant_col : str
        Merchant identifier column name
    datetime_col : str
        Transaction timestamp column name
    amount_col : str
        Transaction amount column name
    time_windows : list, optional
        Time windows for velocity features
    velocity_features : list, optional
        Velocity feature types to compute
    deviation_features : list, optional
        Features to compute deviations for
    merchant_alpha : float, optional
        Bayesian smoothing alpha for merchant features
    merchant_beta : float, optional
        Bayesian smoothing beta for merchant features
    use_shap_selection : bool, optional
        Whether to use SHAP-based feature selection
    n_features : int, optional
        Number of features to select with SHAP
    scale_features : bool, optional
        Whether to scale features with StandardScaler

    Attributes
    ----------
    pipeline_ : Pipeline
        Fitted sklearn pipeline
    feature_names_ : list
        List of feature names after transformation

    Examples
    --------
    >>> pipeline = FraudFeaturePipeline(
    ...     user_col='user_id',
    ...     merchant_col='merchant_id',
    ...     datetime_col='timestamp',
    ...     amount_col='amount'
    ... )
    >>> X_features = pipeline.fit_transform(X, y=fraud_labels)
    """

    def __init__(
        self,
        user_col: str = "user_id",
        merchant_col: str = "merchant_id",
        datetime_col: str = "timestamp",
        amount_col: str = "amount",
        time_windows: list = [(1, "h"), (24, "h"), (7, "d")],
        velocity_features: list = None,
        deviation_features: list = None,
        merchant_alpha: float = 1.0,
        merchant_beta: float = 1.0,
        use_shap_selection: bool = False,
        n_features: int = 20,
        scale_features: bool = False,
    ):
        self.user_col = user_col
        self.merchant_col = merchant_col
        self.datetime_col = datetime_col
        self.amount_col = amount_col
        self.time_windows = time_windows
        self.velocity_features = velocity_features or ["count", "sum", "mean"]
        self.deviation_features = deviation_features or [amount_col]
        self.merchant_alpha = merchant_alpha
        self.merchant_beta = merchant_beta
        self.use_shap_selection = use_shap_selection
        self.n_features = n_features
        self.scale_features = scale_features

    def _build_pipeline(self) -> Pipeline:
        """Build the feature engineering pipeline."""
        # Feature union to combine all transformers
        feature_union = FeatureUnion(
            transformer_list=[
                (
                    "velocity",
                    VelocityFeatures(
                        user_col=self.user_col,
                        datetime_col=self.datetime_col,
                        amount_col=self.amount_col,
                        time_windows=self.time_windows,
                        features=self.velocity_features,
                    ),
                ),
                (
                    "deviation",
                    DeviationFeatures(
                        user_col=self.user_col,
                        features=self.deviation_features,
                    ),
                ),
            ]
        )

        # Build pipeline steps
        steps = [("features", feature_union)]

        # Optionally add SHAP selection
        if self.use_shap_selection:
            steps.append(
                (
                    "shap_selection",
                    SHAPSelector(n_features=self.n_features),
                )
            )

        # Optionally add scaling
        if self.scale_features:
            steps.append(("scaler", StandardScaler()))

        return Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FraudFeaturePipeline":
        """
        Fit the feature engineering pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Input transaction data
        y : pd.Series, optional
            Fraud labels (required for merchant features and SHAP selection)

        Returns
        -------
        self : FraudFeaturePipeline
            Fitted pipeline
        """
        # Build and fit pipeline
        self.pipeline_ = self._build_pipeline()

        # Fit pipeline (merchant features and SHAP selector need y)
        if y is not None:
            self.pipeline_.fit(X, y)
        else:
            # Fit features that don't require y
            if hasattr(self.pipeline_.named_steps["features"], "transformer_list"):
                for name, transformer in self.pipeline_.named_steps[
                    "features"
                ].transformer_list:
                    if name != "merchant":  # Skip merchant features without y
                        transformer.fit(X, y)
            # Fit SHAP selector if enabled and y is provided
            if self.use_shap_selection and y is not None:
                # First transform with feature union
                X_features = self.pipeline_.named_steps["features"].fit_transform(X, y)
                # Then fit SHAP selector
                self.pipeline_.named_steps["shap_selection"].fit(
                    pd.DataFrame(
                        X_features,
                        columns=self.pipeline_.named_steps["features"].get_feature_names_out(),
                    ),
                    y,
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X using fitted pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Input transaction data

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix
        """
        if not hasattr(self, "pipeline_"):
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X_transformed = self.pipeline_.transform(X)

        # Convert to DataFrame with feature names
        feature_names = self._get_feature_names()
        return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input transaction data
        y : pd.Series, optional
            Fraud labels

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix
        """
        self.fit(X, y)
        return self.transform(X)

    def _get_feature_names(self) -> list:
        """Get feature names from pipeline."""
        if not hasattr(self, "pipeline_"):
            return []

        # Get feature names from FeatureUnion
        feature_union = self.pipeline_.named_steps["features"]

        # Get names from each transformer
        velocity_names = feature_union.transformer_list[0][1].get_feature_names_out()
        deviation_names = feature_union.transformer_list[1][1].get_feature_names_out()

        feature_names = velocity_names + deviation_names

        # If SHAP selection was used, get selected features
        if "shap_selection" in self.pipeline_.named_steps:
            shap_selector = self.pipeline_.named_steps["shap_selection"]
            feature_names = shap_selector.get_feature_names_out()

        return feature_names

    def save(self, filepath: str):
        """
        Save pipeline to disk using joblib.

        Parameters
        ----------
        filepath : str
            Path to save the pipeline
        """
        if not hasattr(self, "pipeline_"):
            raise ValueError("No fitted pipeline to save")

        joblib.dump(self, filepath)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FraudFeaturePipeline":
        """
        Load pipeline from disk.

        Parameters
        ----------
        filepath : str
            Path to load the pipeline from

        Returns
        -------
        FraudFeaturePipeline
            Loaded pipeline
        """
        pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
        return pipeline

    def get_feature_names_out(self) -> list:
        """
        Get feature names from the fitted pipeline.

        Returns
        -------
        list
            Feature names
        """
        return self._get_feature_names()
