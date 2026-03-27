"""SHAP-based feature selector for fraud detection."""

from typing import Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on SHAP values from a trained model.

    Uses TreeExplainer for tree-based models to compute SHAP values,
    then selects the top N features based on mean absolute SHAP values.

    Parameters
    ----------
    estimator : object, optional
        Scikit-learn estimator with feature_importances_ or supports SHAP
        Default: RandomForestClassifier(n_estimators=50)
    n_features : int, optional
        Number of top features to select
        Default: 20
    threshold : float, optional
        Minimum mean absolute SHAP value threshold
        If specified, n_features is ignored and features above threshold are selected
        Default: None
    random_state : int, optional
        Random state for estimator

    Attributes
    ----------
    estimator_ : object
        Fitted estimator
    shap_values_ : np.ndarray
        SHAP values from training data
    feature_importance_ : pd.DataFrame
        DataFrame with feature names and importance scores
    selected_features_ : list
        List of selected feature names
    feature_names_in_ : list
        Input feature names

    Examples
    --------
    >>> selector = SHAPSelector(n_features=15)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(selector.selected_features_)
    """

    def __init__(
        self,
        estimator: Optional[object] = None,
        n_features: int = 20,
        threshold: Optional[float] = None,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.threshold = threshold
        self.random_state = random_state

    def _get_default_estimator(self):
        """Get default estimator if none provided."""
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SHAPSelector":
        """
        Fit the selector by training estimator and computing SHAP values.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix
        y : pd.Series
            Target labels

        Returns
        -------
        self : SHAPSelector
            Fitted selector
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )

        # Initialize estimator
        if self.estimator is None:
            self.estimator_ = self._get_default_estimator()
        else:
            self.estimator_ = self.estimator

        # Set random state if possible
        if hasattr(self.estimator_, "set_params"):
            try:
                self.estimator_.set_params(random_state=self.random_state)
            except ValueError:
                pass  # Estimator doesn't support random_state

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Fit estimator
        self.estimator_.fit(X, y)

        # Compute SHAP values
        try:
            explainer = shap.TreeExplainer(self.estimator_)
            self.shap_values_ = explainer.shap_values(X)

            # Handle binary classification (returns list of two arrays)
            if isinstance(self.shap_values_, list):
                self.shap_values_ = self.shap_values_[1]  # Use positive class

        except Exception as e:
            # Fallback to permutation importance if TreeExplainer fails
            print(
                f"TreeExplainer failed: {e}. Using feature_importances_ as fallback."
            )
            self.shap_values_ = None

        # Compute feature importance
        if self.shap_values_ is not None:
            importance = np.abs(self.shap_values_).mean(axis=0)
        else:
            # Fallback to model's feature_importances_
            importance = self.estimator_.feature_importances_

        self.feature_importance_ = pd.DataFrame(
            {"feature": self.feature_names_in_, "importance": importance}
        ).sort_values("importance", ascending=False)

        # Select features
        if self.threshold is not None:
            self.selected_features_ = self.feature_importance_[
                self.feature_importance_["importance"] >= self.threshold
            ]["feature"].tolist()
        else:
            self.selected_features_ = self.feature_importance_.head(self.n_features)[
                "feature"
            ].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X by selecting features.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix

        Returns
        -------
        pd.DataFrame
            DataFrame with selected features only
        """
        if not hasattr(self, "selected_features_"):
            raise ValueError("Selector not fitted. Call fit() first.")

        # Ensure all selected features are present
        missing = [f for f in self.selected_features_ if f not in X.columns]
        if missing:
            raise ValueError(f"Missing features in X: {missing}")

        return X[self.selected_features_].copy()

    def get_feature_names_out(self) -> list:
        """
        Get names of selected features.

        Returns
        -------
        list
            Selected feature names
        """
        if hasattr(self, "selected_features_"):
            return self.selected_features_
        return []

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns
        -------
        pd.DataFrame
            DataFrame with features sorted by importance
        """
        if not hasattr(self, "feature_importance_"):
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.feature_importance_.copy()
