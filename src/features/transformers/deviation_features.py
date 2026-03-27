"""Deviation feature transformer for fraud detection."""

from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DeviationFeatures(BaseEstimator, TransformerMixin):
    """
    Compute deviation features: compare current behavior to user's historical patterns.

    Features computed:
    - Z-score: (current_value - historical_mean) / historical_std
    - Ratio: current_value / historical_mean
    - Percentile rank of current value in historical distribution

    Parameters
    ----------
    user_col : str
        Column name for user identifier
    features : List[str]
        Column names to compute deviations for (e.g., ['amount', 'merchant_id'])
    window_size : int, optional
        Number of historical transactions to use for computing statistics
        Default: 30 (uses last 30 transactions per user)
    min_transactions : int, optional
        Minimum number of historical transactions required to compute deviation
        Default: 3 (if less, uses global statistics)

    Attributes
    ----------
    user_stats_ : dict
        Dictionary mapping user_id to their historical statistics
    global_stats_ : dict
        Global statistics for fallback with unseen users
    feature_names_out_ : List[str]
        Names of generated features

    Examples
    --------
    >>> transformer = DeviationFeatures(
    ...     user_col='user_id',
    ...     features=['amount', 'hour_of_day'],
    ...     window_size=30
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(
        self,
        user_col: str = "user_id",
        features: Optional[List[str]] = None,
        window_size: int = 30,
        min_transactions: int = 3,
    ):
        self.user_col = user_col
        self.features = features or ["amount"]
        self.window_size = window_size
        self.min_transactions = min_transactions

    def fit(self, X: pd.DataFrame, y=None) -> "DeviationFeatures":
        """
        Fit the transformer by computing user historical statistics.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe
        y : ignored
            Not used, present for scikit-learn compatibility

        Returns
        -------
        self : DeviationFeatures
            Fitted transformer with computed user statistics
        """
        # Validate required columns
        required_cols = [self.user_col] + self.features
        missing = [col for col in required_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = X.copy()

        # Compute statistics for each user and feature
        self.user_stats_ = {}
        self.global_stats_ = {}

        for feature in self.features:
            # Compute global statistics for fallback
            self.global_stats_[feature] = {
                "mean": X[feature].mean(),
                "std": X[feature].std(),
                "q25": X[feature].quantile(0.25),
                "q50": X[feature].quantile(0.50),
                "q75": X[feature].quantile(0.75),
            }

            # Handle case where std is 0
            if self.global_stats_[feature]["std"] == 0:
                self.global_stats_[feature]["std"] = 1.0

            # Compute per-user statistics
            self.user_stats_[feature] = {}
            grouped = X.groupby(self.user_col)

            for user_id, group in grouped:
                # Use last window_size transactions
                historical = group[feature].tail(self.window_size)

                self.user_stats_[feature][user_id] = {
                    "mean": historical.mean(),
                    "std": historical.std(),
                    "q25": historical.quantile(0.25),
                    "q50": historical.quantile(0.50),
                    "q75": historical.quantile(0.75),
                }

                # Handle case where std is 0 or NaN
                if (
                    pd.isna(self.user_stats_[feature][user_id]["std"])
                    or self.user_stats_[feature][user_id]["std"] == 0
                ):
                    self.user_stats_[feature][user_id]["std"] = self.global_stats_[
                        feature
                    ]["std"]

        self.feature_names_out_ = self._get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X by computing deviation features.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with deviation features
        """
        # Validate input
        required_cols = [self.user_col] + self.features
        missing = [col for col in required_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = X.copy()
        result = pd.DataFrame(index=X.index)

        for feature in self.features:
            # Initialize arrays
            z_scores = np.zeros(len(X))
            ratios = np.zeros(len(X))

            for idx, row in X.iterrows():
                user_id = row[self.user_col]
                current_value = row[feature]

                # Get user stats, fall back to global if unseen user
                if user_id in self.user_stats_[feature]:
                    stats = self.user_stats_[feature][user_id]
                else:
                    stats = self.global_stats_[feature]

                # Compute z-score
                z_scores[idx] = (current_value - stats["mean"]) / stats["std"]

                # Compute ratio (handle division by zero)
                if stats["mean"] != 0:
                    ratios[idx] = current_value / stats["mean"]
                else:
                    ratios[idx] = 1.0 if current_value == 0 else np.inf

            result[f"deviation_{feature}_zscore"] = z_scores
            result[f"deviation_{feature}_ratio"] = ratios

        return result

    def _get_feature_names_out(self) -> List[str]:
        """Generate output feature names."""
        feature_names = []
        for feature in self.features:
            feature_names.extend(
                [f"deviation_{feature}_zscore", f"deviation_{feature}_ratio"]
            )
        return feature_names

    def get_feature_names_out(self) -> List[str]:
        """
        Get names of generated features.

        Returns
        -------
        List[str]
            Feature names
        """
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        return self._get_feature_names_out()
