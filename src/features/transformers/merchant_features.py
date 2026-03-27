"""Merchant risk feature transformer for fraud detection."""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MerchantRiskFeatures(BaseEstimator, TransformerMixin):
    """
    Compute merchant-level fraud risk features using Bayesian smoothing.

    Features computed:
    - Fraud rate: (fraud_count + alpha) / (total_count + alpha + beta)
    - Fraud count: total number of fraud transactions at merchant
    - Total count: total number of transactions at merchant

    Bayesian smoothing prevents overfitting for merchants with few transactions
    by shrinking their rates toward the global average.

    Parameters
    ----------
    merchant_col : str
        Column name for merchant identifier
    alpha : float, optional
        Beta distribution prior parameter (pseudo-fraud counts)
        Default: 1.0 (uniform prior)
    beta : float, optional
        Beta distribution prior parameter (pseudo-legitimate counts)
        Default: 1.0 (uniform prior)
    global_rate_weight : float, optional
        Weight for global fraud rate in smoothing (0-1)
        Default: 0.5

    Attributes
    ----------
    merchant_fraud_rate_ : dict
        Dictionary mapping merchant_id to smoothed fraud rate
    global_fraud_rate_ : float
        Overall fraud rate in training data
    feature_names_out_ : List[str]
        Names of generated features

    Examples
    --------
    >>> transformer = MerchantRiskFeatures(
    ...     merchant_col='merchant_id',
    ...     alpha=1.0,
    ...     beta=1.0
    ... )
    >>> X_transformed = transformer.fit_transform(X, y=fraud_labels)
    """

    def __init__(
        self,
        merchant_col: str = "merchant_id",
        alpha: float = 1.0,
        beta: float = 1.0,
        global_rate_weight: float = 0.5,
    ):
        self.merchant_col = merchant_col
        self.alpha = alpha
        self.beta = beta
        self.global_rate_weight = global_rate_weight

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "MerchantRiskFeatures":
        """
        Fit the transformer by computing merchant fraud statistics.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with merchant_col
        y : pd.Series
            Binary fraud labels (1=fraud, 0=legitimate)

        Returns
        -------
        self : MerchantRiskFeatures
            Fitted transformer with merchant statistics
        """
        if y is None:
            raise ValueError("y (fraud labels) is required for fitting MerchantRiskFeatures")

        # Validate required columns
        if self.merchant_col not in X.columns:
            raise ValueError(f"Missing required column: {self.merchant_col}")

        X = X.copy()
        X["fraud"] = y.values

        # Compute global fraud rate
        self.global_fraud_rate_ = y.mean()

        # Compute per-merchant statistics
        merchant_stats = (
            X.groupby(self.merchant_col)["fraud"]
            .agg(["sum", "count"])
            .reset_index()
        )
        merchant_stats.columns = [self.merchant_col, "fraud_count", "total_count"]

        # Apply Bayesian smoothing
        # smoothed_rate = (fraud_count + alpha) / (total_count + alpha + beta)
        merchant_stats["raw_rate"] = (
            merchant_stats["fraud_count"] / merchant_stats["total_count"]
        )
        merchant_stats["smoothed_rate"] = (
            merchant_stats["fraud_count"] + self.alpha
        ) / (merchant_stats["total_count"] + self.alpha + self.beta)

        # Blend with global rate for very low-volume merchants
        merchant_stats["final_rate"] = (
            self.global_rate_weight * self.global_fraud_rate_
            + (1 - self.global_rate_weight) * merchant_stats["smoothed_rate"]
        )

        # Store merchant statistics as dictionary for fast lookup
        self.merchant_stats_ = {
            row[self.merchant_col]: {
                "fraud_count": row["fraud_count"],
                "total_count": row["total_count"],
                "rate": row["final_rate"],
            }
            for _, row in merchant_stats.iterrows()
        }

        self.feature_names_out_ = self._get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X by computing merchant risk features.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with merchant risk features
        """
        # Validate input
        if self.merchant_col not in X.columns:
            raise ValueError(f"Missing required column: {self.merchant_col}")

        X = X.copy()
        result = pd.DataFrame(index=X.index)

        # Get merchant statistics
        fraud_counts = []
        total_counts = []
        fraud_rates = []

        for merchant_id in X[self.merchant_col]:
            if merchant_id in self.merchant_stats_:
                stats = self.merchant_stats_[merchant_id]
            else:
                # Handle unseen merchants: use global statistics
                stats = {
                    "fraud_count": 0,
                    "total_count": 0,
                    "rate": self.global_fraud_rate_,
                }

            fraud_counts.append(stats["fraud_count"])
            total_counts.append(stats["total_count"])
            fraud_rates.append(stats["rate"])

        result[f"merchant_fraud_rate"] = fraud_rates
        result[f"merchant_fraud_count"] = fraud_counts
        result[f"merchant_total_count"] = total_counts

        return result

    def _get_feature_names_out(self) -> list[str]:
        """Generate output feature names."""
        return [
            "merchant_fraud_rate",
            "merchant_fraud_count",
            "merchant_total_count",
        ]

    def get_feature_names_out(self) -> list[str]:
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
