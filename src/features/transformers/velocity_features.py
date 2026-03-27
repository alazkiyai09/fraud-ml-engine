"""Velocity feature transformer for fraud detection."""

from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class VelocityFeatures(BaseEstimator, TransformerMixin):
    """
    Compute velocity features: transaction frequency and amounts over time windows.

    Features computed:
    - Transaction count in each time window
    - Total amount in each time window
    - Average amount in each time window
    - Time since last transaction

    Parameters
    ----------
    user_col : str
        Column name for user identifier
    datetime_col : str
        Column name for transaction timestamp
    amount_col : str
        Column name for transaction amount
    time_windows : List[tuple]
        List of (window_size, unit) tuples, e.g., [(1, 'h'), (24, 'h'), (7, 'd')]
        Supported units: 's' (seconds), 'min' (minutes), 'h' (hours), 'd' (days)
    features : List[str], optional
        Features to compute: 'count', 'sum', 'mean', 'time_since_last'
        Default: ['count', 'sum', 'mean', 'time_since_last']

    Attributes
    ----------
    feature_names_in_ : List[str]
        Names of features seen during fit
    feature_names_out_ : List[str]
        Names of generated features

    Examples
    --------
    >>> transformer = VelocityFeatures(
    ...     user_col='user_id',
    ...     datetime_col='timestamp',
    ...     amount_col='amount',
    ...     time_windows=[(1, 'h'), (24, 'h'), (7, 'd')]
    ... )
    >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(
        self,
        user_col: str = "user_id",
        datetime_col: str = "timestamp",
        amount_col: str = "amount",
        time_windows: List[tuple] = [(1, "h"), (24, "h"), (7, "d")],
        features: Optional[List[str]] = None,
    ):
        self.user_col = user_col
        self.datetime_col = datetime_col
        self.amount_col = amount_col
        self.time_windows = time_windows
        self.features = features or ["count", "sum", "mean", "time_since_last"]

    def fit(self, X: pd.DataFrame, y=None) -> "VelocityFeatures":
        """
        Fit the transformer (stores column names).

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with columns: user_col, datetime_col, amount_col
        y : ignored
            Not used, present for scikit-learn compatibility

        Returns
        -------
        self : VelocityFeatures
            Fitted transformer
        """
        # Validate required columns
        required_cols = [self.user_col, self.datetime_col, self.amount_col]
        missing = [col for col in required_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = self._get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X by computing velocity features.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with original columns plus velocity features
        """
        # Validate input
        missing = [col for col in [self.user_col, self.datetime_col, self.amount_col] if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Make a copy to avoid modifying original
        X = X.copy()

        # Ensure datetime column is properly typed
        if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_col]):
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors="coerce")

        # Sort by user and timestamp
        X = X.sort_values([self.user_col, self.datetime_col])

        # Initialize result dataframe
        result = pd.DataFrame(index=X.index)

        # Compute velocity features for each time window
        for window_size, unit in self.time_windows:
            window_str = f"{window_size}{unit}"

            # Group by user and compute rolling statistics
            grouped = X.groupby(self.user_col, group_keys=False)

            if "count" in self.features:
                # Transaction count in window
                rolling_count = grouped[self.amount_col].rolling(
                    window=window_str, min_periods=1, on=self.datetime_col
                ).count()
                result[f"velocity_count_{window_str}"] = rolling_count.values

            if "sum" in self.features:
                # Total amount in window
                rolling_sum = grouped[self.amount_col].rolling(
                    window=window_str, min_periods=1, on=self.datetime_col
                ).sum()
                result[f"velocity_sum_{window_str}"] = rolling_sum.values

            if "mean" in self.features:
                # Average amount in window
                rolling_mean = grouped[self.amount_col].rolling(
                    window=window_str, min_periods=1, on=self.datetime_col
                ).mean()
                result[f"velocity_mean_{window_str}"] = rolling_mean.values

            if "std" in self.features:
                # Std amount in window
                rolling_std = grouped[self.amount_col].rolling(
                    window=window_str, min_periods=1, on=self.datetime_col
                ).std()
                result[f"velocity_std_{window_str}"] = rolling_std.values

        # Compute time since last transaction
        if "time_since_last" in self.features:
            X["prev_timestamp"] = X.groupby(self.user_col)[self.datetime_col].shift(1)
            result["velocity_time_since_last_s"] = (
                X[self.datetime_col] - X["prev_timestamp"]
            ).dt.total_seconds()
            # For first transaction, use a large value (no previous transaction)
            result["velocity_time_since_last_s"] = result["velocity_time_since_last_s"].fillna(999999)

        return result

    def _get_feature_names_out(self) -> List[str]:
        """Generate output feature names."""
        feature_names = []

        for window_size, unit in self.time_windows:
            window_str = f"{window_size}{unit}"
            if "count" in self.features:
                feature_names.append(f"velocity_count_{window_str}")
            if "sum" in self.features:
                feature_names.append(f"velocity_sum_{window_str}")
            if "mean" in self.features:
                feature_names.append(f"velocity_mean_{window_str}")
            if "std" in self.features:
                feature_names.append(f"velocity_std_{window_str}")

        if "time_since_last" in self.features:
            feature_names.append("velocity_time_since_last_s")

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
