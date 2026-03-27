"""Unit tests for VelocityFeatures transformer."""

import pytest
import pandas as pd
import numpy as np
from src.transformers.velocity_features import VelocityFeatures


class TestVelocityFeatures:
    """Test suite for VelocityFeatures transformer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        np.random.seed(42)
        n = 100
        data = {
            "user_id": np.random.randint(1, 11, n),  # 10 users
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "amount": np.random.uniform(10, 500, n),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer(self):
        """Create VelocityFeatures instance."""
        return VelocityFeatures(
            user_col="user_id",
            datetime_col="timestamp",
            amount_col="amount",
            time_windows=[(1, "h"), (24, "h")],
            features=["count", "sum", "mean"],
        )

    def test_fit_returns_self(self, transformer, sample_data):
        """Test that fit returns self."""
        result = transformer.fit(sample_data)
        assert result is transformer

    def test_fit_stores_columns(self, transformer, sample_data):
        """Test that fit stores column names."""
        transformer.fit(sample_data)
        assert hasattr(transformer, "feature_names_in_")
        assert hasattr(transformer, "feature_names_out_")

    def test_transform_returns_dataframe(self, transformer, sample_data):
        """Test that transform returns DataFrame."""
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_output_shape(self, transformer, sample_data):
        """Test that transform has correct output shape."""
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        # 2 windows * 3 features = 6, plus time_since_last = 7
        expected_cols = 7
        assert result.shape[1] == expected_cols

    def test_transform_feature_names(self, transformer, sample_data):
        """Test that feature names are generated correctly."""
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        expected_names = [
            "velocity_count_1h",
            "velocity_sum_1h",
            "velocity_mean_1h",
            "velocity_count_24h",
            "velocity_sum_24h",
            "velocity_mean_24h",
            "velocity_time_since_last_s",
        ]
        assert list(result.columns) == expected_names

    def test_missing_columns_raises_error(self, transformer, sample_data):
        """Test that missing required columns raises ValueError."""
        bad_data = sample_data.drop(columns=["amount"])
        with pytest.raises(ValueError, match="Missing required columns"):
            transformer.fit(bad_data)

    def test_unseen_users_handled(self, transformer, sample_data):
        """Test that unseen users are handled gracefully."""
        transformer.fit(sample_data)
        # Create data with new user
        new_data = pd.DataFrame({
            "user_id": [999],  # Unseen user
            "timestamp": [pd.Timestamp("2024-01-02")],
            "amount": [100.0],
        })
        result = transformer.transform(new_data)
        assert not result.isna().any().any()

    def test_time_since_last_first_transaction(self, transformer, sample_data):
        """Test that first transaction has large time_since_last."""
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        # First transaction for each user should have large value
        first_transactions = sample_data.groupby("user_id").head(1).index
        assert all(result.loc[first_transactions, "velocity_time_since_last_s"] > 100000)


class TestVelocityFeaturesEdgeCases:
    """Test edge cases for VelocityFeatures."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        transformer = VelocityFeatures()
        empty_df = pd.DataFrame(columns=["user_id", "timestamp", "amount"])
        transformer.fit(empty_df)
        result = transformer.transform(empty_df)
        assert len(result) == 0

    def test_single_transaction(self):
        """Test with single transaction."""
        data = pd.DataFrame({
            "user_id": [1],
            "timestamp": [pd.Timestamp("2024-01-01")],
            "amount": [100.0],
        })
        transformer = VelocityFeatures(time_windows=[(1, "h")])
        transformer.fit(data)
        result = transformer.transform(data)
        assert result.shape == (1, 2)  # count and time_since_last

    def test_invalid_datetime_format(self):
        """Test handling of invalid datetime format."""
        data = pd.DataFrame({
            "user_id": [1, 1],
            "timestamp": ["invalid", "2024-01-01"],
            "amount": [100.0, 200.0],
        })
        transformer = VelocityFeatures(time_windows=[(1, "h")])
        transformer.fit(data)
        result = transformer.transform(data)
        # Should handle invalid datetime by coercing to NaT
        assert isinstance(result, pd.DataFrame)

    def test_zero_amounts(self):
        """Test with zero transaction amounts."""
        data = pd.DataFrame({
            "user_id": [1, 1, 1],
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "amount": [0.0, 0.0, 0.0],
        })
        transformer = VelocityFeatures(time_windows=[(1, "h")])
        transformer.fit(data)
        result = transformer.transform(data)
        assert not result.isna().any().any()
