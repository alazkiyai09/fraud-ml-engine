"""Unit tests for DeviationFeatures transformer."""

import pytest
import pandas as pd
import numpy as np
from src.transformers.deviation_features import DeviationFeatures


class TestDeviationFeatures:
    """Test suite for DeviationFeatures transformer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        np.random.seed(42)
        n = 100
        data = {
            "user_id": np.random.randint(1, 11, n),  # 10 users
            "amount": np.random.uniform(10, 500, n),
            "hour_of_day": np.random.randint(0, 24, n),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer(self):
        """Create DeviationFeatures instance."""
        return DeviationFeatures(
            user_col="user_id",
            features=["amount", "hour_of_day"],
            window_size=30,
        )

    def test_fit_returns_self(self, transformer, sample_data):
        """Test that fit returns self."""
        result = transformer.fit(sample_data)
        assert result is transformer

    def test_fit_computes_statistics(self, transformer, sample_data):
        """Test that fit computes user statistics."""
        transformer.fit(sample_data)
        assert hasattr(transformer, "user_stats_")
        assert hasattr(transformer, "global_stats_")
        assert "amount" in transformer.user_stats_
        assert "hour_of_day" in transformer.user_stats_

    def test_transform_returns_dataframe(self, transformer, sample_data):
        """Test that transform returns DataFrame."""
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_transform_output_shape(self, transformer, sample_data):
        """Test that transform has correct output shape."""
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        # 2 features * 2 metrics (zscore, ratio) = 4 columns
        assert result.shape[1] == 4

    def test_transform_feature_names(self, transformer, sample_data):
        """Test that feature names are generated correctly."""
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        expected_names = [
            "deviation_amount_zscore",
            "deviation_amount_ratio",
            "deviation_hour_of_day_zscore",
            "deviation_hour_of_day_ratio",
        ]
        assert list(result.columns) == expected_names

    def test_missing_columns_raises_error(self, transformer, sample_data):
        """Test that missing required columns raises ValueError."""
        bad_data = sample_data.drop(columns=["amount"])
        with pytest.raises(ValueError, match="Missing required columns"):
            transformer.fit(bad_data)

    def test_unseen_users_handled(self, transformer, sample_data):
        """Test that unseen users fall back to global statistics."""
        transformer.fit(sample_data)
        # Create data with new user
        new_data = pd.DataFrame({
            "user_id": [999],  # Unseen user
            "amount": [100.0],
            "hour_of_day": [12],
        })
        result = transformer.transform(new_data)
        assert not result.isna().any().any()

    def test_zscore_calculation(self, sample_data):
        """Test that z-score is calculated correctly."""
        # Create data with known statistics
        data = pd.DataFrame({
            "user_id": [1] * 50,
            "amount": [100.0] * 50,  # All same amount
        })
        transformer = DeviationFeatures(user_col="user_id", features=["amount"])
        transformer.fit(data)

        # Test with same value (should have z-score near 0)
        test_data = pd.DataFrame({
            "user_id": [1],
            "amount": [100.0],
        })
        result = transformer.transform(test_data)
        assert abs(result["deviation_amount_zscore"].iloc[0]) < 1e-10

    def test_ratio_calculation(self, sample_data):
        """Test that ratio is calculated correctly."""
        data = pd.DataFrame({
            "user_id": [1] * 10,
            "amount": [100.0] * 10,  # Mean is 100
        })
        transformer = DeviationFeatures(user_col="user_id", features=["amount"])
        transformer.fit(data)

        test_data = pd.DataFrame({
            "user_id": [1],
            "amount": [200.0],  # 2x the mean
        })
        result = transformer.transform(test_data)
        assert abs(result["deviation_amount_ratio"].iloc[0] - 2.0) < 1e-5


class TestDeviationFeaturesEdgeCases:
    """Test edge cases for DeviationFeatures."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        transformer = DeviationFeatures(features=["amount"])
        empty_df = pd.DataFrame(columns=["user_id", "amount"])
        transformer.fit(empty_df)
        result = transformer.transform(empty_df)
        assert len(result) == 0

    def test_single_user_single_transaction(self):
        """Test with single transaction for single user."""
        data = pd.DataFrame({
            "user_id": [1],
            "amount": [100.0],
        })
        transformer = DeviationFeatures(features=["amount"])
        transformer.fit(data)
        result = transformer.transform(data)
        # Should use global stats (which is just this one value)
        assert not result.isna().any().any()

    def test_zero_variance_feature(self):
        """Test feature with zero variance."""
        data = pd.DataFrame({
            "user_id": [1, 1, 1],
            "amount": [100.0, 100.0, 100.0],  # No variance
        })
        transformer = DeviationFeatures(features=["amount"])
        transformer.fit(data)
        result = transformer.transform(data)
        # Should handle zero std by setting it to 1.0 (from global fallback)
        assert not result.isna().any().any()

    def test_nan_values_in_features(self):
        """Test handling of NaN values in features."""
        data = pd.DataFrame({
            "user_id": [1, 1, 1],
            "amount": [100.0, np.nan, 200.0],
        })
        transformer = DeviationFeatures(features=["amount"])
        transformer.fit(data)
        result = transformer.transform(data)
        # NaN should propagate through calculations
        assert result["deviation_amount_zscore"].iloc[1] != result["deviation_amount_zscore"].iloc[1]  # NaN check

    def test_zero_mean_handling(self):
        """Test ratio calculation when mean is zero."""
        data = pd.DataFrame({
            "user_id": [1, 1, 1],
            "amount": [0.0, 0.0, 0.0],  # All zeros
        })
        transformer = DeviationFeatures(features=["amount"])
        transformer.fit(data)
        result = transformer.transform(data)
        # Ratio should be 1.0 when both current and mean are 0
        assert result["deviation_amount_ratio"].iloc[0] == 1.0
