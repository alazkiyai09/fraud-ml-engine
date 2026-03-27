"""Unit tests for MerchantRiskFeatures transformer."""

import pytest
import pandas as pd
import numpy as np
from src.transformers.merchant_features import MerchantRiskFeatures


class TestMerchantRiskFeatures:
    """Test suite for MerchantRiskFeatures transformer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        np.random.seed(42)
        n = 100
        data = {
            "merchant_id": np.random.randint(1, 11, n),  # 10 merchants
            "amount": np.random.uniform(10, 500, n),
        }
        df = pd.DataFrame(data)
        # Create fraud labels (20% fraud rate)
        df["fraud"] = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        return df

    @pytest.fixture
    def transformer(self):
        """Create MerchantRiskFeatures instance."""
        return MerchantRiskFeatures(
            merchant_col="merchant_id",
            alpha=1.0,
            beta=1.0,
        )

    def test_fit_returns_self(self, transformer, sample_data):
        """Test that fit returns self."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        result = transformer.fit(X, y)
        assert result is transformer

    def test_fit_requires_y(self, transformer, sample_data):
        """Test that fit requires y parameter."""
        X = sample_data.drop(columns=["fraud"])
        with pytest.raises(ValueError, match="y.*required"):
            transformer.fit(X)

    def test_fit_computes_statistics(self, transformer, sample_data):
        """Test that fit computes merchant statistics."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        transformer.fit(X, y)
        assert hasattr(transformer, "merchant_stats_")
        assert hasattr(transformer, "global_fraud_rate_")
        assert 0 <= transformer.global_fraud_rate_ <= 1

    def test_transform_returns_dataframe(self, transformer, sample_data):
        """Test that transform returns DataFrame."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        transformer.fit(X, y)
        result = transformer.transform(X)
        assert isinstance(result, pd.DataFrame)

    def test_transform_output_shape(self, transformer, sample_data):
        """Test that transform has correct output shape (3 columns)."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        transformer.fit(X, y)
        result = transformer.transform(X)
        assert result.shape[1] == 3

    def test_transform_feature_names(self, transformer, sample_data):
        """Test that feature names are generated correctly."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        transformer.fit(X, y)
        result = transformer.transform(X)
        expected_names = [
            "merchant_fraud_rate",
            "merchant_fraud_count",
            "merchant_total_count",
        ]
        assert list(result.columns) == expected_names

    def test_bayesian_smoothing(self, sample_data):
        """Test that Bayesian smoothing is applied."""
        # Create data with merchant that has 1/1 fraud (100% raw rate)
        data = pd.DataFrame({
            "merchant_id": [1] * 10 + [2] * 10,
            "amount": [100.0] * 20,
        })
        # Merchant 1: 1 fraud out of 1 (small sample)
        # Merchant 2: 0 fraud out of 10
        fraud_labels = [1] + [0] * 9 + [0] * 10

        transformer = MerchantRiskFeatures(alpha=1.0, beta=1.0)
        X = data.drop(columns=["amount"])
        transformer.fit(X, pd.Series(fraud_labels))
        result = transformer.transform(X)

        # Merchant 1's smoothed rate should be pulled toward global rate
        merchant_1_rate = result[result["merchant_fraud_count"] == 1]["merchant_fraud_rate"].iloc[0]
        raw_rate_1 = 1.0  # 1/1
        global_rate = transformer.global_fraud_rate_

        # Smoothed rate should be between raw and global
        assert global_rate <= merchant_1_rate <= raw_rate_1 or raw_rate_1 <= merchant_1_rate <= global_rate

    def test_unseen_merchants_handled(self, transformer, sample_data):
        """Test that unseen merchants fall back to global statistics."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        transformer.fit(X, y)

        # Create data with new merchant
        new_data = pd.DataFrame({
            "merchant_id": [999],  # Unseen merchant
            "amount": [100.0],
        })
        result = transformer.transform(new_data)

        # Should use global rate
        assert result["merchant_fraud_rate"].iloc[0] == transformer.global_fraud_rate_
        assert result["merchant_fraud_count"].iloc[0] == 0
        assert result["merchant_total_count"].iloc[0] == 0

    def test_missing_columns_raises_error(self, transformer, sample_data):
        """Test that missing required columns raises ValueError."""
        bad_data = sample_data.drop(columns=["merchant_id"])
        y = sample_data["fraud"]
        with pytest.raises(ValueError, match="Missing required column"):
            transformer.fit(bad_data, y)


class TestMerchantRiskFeaturesEdgeCases:
    """Test edge cases for MerchantRiskFeatures."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        transformer = MerchantRiskFeatures()
        empty_X = pd.DataFrame(columns=["merchant_id"])
        empty_y = pd.Series([], dtype=int)
        transformer.fit(empty_X, empty_y)
        result = transformer.transform(empty_X)
        assert len(result) == 0

    def test_single_transaction(self):
        """Test with single transaction."""
        data = pd.DataFrame({
            "merchant_id": [1],
            "amount": [100.0],
        })
        transformer = MerchantRiskFeatures(alpha=1.0, beta=1.0)
        transformer.fit(data.drop(columns=["amount"]), pd.Series([0]))
        result = transformer.transform(data.drop(columns=["amount"]))
        assert result.shape == (1, 3)

    def test_all_fraud_or_all_legitimate(self):
        """Test when all transactions are fraud or all are legitimate."""
        # All legitimate
        data = pd.DataFrame({
            "merchant_id": [1] * 10,
            "amount": [100.0] * 10,
        })
        transformer = MerchantRiskFeatures(alpha=1.0, beta=1.0)
        transformer.fit(data.drop(columns=["amount"]), pd.Series([0] * 10))
        result = transformer.transform(data.drop(columns=["amount"]))

        # Rate should be smoothed toward 0
        assert all(result["merchant_fraud_rate"] >= 0)
        assert all(result["merchant_fraud_rate"] <= 1.0)

    def test_different_alpha_beta(self):
        """Test with different alpha and beta values."""
        data = pd.DataFrame({
            "merchant_id": [1] * 5,
            "amount": [100.0] * 5,
        })
        y = pd.Series([1, 0, 0, 0, 0])  # 1 fraud out of 5

        # Strong prior toward legitimate (high beta)
        transformer1 = MerchantRiskFeatures(alpha=1.0, beta=10.0)
        transformer1.fit(data.drop(columns=["amount"]), y)
        result1 = transformer1.transform(data.drop(columns=["amount"]))

        # Strong prior toward fraud (high alpha)
        transformer2 = MerchantRiskFeatures(alpha=10.0, beta=1.0)
        transformer2.fit(data.drop(columns=["amount"]), y)
        result2 = transformer2.transform(data.drop(columns=["amount"]))

        # transformer2 should have higher fraud rate due to higher alpha
        assert (
            result2["merchant_fraud_rate"].iloc[0]
            > result1["merchant_fraud_rate"].iloc[0]
        )
