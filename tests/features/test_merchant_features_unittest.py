"""Unittest version of tests for MerchantRiskFeatures transformer."""

import unittest
import pandas as pd
import numpy as np
from src.transformers.merchant_features import MerchantRiskFeatures


class TestMerchantRiskFeatures(unittest.TestCase):
    """Test suite for MerchantRiskFeatures transformer."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n = 100
        data = {
            "merchant_id": np.random.randint(1, 11, n),
            "amount": np.random.uniform(10, 500, n),
        }
        df = pd.DataFrame(data)
        df["fraud"] = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        self.sample_data = df

        self.transformer = MerchantRiskFeatures(
            merchant_col="merchant_id",
            alpha=1.0,
            beta=1.0,
        )

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        result = self.transformer.fit(X, y)
        self.assertIs(result, self.transformer)

    def test_fit_requires_y(self):
        """Test that fit requires y parameter."""
        X = self.sample_data.drop(columns=["fraud"])
        with self.assertRaises(ValueError) as context:
            self.transformer.fit(X)
        self.assertIn("y.*required", str(context.exception))

    def test_fit_computes_statistics(self):
        """Test that fit computes merchant statistics."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.transformer.fit(X, y)
        self.assertTrue(hasattr(self.transformer, "merchant_stats_"))
        self.assertTrue(hasattr(self.transformer, "global_fraud_rate_"))
        self.assertGreaterEqual(self.transformer.global_fraud_rate_, 0)
        self.assertLessEqual(self.transformer.global_fraud_rate_, 1)

    def test_transform_returns_dataframe(self):
        """Test that transform returns DataFrame."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.transformer.fit(X, y)
        result = self.transformer.transform(X)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_output_shape(self):
        """Test that transform has correct output shape (3 columns)."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.transformer.fit(X, y)
        result = self.transformer.transform(X)
        self.assertEqual(result.shape[1], 3)

    def test_transform_feature_names(self):
        """Test that feature names are generated correctly."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.transformer.fit(X, y)
        result = self.transformer.transform(X)
        expected_names = [
            "merchant_fraud_rate",
            "merchant_fraud_count",
            "merchant_total_count",
        ]
        self.assertListEqual(list(result.columns), expected_names)

    def test_unseen_merchants_handled(self):
        """Test that unseen merchants fall back to global statistics."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.transformer.fit(X, y)

        new_data = pd.DataFrame({
            "merchant_id": [999],
            "amount": [100.0],
        })
        result = self.transformer.transform(new_data)

        self.assertEqual(
            result["merchant_fraud_rate"].iloc[0],
            self.transformer.global_fraud_rate_
        )
        self.assertEqual(result["merchant_fraud_count"].iloc[0], 0)
        self.assertEqual(result["merchant_total_count"].iloc[0], 0)


class TestMerchantRiskFeaturesEdgeCases(unittest.TestCase):
    """Test edge cases for MerchantRiskFeatures."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        transformer = MerchantRiskFeatures()
        empty_X = pd.DataFrame(columns=["merchant_id"])
        empty_y = pd.Series([], dtype=int)
        transformer.fit(empty_X, empty_y)
        result = transformer.transform(empty_X)
        self.assertEqual(len(result), 0)

    def test_single_transaction(self):
        """Test with single transaction."""
        data = pd.DataFrame({
            "merchant_id": [1],
            "amount": [100.0],
        })
        transformer = MerchantRiskFeatures(alpha=1.0, beta=1.0)
        transformer.fit(data.drop(columns=["amount"]), pd.Series([0]))
        result = transformer.transform(data.drop(columns=["amount"]))
        self.assertEqual(result.shape, (1, 3))

    def test_all_legitimate(self):
        """Test when all transactions are legitimate."""
        data = pd.DataFrame({
            "merchant_id": [1] * 10,
            "amount": [100.0] * 10,
        })
        transformer = MerchantRiskFeatures(alpha=1.0, beta=1.0)
        transformer.fit(data.drop(columns=["amount"]), pd.Series([0] * 10))
        result = transformer.transform(data.drop(columns=["amount"]))

        self.assertTrue(all(result["merchant_fraud_rate"] >= 0))
        self.assertTrue(all(result["merchant_fraud_rate"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
