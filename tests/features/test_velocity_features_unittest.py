"""Unittest version of tests for VelocityFeatures transformer."""

import unittest
import pandas as pd
import numpy as np
from src.transformers.velocity_features import VelocityFeatures


class TestVelocityFeatures(unittest.TestCase):
    """Test suite for VelocityFeatures transformer."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n = 100
        self.sample_data = pd.DataFrame({
            "user_id": np.random.randint(1, 11, n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "amount": np.random.uniform(10, 500, n),
        })

        self.transformer = VelocityFeatures(
            user_col="user_id",
            datetime_col="timestamp",
            amount_col="amount",
            time_windows=[(1, "h"), (24, "h")],
            features=["count", "sum", "mean"],
        )

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        result = self.transformer.fit(self.sample_data)
        self.assertIs(result, self.transformer)

    def test_fit_stores_columns(self):
        """Test that fit stores column names."""
        self.transformer.fit(self.sample_data)
        self.assertTrue(hasattr(self.transformer, "feature_names_in_"))
        self.assertTrue(hasattr(self.transformer, "feature_names_out_"))

    def test_transform_returns_dataframe(self):
        """Test that transform returns DataFrame."""
        self.transformer.fit(self.sample_data)
        result = self.transformer.transform(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_output_shape(self):
        """Test that transform has correct output shape."""
        self.transformer.fit(self.sample_data)
        result = self.transformer.transform(self.sample_data)
        expected_cols = 7  # 2 windows * 3 features + time_since_last
        self.assertEqual(result.shape[1], expected_cols)

    def test_transform_feature_names(self):
        """Test that feature names are generated correctly."""
        self.transformer.fit(self.sample_data)
        result = self.transformer.transform(self.sample_data)
        expected_names = [
            "velocity_count_1h",
            "velocity_sum_1h",
            "velocity_mean_1h",
            "velocity_count_24h",
            "velocity_sum_24h",
            "velocity_mean_24h",
            "velocity_time_since_last_s",
        ]
        self.assertListEqual(list(result.columns), expected_names)

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        bad_data = self.sample_data.drop(columns=["amount"])
        with self.assertRaises(ValueError) as context:
            self.transformer.fit(bad_data)
        self.assertIn("Missing required columns", str(context.exception))

    def test_unseen_users_handled(self):
        """Test that unseen users are handled gracefully."""
        self.transformer.fit(self.sample_data)
        new_data = pd.DataFrame({
            "user_id": [999],
            "timestamp": [pd.Timestamp("2024-01-02")],
            "amount": [100.0],
        })
        result = self.transformer.transform(new_data)
        self.assertFalse(result.isna().any().any())


class TestVelocityFeaturesEdgeCases(unittest.TestCase):
    """Test edge cases for VelocityFeatures."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        transformer = VelocityFeatures()
        empty_df = pd.DataFrame(columns=["user_id", "timestamp", "amount"])
        transformer.fit(empty_df)
        result = transformer.transform(empty_df)
        self.assertEqual(len(result), 0)

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
        self.assertEqual(result.shape, (1, 2))

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
        self.assertFalse(result.isna().any().any())


if __name__ == "__main__":
    unittest.main()
