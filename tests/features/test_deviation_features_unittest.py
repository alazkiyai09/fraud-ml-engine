"""Unittest version of tests for DeviationFeatures transformer."""

import unittest
import pandas as pd
import numpy as np
from src.transformers.deviation_features import DeviationFeatures


class TestDeviationFeatures(unittest.TestCase):
    """Test suite for DeviationFeatures transformer."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n = 100
        self.sample_data = pd.DataFrame({
            "user_id": np.random.randint(1, 11, n),
            "amount": np.random.uniform(10, 500, n),
            "hour_of_day": np.random.randint(0, 24, n),
        })

        self.transformer = DeviationFeatures(
            user_col="user_id",
            features=["amount", "hour_of_day"],
            window_size=30,
        )

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        result = self.transformer.fit(self.sample_data)
        self.assertIs(result, self.transformer)

    def test_fit_computes_statistics(self):
        """Test that fit computes user statistics."""
        self.transformer.fit(self.sample_data)
        self.assertTrue(hasattr(self.transformer, "user_stats_"))
        self.assertTrue(hasattr(self.transformer, "global_stats_"))
        self.assertIn("amount", self.transformer.user_stats_)
        self.assertIn("hour_of_day", self.transformer.user_stats_)

    def test_transform_returns_dataframe(self):
        """Test that transform returns DataFrame."""
        self.transformer.fit(self.sample_data)
        result = self.transformer.transform(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_transform_output_shape(self):
        """Test that transform has correct output shape."""
        self.transformer.fit(self.sample_data)
        result = self.transformer.transform(self.sample_data)
        self.assertEqual(result.shape[1], 4)  # 2 features * 2 metrics

    def test_transform_feature_names(self):
        """Test that feature names are generated correctly."""
        self.transformer.fit(self.sample_data)
        result = self.transformer.transform(self.sample_data)
        expected_names = [
            "deviation_amount_zscore",
            "deviation_amount_ratio",
            "deviation_hour_of_day_zscore",
            "deviation_hour_of_day_ratio",
        ]
        self.assertListEqual(list(result.columns), expected_names)

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        bad_data = self.sample_data.drop(columns=["amount"])
        with self.assertRaises(ValueError) as context:
            self.transformer.fit(bad_data)
        self.assertIn("Missing required columns", str(context.exception))

    def test_unseen_users_handled(self):
        """Test that unseen users fall back to global statistics."""
        self.transformer.fit(self.sample_data)
        new_data = pd.DataFrame({
            "user_id": [999],
            "amount": [100.0],
            "hour_of_day": [12],
        })
        result = self.transformer.transform(new_data)
        self.assertFalse(result.isna().any().any())

    def test_zscore_calculation(self):
        """Test that z-score is calculated correctly."""
        data = pd.DataFrame({
            "user_id": [1] * 50,
            "amount": [100.0] * 50,
        })
        transformer = DeviationFeatures(user_col="user_id", features=["amount"])
        transformer.fit(data)

        test_data = pd.DataFrame({
            "user_id": [1],
            "amount": [100.0],
        })
        result = transformer.transform(test_data)
        self.assertAlmostEqual(result["deviation_amount_zscore"].iloc[0], 0.0, places=10)

    def test_ratio_calculation(self):
        """Test that ratio is calculated correctly."""
        data = pd.DataFrame({
            "user_id": [1] * 10,
            "amount": [100.0] * 10,
        })
        transformer = DeviationFeatures(user_col="user_id", features=["amount"])
        transformer.fit(data)

        test_data = pd.DataFrame({
            "user_id": [1],
            "amount": [200.0],
        })
        result = transformer.transform(test_data)
        self.assertAlmostEqual(result["deviation_amount_ratio"].iloc[0], 2.0, places=5)


class TestDeviationFeaturesEdgeCases(unittest.TestCase):
    """Test edge cases for DeviationFeatures."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        transformer = DeviationFeatures(features=["amount"])
        empty_df = pd.DataFrame(columns=["user_id", "amount"])
        transformer.fit(empty_df)
        result = transformer.transform(empty_df)
        self.assertEqual(len(result), 0)

    def test_zero_variance_feature(self):
        """Test feature with zero variance."""
        data = pd.DataFrame({
            "user_id": [1, 1, 1],
            "amount": [100.0, 100.0, 100.0],
        })
        transformer = DeviationFeatures(features=["amount"])
        transformer.fit(data)
        result = transformer.transform(data)
        self.assertFalse(result.isna().any().any())

    def test_zero_mean_handling(self):
        """Test ratio calculation when mean is zero."""
        data = pd.DataFrame({
            "user_id": [1, 1, 1],
            "amount": [0.0, 0.0, 0.0],
        })
        transformer = DeviationFeatures(features=["amount"])
        transformer.fit(data)
        result = transformer.transform(data)
        self.assertEqual(result["deviation_amount_ratio"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
