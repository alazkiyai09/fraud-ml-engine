"""Unittest version of tests for FraudFeaturePipeline."""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from src.pipeline import FraudFeaturePipeline


class TestFraudFeaturePipeline(unittest.TestCase):
    """Test suite for FraudFeaturePipeline."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n = 200
        data = {
            "user_id": np.random.randint(1, 11, n),
            "merchant_id": np.random.randint(1, 11, n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "amount": np.random.uniform(10, 500, n),
        }
        df = pd.DataFrame(data)
        df["fraud"] = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        self.sample_data = df

        self.pipeline = FraudFeaturePipeline(
            user_col="user_id",
            merchant_col="merchant_id",
            datetime_col="timestamp",
            amount_col="amount",
            time_windows=[(1, "h"), (24, "h")],
        )

    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly."""
        self.assertEqual(self.pipeline.user_col, "user_id")
        self.assertEqual(self.pipeline.merchant_col, "merchant_id")
        self.assertEqual(self.pipeline.datetime_col, "timestamp")
        self.assertEqual(self.pipeline.amount_col, "amount")

    def test_fit_returns_self(self):
        """Test that fit returns self."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        result = self.pipeline.fit(X, y)
        self.assertIs(result, self.pipeline)

    def test_fit_creates_pipeline(self):
        """Test that fit creates internal pipeline."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.pipeline.fit(X, y)
        self.assertTrue(hasattr(self.pipeline, "pipeline_"))

    def test_transform_returns_dataframe(self):
        """Test that transform returns DataFrame."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.pipeline.fit(X, y)
        result = self.pipeline.transform(X)
        self.assertIsInstance(result, pd.DataFrame)

    def test_fit_transform(self):
        """Test fit_transform method."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        result = self.pipeline.fit_transform(X, y)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(X))

    def test_feature_names_out(self):
        """Test get_feature_names_out."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.pipeline.fit(X, y)
        feature_names = self.pipeline.get_feature_names_out()
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)

    def test_serialization(self):
        """Test save and load functionality."""
        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        self.pipeline.fit(X, y)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp_path = tmp.name

        try:
            self.pipeline.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            loaded_pipeline = FraudFeaturePipeline.load(tmp_path)
            self.assertTrue(hasattr(loaded_pipeline, "pipeline_"))

            result = loaded_pipeline.transform(X)
            self.assertIsInstance(result, pd.DataFrame)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestFraudFeaturePipelineWithSHAP(unittest.TestCase):
    """Test pipeline with SHAP feature selection."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n = 200
        data = {
            "user_id": np.random.randint(1, 11, n),
            "merchant_id": np.random.randint(1, 11, n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "amount": np.random.uniform(10, 500, n),
        }
        df = pd.DataFrame(data)
        df["fraud"] = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        self.sample_data = df

    def test_pipeline_with_shap_selection(self):
        """Test pipeline with SHAP feature selection enabled."""
        pipeline = FraudFeaturePipeline(
            user_col="user_id",
            merchant_col="merchant_id",
            datetime_col="timestamp",
            amount_col="amount",
            time_windows=[(1, "h")],
            use_shap_selection=True,
            n_features=10,
        )

        X = self.sample_data.drop(columns=["fraud"])
        y = self.sample_data["fraud"]
        pipeline.fit(X, y)
        result = pipeline.transform(X)

        feature_names = pipeline.get_feature_names_out()
        self.assertLessEqual(len(feature_names), 10)


class TestFraudFeaturePipelineEdgeCases(unittest.TestCase):
    """Test edge cases for FraudFeaturePipeline."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        pipeline = FraudFeaturePipeline()
        empty_X = pd.DataFrame(columns=["user_id", "merchant_id", "timestamp", "amount"])
        empty_y = pd.Series([], dtype=int)
        pipeline.fit(empty_X, empty_y)
        result = pipeline.transform(empty_X)
        self.assertEqual(len(result), 0)

    def test_single_transaction(self):
        """Test with single transaction."""
        data = pd.DataFrame({
            "user_id": [1],
            "merchant_id": [1],
            "timestamp": [pd.Timestamp("2024-01-01")],
            "amount": [100.0],
        })
        pipeline = FraudFeaturePipeline(time_windows=[(1, "h")])
        pipeline.fit(data, pd.Series([0]))
        result = pipeline.transform(data)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result, pd.DataFrame)

    def test_save_before_fit_raises_error(self):
        """Test that saving before fit raises error."""
        pipeline = FraudFeaturePipeline()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as context:
                pipeline.save(tmp_path)
            self.assertIn("No fitted pipeline", str(context.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_transform_before_fit_raises_error(self):
        """Test that transforming before fit raises error."""
        pipeline = FraudFeaturePipeline()
        X = pd.DataFrame({
            "user_id": [1],
            "merchant_id": [1],
            "timestamp": [pd.Timestamp("2024-01-01")],
            "amount": [100.0],
        })
        with self.assertRaises(ValueError) as context:
            pipeline.transform(X)
        self.assertIn("not fitted", str(context.exception))


if __name__ == "__main__":
    unittest.main()
