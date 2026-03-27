"""Unit tests for FraudFeaturePipeline."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.pipeline import FraudFeaturePipeline


class TestFraudFeaturePipeline:
    """Test suite for FraudFeaturePipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
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
        return df

    @pytest.fixture
    def pipeline(self):
        """Create FraudFeaturePipeline instance."""
        return FraudFeaturePipeline(
            user_col="user_id",
            merchant_col="merchant_id",
            datetime_col="timestamp",
            amount_col="amount",
            time_windows=[(1, "h"), (24, "h")],
        )

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline.user_col == "user_id"
        assert pipeline.merchant_col == "merchant_id"
        assert pipeline.datetime_col == "timestamp"
        assert pipeline.amount_col == "amount"

    def test_fit_returns_self(self, pipeline, sample_data):
        """Test that fit returns self."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        result = pipeline.fit(X, y)
        assert result is pipeline

    def test_fit_creates_pipeline(self, pipeline, sample_data):
        """Test that fit creates internal pipeline."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        pipeline.fit(X, y)
        assert hasattr(pipeline, "pipeline_")

    def test_transform_returns_dataframe(self, pipeline, sample_data):
        """Test that transform returns DataFrame."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        pipeline.fit(X, y)
        result = pipeline.transform(X)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform(self, pipeline, sample_data):
        """Test fit_transform method."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        result = pipeline.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)

    def test_feature_names_out(self, pipeline, sample_data):
        """Test get_feature_names_out."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        pipeline.fit(X, y)
        feature_names = pipeline.get_feature_names_out()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_unseen_users_handled(self, pipeline, sample_data):
        """Test that unseen users are handled."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        pipeline.fit(X, y)

        # Create data with new user
        new_X = pd.DataFrame({
            "user_id": [999],
            "merchant_id": [1],
            "timestamp": [pd.Timestamp("2024-01-02")],
            "amount": [100.0],
        })
        result = pipeline.transform(new_X)
        assert not result.isna().any().any()

    def test_serialization(self, pipeline, sample_data):
        """Test save and load functionality."""
        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        pipeline.fit(X, y)

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp_path = tmp.name

        try:
            pipeline.save(tmp_path)
            assert os.path.exists(tmp_path)

            # Load and verify
            loaded_pipeline = FraudFeaturePipeline.load(tmp_path)
            assert hasattr(loaded_pipeline, "pipeline_")

            # Transform with loaded pipeline
            result = loaded_pipeline.transform(X)
            assert isinstance(result, pd.DataFrame)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestFraudFeaturePipelineWithSHAP:
    """Test pipeline with SHAP feature selection."""

    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
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
        return df

    def test_pipeline_with_shap_selection(self, sample_data):
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

        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        pipeline.fit(X, y)
        result = pipeline.transform(X)

        # Should have fewer features after SHAP selection
        feature_names = pipeline.get_feature_names_out()
        assert len(feature_names) <= 10

    def test_pipeline_without_shap_selection(self, sample_data):
        """Test pipeline without SHAP feature selection."""
        pipeline = FraudFeaturePipeline(
            user_col="user_id",
            merchant_col="merchant_id",
            datetime_col="timestamp",
            amount_col="amount",
            time_windows=[(1, "h")],
            use_shap_selection=False,
        )

        X = sample_data.drop(columns=["fraud"])
        y = sample_data["fraud"]
        pipeline.fit(X, y)
        result = pipeline.transform(X)

        # Should have all features
        assert result.shape[1] > 10


class TestFraudFeaturePipelineEdgeCases:
    """Test edge cases for FraudFeaturePipeline."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        pipeline = FraudFeaturePipeline()
        empty_X = pd.DataFrame(columns=["user_id", "merchant_id", "timestamp", "amount"])
        empty_y = pd.Series([], dtype=int)
        pipeline.fit(empty_X, empty_y)
        result = pipeline.transform(empty_X)
        assert len(result) == 0

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
        assert len(result) == 1
        assert isinstance(result, pd.DataFrame)

    def test_save_before_fit_raises_error(self):
        """Test that saving before fit raises error."""
        pipeline = FraudFeaturePipeline()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="No fitted pipeline"):
                pipeline.save(tmp_path)
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
        with pytest.raises(ValueError, match="not fitted"):
            pipeline.transform(X)
