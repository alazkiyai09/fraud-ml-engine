"""Predictor tests."""

import pytest

from app.models.schemas import TransactionRequest
from app.models.predictor import FraudPredictor
from app.services.model_loader import ModelLoadError
from app.utils.helpers import classify_risk_tier, get_top_risk_factors


def test_classify_risk_tier():
    """Test risk tier classification."""
    assert classify_risk_tier(0.05) == "LOW"
    assert classify_risk_tier(0.1) == "MEDIUM"
    assert classify_risk_tier(0.3) == "MEDIUM"
    assert classify_risk_tier(0.5) == "HIGH"
    assert classify_risk_tier(0.7) == "HIGH"
    assert classify_risk_tier(0.9) == "CRITICAL"
    assert classify_risk_tier(1.0) == "CRITICAL"


def test_classify_risk_tier_invalid():
    """Test risk tier classification with invalid probability."""
    with pytest.raises(ValueError):
        classify_risk_tier(1.5)

    with pytest.raises(ValueError):
        classify_risk_tier(-0.1)


def test_get_top_risk_factors():
    """Test top risk factors extraction."""
    feature_names = ["amount", "hour", "velocity_count", "deviation_zscore"]
    feature_values = [0.1, 0.2, 0.9, 0.7]

    top_factors = get_top_risk_factors(feature_names, feature_values, top_n=2)

    assert len(top_factors) == 2
    assert "velocity_count" in top_factors  # Highest absolute value
    assert "deviation_zscore" in top_factors  # Second highest


def test_get_top_risk_factors_empty():
    """Test top risk factors with empty features."""
    feature_names = []
    feature_values = []

    top_factors = get_top_risk_factors(feature_names, feature_values)

    assert top_factors == []


def test_get_top_risk_factors_mismatch():
    """Test top risk factors with mismatched lengths."""
    feature_names = ["amount", "hour"]
    feature_values = [0.1, 0.2, 0.3]

    with pytest.raises(ValueError):
        get_top_risk_factors(feature_names, feature_values)


def test_predictor_initialization():
    """Test predictor initialization."""
    predictor = FraudPredictor(
        model_path="fake_model.pkl",
        pipeline_path="fake_pipeline.pkl",
        model_version="2.0.0",
    )

    assert predictor.model_version == "2.0.0"
    assert predictor._model is None
    assert predictor._pipeline is None


def test_predictor_is_model_loaded():
    """Test predictor model loaded check."""
    predictor = FraudPredictor(
        model_path="fake_model.pkl",
        pipeline_path="fake_pipeline.pkl",
    )

    # Model not loaded initially
    assert predictor.is_model_loaded() is False

    # Try to load (will fail with fake paths)
    with pytest.raises(ModelLoadError):
        predictor.load_model()
