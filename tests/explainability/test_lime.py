"""Unit tests for LIME explainer."""

import pytest
import numpy as np
from unittest.mock import Mock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from explainers.lime_explainer import LIMEExplainer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.predict_proba = Mock(return_value=np.array([[0.7, 0.3], [0.4, 0.6]]))
    model.predict = Mock(return_value=np.array([0, 1]))
    return model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    return X, feature_names


class TestLIMEExplainer:
    """Test suite for LIMEExplainer."""

    def test_initialization(self, mock_model, sample_data):
        """Test LIME explainer initialization."""
        X, feature_names = sample_data

        explainer = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names,
            model_type='generic'
        )

        assert explainer.model == mock_model
        assert explainer.model_type == 'generic'
        assert explainer.training_data is X
        assert explainer.feature_names == feature_names
        assert explainer.explainer is not None

    def test_initialization_requires_training_data(self, mock_model):
        """Test that training_data is required."""
        with pytest.raises(TypeError):
            LIMEExplainer(
                model=mock_model,
                training_data=None,
                feature_names=['feature_1'],
                model_type='generic'
            )

    def test_explain_local_returns_top_5_features(self, mock_model, sample_data):
        """Test that local explanation returns top 5 features."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names
        )

        result = explainer.explain_local(X_sample, feature_names)

        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_explain_local_converts_1d_to_2d(self, mock_model, sample_data):
        """Test that explain_local handles both 1D and 2D input."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names
        )

        # Test 1D input
        result_1d = explainer.explain_local(X_sample, feature_names)

        # Test 2D input
        result_2d = explainer.explain_local(X_sample.reshape(1, -1), feature_names)

        # Both should return dictionaries
        assert isinstance(result_1d, dict)
        assert isinstance(result_2d, dict)

    def test_explain_local_custom_num_features(self, mock_model, sample_data):
        """Test custom number of features parameter."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names
        )

        result = explainer.explain_local(X_sample, feature_names, num_features=3)

        assert len(result) <= 3

    def test_explain_global_aggregates_local_explanations(
        self, mock_model, sample_data
    ):
        """Test that global explanation aggregates local explanations."""
        X, feature_names = sample_data

        explainer = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names
        )

        result = explainer.explain_global(X[:10], feature_names, sample_size=10)

        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_explain_global_samples_large_datasets(self, mock_model, sample_data):
        """Test that global explanation respects sample_size parameter."""
        X, feature_names = sample_data

        explainer = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names
        )

        # Use small sample_size
        result = explainer.explain_global(X, feature_names, sample_size=5)

        assert isinstance(result, dict)

    def test_explain_local_feature_name_mismatch(self, mock_model, sample_data):
        """Test that feature name mismatch raises error."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names
        )

        # Wrong number of features
        wrong_names = ['feature_1', 'feature_2']

        with pytest.raises(ValueError, match="Feature names must match"):
            explainer.explain_local(X_sample, wrong_names)

    def test_random_state_for_reproducibility(self, mock_model, sample_data):
        """Test that random_state ensures reproducibility."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer1 = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names,
            random_state=42
        )

        explainer2 = LIMEExplainer(
            model=mock_model,
            training_data=X,
            feature_names=feature_names,
            random_state=42
        )

        result1 = explainer1.explain_local(X_sample, feature_names)
        result2 = explainer2.explain_local(X_sample, feature_names)

        # Should produce same results
        assert list(result1.keys()) == list(result2.keys())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
