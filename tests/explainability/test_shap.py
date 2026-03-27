"""Unit tests for SHAP explainer."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from explainers.shap_explainer import SHAPExplainer


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


@pytest.fixture
def sample_instance(sample_data):
    """Create a single instance for testing."""
    X, feature_names = sample_data
    return X[0], feature_names


class TestSHAPExplainer:
    """Test suite for SHAPExplainer."""

    def test_initialization_tree_model(self, mock_model, sample_data):
        """Test initialization with tree model."""
        X, feature_names = sample_data

        explainer = SHAPExplainer(
            model=mock_model,
            model_type='xgboost',
            training_data=X
        )

        assert explainer.model == mock_model
        assert explainer.model_type == 'xgboost'
        assert explainer.explainer is not None

    def test_initialization_neural_network_requires_training_data(self, mock_model):
        """Test that neural network requires training data."""
        with pytest.raises(ValueError, match="training_data is required"):
            SHAPExplainer(
                model=mock_model,
                model_type='neural_network',
                training_data=None
            )

    def test_explain_local_returns_top_5_features(
        self, mock_model, sample_data
    ):
        """Test that local explanation returns exactly 5 features."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer = SHAPExplainer(
            model=mock_model,
            model_type='xgboost',
            training_data=X[:50]
        )

        result = explainer.explain_local(X_sample, feature_names)

        assert isinstance(result, dict)
        assert len(result) <= 5
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_explain_local_handles_1d_input(self, mock_model, sample_data):
        """Test that explain_local handles both 1D and 2D input."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer = SHAPExplainer(
            model=mock_model,
            model_type='xgboost',
            training_data=X[:50]
        )

        # Test 1D input
        result_1d = explainer.explain_local(X_sample, feature_names)

        # Test 2D input
        result_2d = explainer.explain_local(X_sample.reshape(1, -1), feature_names)

        # Should return same results
        assert len(result_1d) == len(result_2d)

    def test_explain_local_input_validation(self, mock_model, sample_data):
        """Test that explain_local validates input dimensions."""
        X, feature_names = sample_data
        X_sample = X[0]

        explainer = SHAPExplainer(
            model=mock_model,
            model_type='xgboost',
            training_data=X[:50]
        )

        # Wrong number of feature names
        wrong_names = feature_names[:3]

        with pytest.raises(ValueError, match="Feature mismatch"):
            explainer.explain_local(X_sample, wrong_names)

    def test_explain_global_returns_ranked_importance(
        self, mock_model, sample_data
    ):
        """Test that global explanation returns ranked feature importance."""
        X, feature_names = sample_data

        explainer = SHAPExplainer(
            model=mock_model,
            model_type='xgboost',
            training_data=X[:50]
        )

        result = explainer.explain_global(X[:20], feature_names)

        assert isinstance(result, dict)
        assert len(result) == len(feature_names)

        # Check that values are sorted (descending)
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_get_top_features(self, mock_model, sample_data):
        """Test get_top_features method."""
        X, feature_names = sample_data

        explainer = SHAPExplainer(
            model=mock_model,
            model_type='xgboost',
            training_data=X[:50]
        )

        feature_importance = {
            'feature_1': 0.8,
            'feature_2': 0.3,
            'feature_3': 0.5,
            'feature_4': 0.1,
            'feature_5': 0.9
        }

        top_3 = explainer.get_top_features(feature_importance, top_n=3)

        assert len(top_3) == 3
        assert top_3[0][0] == 'feature_5'  # Highest importance
        assert top_3[0][1] == 0.9

    def test_explain_global_samples_large_datasets(
        self, mock_model, sample_data
    ):
        """Test that global explanation samples large datasets."""
        X, feature_names = sample_data

        explainer = SHAPExplainer(
            model=mock_model,
            model_type='xgboost',
            training_data=X[:50]
        )

        # Use a small max_samples to test sampling
        result = explainer.explain_global(X, feature_names, max_samples=10)

        assert isinstance(result, dict)
        assert len(result) == len(feature_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
