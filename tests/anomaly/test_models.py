"""Unit tests for model fitting and basic functionality."""

import pytest
import numpy as np
from sklearn.datasets import make_blobs

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.isolation_forest import IsolationForestDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.lof import LOFDetector
from src.models.autoencoder import AutoencoderDetector


@pytest.fixture
def train_data():
    """Generate training data (class 0 only)."""
    X, _ = make_blobs(
        n_samples=300,
        centers=1,
        n_features=10,
        random_state=42
    )
    return X


@pytest.fixture
def test_data_mixed():
    """Generate mixed test data."""
    X_normal, _ = make_blobs(
        n_samples=200,
        centers=1,
        n_features=10,
        random_state=42
    )

    X_anomaly, _ = make_blobs(
        n_samples=20,
        centers=[[5, 5, 5, 5, 5, 5, 5, 5, 5, 5]],
        n_features=10,
        random_state=42
    )

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * 200 + [1] * 20)

    return X, y


class TestModelFitting:
    """Test suite for model fitting."""

    def test_isolation_forest_fitting(self, train_data):
        """Test Isolation Forest fitting."""
        model = IsolationForestDetector(contamination=0.1)

        assert not model.is_fitted

        model.fit(train_data)

        assert model.is_fitted
        assert model.model is not None

    def test_one_class_svm_fitting(self, train_data):
        """Test One-Class SVM fitting."""
        model = OneClassSVMDetector(nu=0.1)

        assert not model.is_fitted

        model.fit(train_data)

        assert model.is_fitted
        assert model.model is not None

    def test_lof_fitting(self, train_data):
        """Test LOF fitting."""
        model = LOFDetector(contamination=0.1, n_neighbors=20)

        assert not model.is_fitted

        model.fit(train_data)

        assert model.is_fitted
        assert model.model is not None

    def test_autoencoder_fitting(self, train_data):
        """Test Autoencoder fitting."""
        model = AutoencoderDetector(
            input_dim=train_data.shape[1],
            hidden_dims=[32, 16],
            latent_dim=8,
            device="cpu"
        )

        assert not model.is_fitted

        model.fit(train_data, epochs=5, batch_size=64, verbose=False)

        assert model.is_fitted
        assert len(model.training_losses) > 0

    def test_autoencoder_loss_decreases(self, train_data):
        """Test that Autoencoder loss decreases during training."""
        model = AutoencoderDetector(
            input_dim=train_data.shape[1],
            hidden_dims=[32, 16],
            latent_dim=8,
            device="cpu"
        )

        model.fit(train_data, epochs=10, batch_size=64, verbose=False)

        # Loss should decrease (last epoch < first epoch)
        assert model.training_losses[-1] < model.training_losses[0]


class TestPredictionShapes:
    """Test prediction output shapes."""

    def test_isolation_forest_prediction_shape(self, train_data, test_data_mixed):
        """Test Isolation Forest prediction shapes."""
        X_test, y_test = test_data_mixed

        model = IsolationForestDetector(contamination=0.1)
        model.fit(train_data)

        scores = model.predict_anomaly_score(X_test)
        assert scores.shape == (len(X_test),)

        model.set_threshold(train_data, target_fpr=0.05)
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_one_class_svm_prediction_shape(self, train_data, test_data_mixed):
        """Test One-Class SVM prediction shapes."""
        X_test, y_test = test_data_mixed

        model = OneClassSVMDetector(nu=0.1)
        model.fit(train_data)

        scores = model.predict_anomaly_score(X_test)
        assert scores.shape == (len(X_test),)

        model.set_threshold(train_data, target_fpr=0.05)
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_lof_prediction_shape(self, train_data, test_data_mixed):
        """Test LOF prediction shapes."""
        X_test, y_test = test_data_mixed

        model = LOFDetector(contamination=0.1, n_neighbors=20)
        model.fit(train_data)

        scores = model.predict_anomaly_score(X_test)
        assert scores.shape == (len(X_test),)

        model.set_threshold(train_data, target_fpr=0.05)
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_autoencoder_prediction_shape(self, train_data, test_data_mixed):
        """Test Autoencoder prediction shapes."""
        X_test, y_test = test_data_mixed

        model = AutoencoderDetector(
            input_dim=train_data.shape[1],
            hidden_dims=[32, 16],
            latent_dim=8,
            device="cpu"
        )
        model.fit(train_data, epochs=5, batch_size=64, verbose=False)

        scores = model.predict_anomaly_score(X_test)
        assert scores.shape == (len(X_test),)

        model.set_threshold(train_data, target_fpr=0.05)
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert np.all(np.isin(predictions, [0, 1]))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_training_data(self):
        """Test model behavior with empty training data."""
        model = IsolationForestDetector(contamination=0.1)
        X_empty = np.array([]).reshape(0, 10)

        with pytest.raises((ValueError, IndexError)):
            model.fit(X_empty)

    def test_single_sample_training(self):
        """Test model behavior with single training sample."""
        X_single = np.random.randn(1, 10)

        # LOF should fail with single sample
        model_lof = LOFDetector(contamination=0.1)
        with pytest.raises(ValueError):
            model_lof.fit(X_single)

    def test_mismatched_feature_dimensions(self, train_data):
        """Test prediction with wrong number of features."""
        model = IsolationForestDetector(contamination=0.1)
        model.fit(train_data)

        X_wrong_dim = np.random.randn(50, 15)  # Wrong number of features

        with pytest.raises((ValueError, TypeError)):
            model.predict_anomaly_score(X_wrong_dim)

    def test_invalid_contamination_values(self, train_data):
        """Test invalid contamination parameter values."""
        # Contamination > 1
        model = IsolationForestDetector(contamination=1.5)
        model.fit(train_data)
        # Should still work but may not behave as expected

        # Negative contamination
        model2 = IsolationForestDetector(contamination=-0.1)
        with pytest.raises((ValueError, TypeError)):
            model2.fit(train_data)


class TestAutoencoderSpecific:
    """Tests specific to Autoencoder."""

    def test_model_save_load(self, train_data, tmp_path):
        """Test saving and loading Autoencoder model."""
        model = AutoencoderDetector(
            input_dim=train_data.shape[1],
            hidden_dims=[32, 16],
            latent_dim=8,
            device="cpu"
        )
        model.fit(train_data, epochs=5, batch_size=64, verbose=False)

        # Save
        save_path = tmp_path / "model.pt"
        model.save_model(str(save_path))

        # Load into new model
        model2 = AutoencoderDetector(
            input_dim=train_data.shape[1],
            device="cpu"
        )
        model2.load_model(str(save_path))

        # Check predictions match
        scores1 = model.predict_anomaly_score(train_data[:10])
        scores2 = model2.predict_anomaly_score(train_data[:10])

        np.testing.assert_array_almost_equal(scores1, scores2)

    def test_latent_representation(self, train_data):
        """Test that encoder produces latent representations."""
        model = AutoencoderDetector(
            input_dim=train_data.shape[1],
            hidden_dims=[32, 16],
            latent_dim=8,
            device="cpu"
        )
        model.fit(train_data, epochs=5, batch_size=64, verbose=False)

        import torch

        # Get latent representation
        model.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(train_data[:10])
            latent = model.model.encode(X_tensor)

        assert latent.shape == (10, 8)  # batch_size x latent_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
