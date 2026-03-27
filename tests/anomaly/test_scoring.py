"""Unit tests for anomaly scoring functionality."""

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
def sample_data():
    """Generate sample data for testing."""
    # Normal data
    X_normal, _ = make_blobs(
        n_samples=500,
        centers=1,
        n_features=10,
        random_state=42
    )

    # Anomalous data
    X_anomaly, _ = make_blobs(
        n_samples=50,
        centers=[[5, 5, 5, 5, 5, 5, 5, 5, 5, 5]],
        n_features=10,
        random_state=42
    )

    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0] * 500 + [1] * 50)

    return X_normal, X_anomaly, X, y


class TestAnomalyScoring:
    """Test suite for anomaly scoring."""

    def test_isolation_forest_scoring(self, sample_data):
        """Test Isolation Forest scoring output."""
        X_normal, X_anomaly, X, y = sample_data

        model = IsolationForestDetector(contamination=0.1)
        model.fit(X_normal)

        # Test scoring on normal data
        scores_normal = model.predict_anomaly_score(X_normal)
        assert scores_normal.shape == (len(X_normal),)
        assert np.all(scores_normal >= 0)

        # Test scoring on anomaly data
        scores_anomaly = model.predict_anomaly_score(X_anomaly)
        assert scores_anomaly.shape == (len(X_anomaly),)
        assert np.all(scores_anomaly >= 0)

        # Anomalies should have higher scores
        assert np.mean(scores_anomaly) > np.mean(scores_normal)

    def test_one_class_svm_scoring(self, sample_data):
        """Test One-Class SVM scoring output."""
        X_normal, X_anomaly, X, y = sample_data

        model = OneClassSVMDetector(nu=0.1)
        model.fit(X_normal)

        # Test scoring
        scores_normal = model.predict_anomaly_score(X_normal)
        scores_anomaly = model.predict_anomaly_score(X_anomaly)

        assert scores_normal.shape == (len(X_normal),)
        assert scores_anomaly.shape == (len(X_anomaly),)
        assert np.all(scores_normal >= 0)
        assert np.all(scores_anomaly >= 0)

        # Anomalies should have higher scores
        assert np.mean(scores_anomaly) > np.mean(scores_normal)

    def test_lof_scoring(self, sample_data):
        """Test LOF scoring output."""
        X_normal, X_anomaly, X, y = sample_data

        model = LOFDetector(contamination=0.1, n_neighbors=20)
        model.fit(X_normal)

        # Test scoring
        scores_normal = model.predict_anomaly_score(X_normal)
        scores_anomaly = model.predict_anomaly_score(X_anomaly)

        assert scores_normal.shape == (len(X_normal),)
        assert scores_anomaly.shape == (len(X_anomaly),)
        assert np.all(scores_normal >= 0)
        assert np.all(scores_anomaly >= 0)

        # Anomalies should have higher scores
        assert np.mean(scores_anomaly) > np.mean(scores_normal)

    def test_autoencoder_scoring(self, sample_data):
        """Test Autoencoder scoring output."""
        X_normal, X_anomaly, X, y = sample_data

        model = AutoencoderDetector(
            input_dim=X_normal.shape[1],
            hidden_dims=[32, 16],
            latent_dim=8,
            device="cpu"
        )

        # Train with minimal epochs for testing
        model.fit(X_normal, epochs=5, batch_size=64, verbose=False)

        # Test scoring
        scores_normal = model.predict_anomaly_score(X_normal)
        scores_anomaly = model.predict_anomaly_score(X_anomaly)

        assert scores_normal.shape == (len(X_normal),)
        assert scores_anomaly.shape == (len(X_anomaly),)
        assert np.all(scores_normal >= 0)
        assert np.all(scores_anomaly >= 0)

        # Anomalies should have higher reconstruction error
        assert np.mean(scores_anomaly) > np.mean(scores_normal)

    def test_threshold_setting(self, sample_data):
        """Test threshold setting functionality."""
        X_normal, X_anomaly, X, y = sample_data

        model = IsolationForestDetector(contamination=0.1)
        model.fit(X_normal)

        # Set threshold
        threshold = model.set_threshold(X_normal, target_fpr=0.05)
        assert threshold is not None
        assert isinstance(threshold, (int, float))

        # Test prediction with threshold
        predictions = model.predict(X_normal, threshold=threshold)
        assert predictions.shape == (len(X_normal),)
        assert np.all(np.isin(predictions, [0, 1]))

        # Should achieve approximately target FPR
        fpr = np.mean(predictions)
        assert 0.0 <= fpr <= 0.1  # Allow some tolerance

    def test_score_range_and_distribution(self, sample_data):
        """Test that scores have reasonable range and distribution."""
        X_normal, X_anomaly, X, y = sample_data

        model = IsolationForestDetector(contamination=0.1)
        model.fit(X_normal)

        scores = model.predict_anomaly_score(X)

        # Test range
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)

        # Test distribution (scores should vary)
        assert np.std(scores) > 0

        # Test percentiles exist
        assert np.percentile(scores, 50) >= 0
        assert np.percentile(scores, 95) >= 0

    def test_fitted_check(self, sample_data):
        """Test that models raise error when predicting before fitting."""
        X_normal, X_anomaly, X, y = sample_data

        models_to_test = [
            IsolationForestDetector(),
            OneClassSVMDetector(),
            LOFDetector(),
            AutoencoderDetector(input_dim=10)
        ]

        for model in models_to_test:
            with pytest.raises(RuntimeError):
                model.predict_anomaly_score(X)


class TestModelConsistency:
    """Test consistency across multiple runs."""

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random state."""
        X_normal, X_anomaly, X, y = sample_data

        model1 = IsolationForestDetector(contamination=0.1, random_state=42)
        model1.fit(X_normal)
        scores1 = model1.predict_anomaly_score(X)

        model2 = IsolationForestDetector(contamination=0.1, random_state=42)
        model2.fit(X_normal)
        scores2 = model2.predict_anomaly_score(X)

        np.testing.assert_array_almost_equal(scores1, scores2)

    def test_batch_vs_single_prediction(self, sample_data):
        """Test that batch predictions match individual predictions."""
        X_normal, X_anomaly, X, y = sample_data

        model = IsolationForestDetector(contamination=0.1)
        model.fit(X_normal)

        # Batch prediction
        scores_batch = model.predict_anomaly_score(X[:10])

        # Individual predictions
        scores_individual = np.array([
            model.predict_anomaly_score(X[i:i+1])[0]
            for i in range(10)
        ])

        np.testing.assert_array_almost_equal(scores_batch, scores_individual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
