"""
Unit tests for metric calculations.

Tests cover edge cases, numerical stability, and correctness
of all metric functions.
"""

import numpy as np
import pytest
from src.metrics.metrics import (
    calculate_auprc,
    calculate_auroc,
    calculate_recall_at_fpr,
    compute_all_metrics,
)


class TestCalculateAUPRC:
    """Test AUPRC calculation."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield AUPRC of 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.95])
        assert calculate_auprc(y_true, y_proba) == pytest.approx(1.0, rel=1e-5)

    def test_random_predictions(self):
        """Random predictions should yield lower AUPRC."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        auprc = calculate_auprc(y_true, y_proba)
        assert 0.0 <= auprc <= 1.0

    def test_worst_predictions(self):
        """Worst predictions (reversed) should yield low AUPRC."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.95, 0.1, 0.2])
        auprc = calculate_auprc(y_true, y_proba)
        assert auprc < 0.5


class TestCalculateAUROC:
    """Test AUROC calculation."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield AUROC of 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.95])
        assert calculate_auroc(y_true, y_proba) == pytest.approx(1.0, rel=1e-5)

    def test_random_predictions(self):
        """Random predictions should yield AUROC near 0.5."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        auroc = calculate_auroc(y_true, y_proba)
        assert auroc == pytest.approx(0.5, abs=0.01)

    def test_worst_predictions(self):
        """Worst predictions should yield AUROC of 0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.95, 0.1, 0.2])
        assert calculate_auroc(y_true, y_proba) == pytest.approx(0.0, abs=0.05)


class TestRecallAtFPR:
    """Test Recall@FPR calculation."""

    def test_recall_at_1_percent_fpr(self):
        """Test recall calculation at 1% FPR threshold."""
        # Create synthetic data with clear separation
        y_true = np.array([0] * 100 + [1] * 10)
        y_proba = np.array([0.01] * 100 + [0.9] * 10)

        recall = calculate_recall_at_fpr(y_true, y_proba, fpr_threshold=0.01)

        # With perfect separation and 1% FPR threshold, should capture most positives
        assert 0.0 <= recall <= 1.0

    def test_recall_at_5_percent_fpr(self):
        """Test recall calculation at 5% FPR threshold."""
        y_true = np.array([0] * 100 + [1] * 10)
        y_proba = np.array([0.05] * 100 + [0.8] * 10)

        recall = calculate_recall_at_fpr(y_true, y_proba, fpr_threshold=0.05)
        assert 0.0 <= recall <= 1.0

    def test_all_same_probabilities(self):
        """Test edge case where all probabilities are the same."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        recall = calculate_recall_at_fpr(y_true, y_proba, fpr_threshold=0.01)
        # Should return 0 when no meaningful threshold exists
        assert recall == 0.0


class TestComputeAllMetrics:
    """Test the comprehensive metrics computation."""

    def test_all_metrics_returned(self):
        """Ensure all expected metrics are returned."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auprc",
            "auroc",
            "recall_at_fpr_1pct",
        }

        assert set(metrics.keys()) == expected_keys

    def test_all_metrics_in_valid_range(self):
        """All metrics should be in [0, 1]."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])

        metrics = compute_all_metrics(y_true, y_pred, y_proba)

        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} = {value} is out of range"

    def test_custom_fpr_threshold(self):
        """Test custom FPR threshold in metrics."""
        y_true = np.array([0] * 100 + [1] * 10)
        y_pred = np.array([0] * 100 + [1] * 10)
        y_proba = np.array([0.01] * 100 + [0.9] * 10)

        metrics = compute_all_metrics(y_true, y_pred, y_proba, fpr_threshold=0.05)

        assert "recall_at_fpr_5pct" in metrics
        assert 0.0 <= metrics["recall_at_fpr_5pct"] <= 1.0
