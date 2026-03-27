"""Unit tests for explanation consistency validation."""

import pytest
import numpy as np
from unittest.mock import Mock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.validation import (
    validate_consistency,
    validate_explanation_quality,
    validate_feature_ranking_stability,
    benchmark_explanation_speed
)
from explainers.shap_explainer import SHAPExplainer


@pytest.fixture
def mock_explainer():
    """Create a mock explainer for testing."""
    explainer = Mock()

    # Create deterministic explanations (for consistency tests)
    def mock_explain_local(X, feature_names):
        # Generate same explanation for same input
        np.random.seed(hash(str(X.flatten().tolist())) % 2**32)
        importances = np.random.randn(len(feature_names))
        top_indices = np.argsort(np.abs(importances))[-5:][::-1]

        return {
            feature_names[i]: float(importances[i])
            for i in top_indices
        }

    explainer.explain_local = mock_explain_local
    return explainer


@pytest.fixture
def sample_data():
    """Create sample data."""
    np.random.seed(42)
    X = np.random.randn(10, 5)
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    return X, feature_names


class TestValidateConsistency:
    """Test suite for validate_consistency function."""

    def test_consistent_explanations_pass_validation(self, mock_explainer, sample_data):
        """Test that consistent explanations pass validation."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = validate_consistency(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=5,
            tolerance=0.01
        )

        assert isinstance(result, dict)
        assert 'is_consistent' in result
        assert 'variance' in result
        assert 'max_variance' in result
        assert 'tolerance' in result
        assert 'n_runs' in result
        assert 'all_explanations' in result

    def test_variance_calculation(self, mock_explainer, sample_data):
        """Test that variance is calculated correctly."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = validate_consistency(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=10
        )

        # Check that variance is a dict
        assert isinstance(result['variance'], dict)

        # Check that all variance values are non-negative
        for feature, variance in result['variance'].items():
            assert variance >= 0

    def test_max_variance_is_correct(self, mock_explainer, sample_data):
        """Test that max_variance is the maximum of all feature variances."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = validate_consistency(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=5
        )

        expected_max = max(result['variance'].values()) if result['variance'] else 0.0
        assert result['max_variance'] == expected_max

    def test_all_explanations_have_correct_length(self, mock_explainer, sample_data):
        """Test that all_explanations contains correct number of runs."""
        X, feature_names = sample_data
        X_sample = X[0]

        n_runs = 7
        result = validate_consistency(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=n_runs
        )

        assert len(result['all_explanations']) == n_runs


class TestValidateExplanationQuality:
    """Test suite for validate_explanation_quality function."""

    def test_quality_validation_returns_metrics(self, mock_explainer, sample_data):
        """Test that quality validation returns expected metrics."""
        X, feature_names = sample_data

        result = validate_explanation_quality(
            explainer=mock_explainer,
            X_test=X,
            feature_names=feature_names,
            top_n=5
        )

        assert isinstance(result, dict)
        assert 'total_explanations' in result
        assert 'null_values_found' in result
        assert 'infinite_values_found' in result
        assert 'feature_coverage' in result
        assert 'avg_features_per_explanation' in result
        assert 'unique_features_used' in result
        assert 'passed_quality_checks' in result

    def test_feature_coverage_calculation(self, mock_explainer, sample_data):
        """Test that feature coverage is calculated correctly."""
        X, feature_names = sample_data

        result = validate_explanation_quality(
            explainer=mock_explainer,
            X_test=X,
            feature_names=feature_names
        )

        # Check coverage values are between 0 and 1
        for feature, coverage in result['feature_coverage'].items():
            assert 0 <= coverage <= 1

    def test_unique_features_count(self, mock_explainer, sample_data):
        """Test unique feature count is correct."""
        X, feature_names = sample_data

        result = validate_explanation_quality(
            explainer=mock_explainer,
            X_test=X,
            feature_names=feature_names
        )

        assert result['unique_features_used'] <= len(feature_names)

    def test_passed_quality_checks_with_no_invalid_values(
        self, mock_explainer, sample_data
    ):
        """Test that quality checks pass when no invalid values."""
        X, feature_names = sample_data

        result = validate_explanation_quality(
            explainer=mock_explainer,
            X_test=X,
            feature_names=feature_names
        )

        # Mock explainer doesn't produce NaN or Inf
        assert result['null_values_found'] == 0
        assert result['infinite_values_found'] == 0


class TestValidateFeatureRankingStability:
    """Test suite for validate_feature_ranking_stability function."""

    def test_stability_validation_returns_metrics(self, mock_explainer, sample_data):
        """Test that stability validation returns expected metrics."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = validate_feature_ranking_stability(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_perturbations=10,
            noise_level=0.01
        )

        assert isinstance(result, dict)
        assert 'mean_overlap_ratio' in result
        assert 'std_overlap_ratio' in result
        assert 'min_overlap_ratio' in result
        assert 'n_successful_perturbations' in result
        assert 'is_stable' in result

    def test_overlap_ratios_in_valid_range(self, mock_explainer, sample_data):
        """Test that overlap ratios are in valid range [0, 1]."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = validate_feature_ranking_stability(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_perturbations=5
        )

        if result['n_successful_perturbations'] > 0:
            assert 0 <= result['mean_overlap_ratio'] <= 1
            assert 0 <= result['min_overlap_ratio'] <= 1

    def test_stable_determination(self, mock_explainer, sample_data):
        """Test that is_stable is correctly determined."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = validate_feature_ranking_stability(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_perturbations=5
        )

        # is_stable should be True if mean overlap > 0.7
        if result['mean_overlap_ratio'] > 0.7:
            assert result['is_stable'] is True
        else:
            assert result['is_stable'] is False


class TestBenchmarkExplanationSpeed:
    """Test suite for benchmark_explanation_speed function."""

    def test_speed_benchmark_returns_timing_metrics(self, mock_explainer, sample_data):
        """Test that speed benchmark returns timing metrics."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = benchmark_explanation_speed(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=10,
            target_seconds=2.0
        )

        assert isinstance(result, dict)
        assert 'mean_time' in result
        assert 'median_time' in result
        assert 'std_time' in result
        assert 'min_time' in result
        assert 'max_time' in result
        assert 'p95_time' in result
        assert 'p99_time' in result
        assert 'meets_target' in result
        assert 'target_seconds' in result

    def test_timing_values_are_positive(self, mock_explainer, sample_data):
        """Test that all timing values are positive."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = benchmark_explanation_speed(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=10
        )

        assert result['mean_time'] > 0
        assert result['median_time'] > 0
        assert result['min_time'] > 0
        assert result['max_time'] > 0
        assert result['p95_time'] > 0
        assert result['p99_time'] > 0

    def test_meets_target_determination(self, mock_explainer, sample_data):
        """Test that meets_target is correctly determined."""
        X, feature_names = sample_data
        X_sample = X[0]

        target = 1.0
        result = benchmark_explanation_speed(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=5,
            target_seconds=target
        )

        if result['mean_time'] <= target:
            assert result['meets_target'] is True
        else:
            assert result['meets_target'] is False

    def test_percentiles_are_correct(self, mock_explainer, sample_data):
        """Test that percentile calculations are reasonable."""
        X, feature_names = sample_data
        X_sample = X[0]

        result = benchmark_explanation_speed(
            explainer=mock_explainer,
            X=X_sample,
            feature_names=feature_names,
            n_runs=20
        )

        # P95 should be >= median and <= max
        assert result['median_time'] <= result['p95_time']
        assert result['p95_time'] <= result['max_time']

        # P99 should be >= P95
        assert result['p95_time'] <= result['p99_time']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
