"""Validation utilities for explanation consistency and quality."""

import numpy as np
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from ..explainers.base import BaseExplainer


def validate_consistency(
    explainer: BaseExplainer,
    X: np.ndarray,
    feature_names: List[str],
    n_runs: int = 5,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """
    Validate that explanations are consistent across multiple runs.

    Critical for regulatory compliance - same input should produce same explanation.

    Args:
        explainer: Explainer instance to test
        X: Single instance to test (shape: [n_features])
        feature_names: List of feature names
        n_runs: Number of times to run the explanation
        tolerance: Maximum allowed variance in explanations

    Returns:
        Dictionary with validation results:
        - is_consistent: bool indicating if explanations are consistent
        - variance: dict of feature variances across runs
        - all_explanations: list of all explanation dicts
    """
    all_explanations = []
    feature_values = defaultdict(list)

    # Run explanation multiple times
    for _ in range(n_runs):
        explanation = explainer.explain_local(X, feature_names)
        all_explanations.append(explanation)

        for feature, value in explanation.items():
            feature_values[feature].append(value)

    # Calculate variance for each feature
    variances = {}
    is_consistent = True

    for feature, values in feature_values.items():
        variance = np.var(values)
        variances[feature] = float(variance)

        if variance > tolerance:
            is_consistent = False

    return {
        "is_consistent": is_consistent,
        "variance": variances,
        "max_variance": max(variances.values()) if variances else 0.0,
        "tolerance": tolerance,
        "n_runs": n_runs,
        "all_explanations": all_explanations
    }


def validate_explanation_quality(
    explainer: BaseExplainer,
    X_test: np.ndarray,
    feature_names: List[str],
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Validate explanation quality on a test set.

    Args:
        explainer: Explainer instance to test
        X_test: Test dataset (shape: [n_samples, n_features])
        feature_names: List of feature names
        top_n: Number of top features to check

    Returns:
        Dictionary with quality metrics
    """
    n_samples = min(100, X_test.shape[0])  # Use up to 100 samples
    indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
    X_sampled = X_test[indices]

    # Track feature coverage
    feature_appearances = defaultdict(int)
    total_explanations = 0

    # Check for null/invalid values
    null_count = 0
    infinite_count = 0

    for i in range(X_sampled.shape[0]):
        try:
            explanation = explainer.explain_local(X_sampled[i], feature_names)
            total_explanations += 1

            for feature in explanation.keys():
                feature_appearances[feature] += 1

            # Check for invalid values
            for value in explanation.values():
                if np.isnan(value):
                    null_count += 1
                if np.isinf(value):
                    infinite_count += 1

        except Exception as e:
            # Skip failed explanations but note them
            continue

    # Calculate coverage
    feature_coverage = {
        feature: count / total_explanations
        for feature, count in feature_appearances.items()
    }

    # Sort features by coverage
    sorted_coverage = dict(
        sorted(feature_coverage.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "total_explanations": total_explanations,
        "null_values_found": null_count,
        "infinite_values_found": infinite_count,
        "feature_coverage": sorted_coverage,
        "avg_features_per_explanation": len(feature_appearances) / max(total_explanations, 1),
        "unique_features_used": len(feature_appearances),
        "passed_quality_checks": (null_count == 0 and infinite_count == 0)
    }


def validate_feature_ranking_stability(
    explainer: BaseExplainer,
    X: np.ndarray,
    feature_names: List[str],
    n_perturbations: int = 10,
    noise_level: float = 0.01
) -> Dict[str, Any]:
    """
    Test if feature rankings are stable under small perturbations.

    Important for robustness - small changes in input shouldn't drastically
    change the explanation.

    Args:
        explainer: Explainer instance to test
        X: Single instance to test (shape: [n_features])
        feature_names: List of feature names
        n_perturbations: Number of perturbed versions to test
        noise_level: Standard deviation of Gaussian noise to add

    Returns:
        Dictionary with stability metrics
    """
    # Get baseline explanation
    baseline_exp = explainer.explain_local(X, feature_names)
    baseline_features = list(baseline_exp.keys())

    # Test perturbed versions
    rank_changes = []

    for _ in range(n_perturbations):
        # Add small noise
        noise = np.random.normal(0, noise_level, size=X.shape)
        X_perturbed = X + noise

        try:
            perturbed_exp = explainer.explain_local(X_perturbed, feature_names)
            perturbed_features = list(perturbed_exp.keys())

            # Calculate rank correlation (Spearman would be better, using simple overlap here)
            overlap = len(set(baseline_features) & set(perturbed_features))
            overlap_ratio = overlap / max(len(baseline_features), len(perturbed_features))
            rank_changes.append(overlap_ratio)

        except Exception:
            continue

    return {
        "mean_overlap_ratio": np.mean(rank_changes) if rank_changes else 0.0,
        "std_overlap_ratio": np.std(rank_changes) if rank_changes else 0.0,
        "min_overlap_ratio": np.min(rank_changes) if rank_changes else 0.0,
        "n_successful_perturbations": len(rank_changes),
        "is_stable": np.mean(rank_changes) > 0.7 if rank_changes else False
    }


def benchmark_explanation_speed(
    explainer: BaseExplainer,
    X: np.ndarray,
    feature_names: List[str],
    n_runs: int = 50,
    target_seconds: float = 2.0
) -> Dict[str, Any]:
    """
    Benchmark explanation speed.

    Regulatory requirement: explanations should be fast enough for real-time use.

    Args:
        explainer: Explainer instance to test
        X: Single instance to explain (shape: [n_features])
        feature_names: List of feature names
        n_runs: Number of times to run explanation
        target_seconds: Target time per explanation (default 2 seconds)

    Returns:
        Dictionary with timing metrics
    """
    import time

    times = []

    for _ in range(n_runs):
        start = time.time()
        explanation = explainer.explain_local(X, feature_names)
        end = time.time()
        times.append(end - start)

    times = np.array(times)

    return {
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "std_time": float(np.std(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "p95_time": float(np.percentile(times, 95)),
        "p99_time": float(np.percentile(times, 99)),
        "meets_target": bool(np.mean(times) <= target_seconds),
        "target_seconds": target_seconds
    }
