"""Helper functions for risk classification and feature analysis."""

import hashlib
import json
from typing import Any

import numpy as np


def classify_risk_tier(probability: float) -> str:
    """
    Classify fraud probability into risk tier.

    Risk tiers:
    - LOW: probability < 0.1
    - MEDIUM: 0.1 <= probability < 0.5
    - HIGH: 0.5 <= probability < 0.9
    - CRITICAL: probability >= 0.9

    Args:
        probability: Fraud probability (0-1)

    Returns:
        Risk tier string (LOW, MEDIUM, HIGH, CRITICAL)

    Raises:
        ValueError: If probability is not between 0 and 1
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")

    if probability < 0.1:
        return "LOW"
    elif probability < 0.5:
        return "MEDIUM"
    elif probability < 0.9:
        return "HIGH"
    else:
        return "CRITICAL"


def get_top_risk_factors(
    feature_names: list[str],
    feature_values: np.ndarray,
    top_n: int = 3,
) -> list[str]:
    """
    Get top N features contributing to fraud score.

    This function ranks features by their absolute value contribution.
    Higher absolute values indicate stronger influence on the prediction.

    Args:
        feature_names: List of feature names
        feature_values: Array of feature values (e.g., SHAP values or coefficients)
        top_n: Number of top features to return

    Returns:
        List of top N feature names ranked by contribution
    """
    if len(feature_names) != len(feature_values):
        raise ValueError(
            f"Feature names ({len(feature_names)}) and values "
            f"({len(feature_values)}) must have same length"
        )

    if top_n <= 0:
        return []

    # Get indices of top N features by absolute value
    indices = np.argsort(np.abs(feature_values))[::-1][:top_n]

    # Return feature names at those indices
    top_features = [feature_names[i] for i in indices if i < len(feature_names)]

    return top_features


def compute_feature_hash(features: dict[str, Any]) -> str:
    """
    Compute hash of feature dict for cache key generation.

    Creates a deterministic hash from feature values to detect
    data changes that would affect predictions.

    Args:
        features: Dictionary of feature names to values

    Returns:
        Hexadecimal hash string
    """
    # Sort keys for deterministic ordering
    sorted_features = dict(sorted(features.items()))

    # Convert to JSON string
    feature_json = json.dumps(sorted_features, sort_keys=True, default=str)

    # Compute SHA256 hash
    hash_object = hashlib.sha256(feature_json.encode())
    return hash_object.hexdigest()[:16]  # First 16 chars


def format_probability(probability: float, decimals: int = 4) -> float:
    """
    Format probability to specified decimal places.

    Args:
        probability: Probability value (0-1)
        decimals: Number of decimal places

    Returns:
        Formatted probability
    """
    return round(probability, decimals)


def validate_transaction_features(features: dict[str, Any]) -> bool:
    """
    Validate that required transaction features are present.

    Args:
        features: Dictionary of transaction features

    Returns:
        True if all required features are present

    Raises:
        ValueError: If required features are missing
    """
    required_fields = {
        "transaction_id",
        "user_id",
        "merchant_id",
        "amount",
        "timestamp",
    }

    missing_fields = required_fields - set(features.keys())

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    return True


def compute_latency_ms(start_time_ns: int, end_time_ns: int) -> float:
    """
    Compute latency in milliseconds from nanosecond timestamps.

    Args:
        start_time_ns: Start time in nanoseconds (from time.time_ns())
        end_time_ns: End time in nanoseconds

    Returns:
        Latency in milliseconds
    """
    return (end_time_ns - start_time_ns) / 1_000_000
