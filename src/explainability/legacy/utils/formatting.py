"""Formatting utilities for human-readable explanations."""

from typing import Dict, List, Tuple, Any
import numpy as np


def format_risk_factors(
    feature_importance: Dict[str, float],
    top_n: int = 5,
    feature_descriptions: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Format feature importance into human-readable risk factors.

    Args:
        feature_importance: Dictionary of {feature_name: importance_score}
        top_n: Number of top features to return
        feature_descriptions: Optional mapping of feature names to human-readable descriptions

    Returns:
        List of dicts with keys: feature, description, importance, direction
    """
    # Sort features by absolute importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    # If feature_descriptions not provided, use feature names
    if feature_descriptions is None:
        feature_descriptions = {}

    formatted_factors = []
    for feature, importance in sorted_features:
        # Determine direction
        direction = "increases" if importance > 0 else "decreases"

        # Get description or use feature name
        description = feature_descriptions.get(feature, _make_readable_name(feature))

        formatted_factors.append({
            "feature": feature,
            "description": description,
            "importance": float(importance),
            "abs_importance": float(abs(importance)),
            "direction": direction,
            "impact_level": _get_impact_level(abs(importance))
        })

    return formatted_factors


def format_importance_scores(
    feature_importance: Dict[str, float],
    normalize: bool = True
) -> List[Tuple[str, float, str]]:
    """
    Format importance scores with percentages.

    Args:
        feature_importance: Dictionary of {feature_name: importance_score}
        normalize: Whether to normalize to percentages

    Returns:
        List of tuples (feature_name, score, formatted_string)
    """
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    if normalize and sorted_features:
        total = sum(abs(score) for _, score in sorted_features)
        if total > 0:
            normalized = [
                (feature, score, f"{abs(score) / total * 100:.1f}%")
                for feature, score in sorted_features
            ]
        else:
            normalized = [
                (feature, score, "0.0%")
                for feature, score in sorted_features
            ]
        return normalized
    else:
        return [
            (feature, score, f"{score:.4f}")
            for feature, score in sorted_features
        ]


def format_explanation_html(
    transaction_id: str,
    prediction: float,
    risk_factors: List[Dict[str, Any]],
    model_metadata: Dict[str, Any]
) -> str:
    """
    Format explanation as HTML snippet for embedding in reports.

    Args:
        transaction_id: Transaction identifier
        prediction: Fraud probability score
        risk_factors: List of formatted risk factors from format_risk_factors
        model_metadata: Model information (name, version, date, etc.)

    Returns:
        HTML string
    """
    risk_level = _get_risk_level(prediction)

    html_parts = [
        '<div class="explanation-summary">',
        f'  <h3>Transaction Analysis: {transaction_id}</h3>',
        f'  <div class="prediction-section">',
        f'    <p><strong>Fraud Probability:</strong> {prediction:.1%}</p>',
        f'    <p><strong>Risk Level:</strong> <span class="risk-{risk_level.lower()}">{risk_level}</span></p>',
        f'  </div>',
        f'  <div class="risk-factors-section">',
        f'    <h4>Top Risk Factors</h4>',
        f'    <ul class="risk-factors-list">'
    ]

    for i, factor in enumerate(risk_factors, 1):
        html_parts.append(
            f'      <li class="risk-factor-{i}">'
            f'<strong>{factor["description"]}</strong> '
            f'{factor["direction"]} fraud risk '
            f'(score: {factor["importance"]:.4f})'
            f'</li>'
        )

    html_parts.extend([
        '    </ul>',
        '  </div>',
        '  <div class="model-info">',
        f'    <p><small>Model: {model_metadata.get("name", "Unknown")} '
        f'| Version: {model_metadata.get("version", "N/A")} '
        f'| Date: {model_metadata.get("date", "N/A")}</small></p>',
        '  </div>',
        '</div>'
    ])

    return '\n'.join(html_parts)


def _make_readable_name(feature_name: str) -> str:
    """
    Convert feature name to human-readable format.

    Examples:
        'transaction_amount' -> 'Transaction Amount'
        'avg_transaction_amt_30d' -> 'Average Transaction Amount (30 Days)'
        'is_international' -> 'Is International Transaction'
    """
    # Replace underscores with spaces
    readable = feature_name.replace('_', ' ')

    # Capitalize first letter of each word
    readable = ' '.join(word.capitalize() for word in readable.split())

    # Expand common abbreviations
    abbreviations = {
        'Amt': 'Amount',
        'Num': 'Number',
        'Avg': 'Average',
        'Std': 'Standard Deviation',
        'Max': 'Maximum',
        'Min': 'Minimum',
        'Txn': 'Transaction',
        '30d': '30 Days',
        '7d': '7 Days',
        '24h': '24 Hours',
    }

    for abbr, expansion in abbreviations.items():
        readable = readable.replace(abbr, expansion)

    return readable


def _get_impact_level(importance: float) -> str:
    """
    Categorize importance into impact levels.

    Args:
        importance: Absolute importance score

    Returns:
        Impact level string
    """
    if importance > 0.5:
        return "Very High"
    elif importance > 0.3:
        return "High"
    elif importance > 0.1:
        return "Medium"
    elif importance > 0.05:
        return "Low"
    else:
        return "Very Low"


def _get_risk_level(probability: float) -> str:
    """
    Categorize probability into risk levels.

    Args:
        probability: Fraud probability (0-1)

    Returns:
        Risk level string
    """
    if probability >= 0.8:
        return "Critical"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"


def create_feature_description_mapping(
    feature_names: List[str],
    custom_descriptions: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Create a mapping of feature names to human-readable descriptions.

    Args:
        feature_names: List of feature names
        custom_descriptions: Optional custom descriptions to override auto-generated ones

    Returns:
        Dictionary mapping feature names to descriptions
    """
    descriptions = {}

    for feature in feature_names:
        if custom_descriptions and feature in custom_descriptions:
            descriptions[feature] = custom_descriptions[feature]
        else:
            descriptions[feature] = _make_readable_name(feature)

    return descriptions


def format_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Calculate confidence interval for explanation values.

    Args:
        values: List of importance values for a feature
        confidence: Confidence level (default 0.95)

    Returns:
        Dictionary with mean, lower, upper bounds
    """
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)

    # Use t-distribution for small samples, normal for large
    from scipy import stats
    n = len(values)
    if n < 30:
        # t-distribution
        t_score = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_score * std / np.sqrt(n)
    else:
        # normal distribution
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * std / np.sqrt(n)

    return {
        "mean": float(mean),
        "lower": float(mean - margin),
        "upper": float(mean + margin),
        "margin_of_error": float(margin),
        "confidence_level": confidence
    }
