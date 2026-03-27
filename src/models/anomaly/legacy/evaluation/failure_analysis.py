"""Failure analysis utilities for anomaly detection."""

from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_failures(
    X: np.ndarray,  # type: ignore[valid-type]
    y_true: np.ndarray,  # type: ignore[valid-type]
    y_pred: np.ndarray,  # type: ignore[valid-type]
    anomaly_scores: np.ndarray,  # type: ignore[valid-type]
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Analyze failure cases (false positives and false negatives).

    Args:
        X: Feature matrix.
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        anomaly_scores: Anomaly scores.
        feature_names: Optional list of feature names.

    Returns:
        DataFrame containing failure cases with features and metadata.
    """
    n_samples = len(y_true)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['true_label'] = y_true
    df['predicted_label'] = y_pred
    df['anomaly_score'] = anomaly_scores

    # Classify prediction types
    df['prediction_type'] = 'True Negative'
    df.loc[(df['true_label'] == 0) & (df['predicted_label'] == 1), 'prediction_type'] = 'False Positive'
    df.loc[(df['true_label'] == 1) & (df['predicted_label'] == 0), 'prediction_type'] = 'False Negative'
    df.loc[(df['true_label'] == 1) & (df['predicted_label'] == 1), 'prediction_type'] = 'True Positive'

    # Filter for failures only
    failures = df[df['prediction_type'].isin(['False Positive', 'False Negative'])].copy()

    return failures


def summarize_failures(failures_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for failure cases.

    Args:
        failures_df: DataFrame from analyze_failures().

    Returns:
        Dictionary with failure statistics.
    """
    summary = {
        'total_failures': len(failures_df),
        'false_positives': sum(failures_df['prediction_type'] == 'False Positive'),
        'false_negatives': sum(failures_df['prediction_type'] == 'False Negative'),
        'fp_mean_score': failures_df[failures_df['prediction_type'] == 'False Positive']['anomaly_score'].mean(),
        'fn_mean_score': failures_df[failures_df['prediction_type'] == 'False Negative']['anomaly_score'].mean(),
    }

    return summary


def visualize_failure_distributions(
    failures_df: pd.DataFrame,
    save_path: str = None
) -> None:
    """
    Visualize anomaly score distributions for failure types.

    Args:
        failures_df: DataFrame from analyze_failures().
        save_path: Optional path to save plot.
    """
    fp_scores = failures_df[failures_df['prediction_type'] == 'False Positive']['anomaly_score']
    fn_scores = failures_df[failures_df['prediction_type'] == 'False Negative']['anomaly_score']

    plt.figure(figsize=(12, 5))

    # Score distribution
    plt.subplot(1, 2, 1)
    plt.hist(fp_scores, bins=50, alpha=0.5, label='False Positive', color='red')
    plt.hist(fn_scores, bins=50, alpha=0.5, label='False Negative', color='blue')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution for Failures')
    plt.legend()

    # Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [fp_scores.values, fn_scores.values]
    plt.boxplot(data_to_plot, labels=['False Positive', 'False Negative'])
    plt.ylabel('Anomaly Score')
    plt.title('Score Distribution by Failure Type')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Failure distribution plot saved to {save_path}")

    plt.close()


def visualize_feature_importance_for_failures(
    failures_df: pd.DataFrame,
    feature_names: List[str] = None,
    top_n: int = 10,
    save_path: str = None
) -> None:
    """
    Visualize feature statistics for failure cases.

    Args:
        failures_df: DataFrame from analyze_failures().
        feature_names: List of feature names to analyze.
        top_n: Number of top features to show.
        save_path: Optional path to save plot.
    """
    if feature_names is None:
        # Exclude metadata columns
        feature_names = [col for col in failures_df.columns
                        if col not in ['true_label', 'predicted_label', 'anomaly_score', 'prediction_type']]

    # Separate FP and FN
    fp = failures_df[failures_df['prediction_type'] == 'False Positive'][feature_names]
    fn = failures_df[failures_df['prediction_type'] == 'False Negative'][feature_names]

    # Compute mean absolute values for each feature
    fp_means = fp.abs().mean().sort_values(ascending=False)
    fn_means = fn.abs().mean().sort_values(ascending=False)

    # Get top features
    top_features = list((fp_means + fn_means).sort_values(ascending=False).head(top_n).index)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # False Positives
    fp[top_features].mean().sort_values().plot(kind='barh', ax=axes[0])
    axes[0].set_xlabel('Mean Feature Value')
    axes[0].set_title('False Positives - Top Features')

    # False Negatives
    fn[top_features].mean().sort_values().plot(kind='barh', ax=axes[1])
    axes[1].set_xlabel('Mean Feature Value')
    axes[1].set_title('False Negatives - Top Features')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.close()


def export_failure_cases(
    failures_df: pd.DataFrame,
    output_path: str,
    format: str = "csv"
) -> None:
    """
    Export failure cases to file.

    Args:
        failures_df: DataFrame from analyze_failures().
        output_path: Path to save file.
        format: File format ('csv', 'parquet', 'excel').
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        failures_df.to_csv(output_path, index=False)
    elif format == "parquet":
        failures_df.to_parquet(output_path, index=False)
    elif format == "excel":
        failures_df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Failure cases exported to {output_path}")


def compare_model_failures(
    X: np.ndarray,  # type: ignore[valid-type]
    y_true: np.ndarray,  # type: ignore[valid-type]
    model_predictions: Dict[str, np.ndarray],  # type: ignore[valid-type]
    model_scores: Dict[str, np.ndarray],  # type: ignore[valid-type]
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Compare failure cases across multiple models.

    Args:
        X: Feature matrix.
        y_true: True labels.
        model_predictions: Dict mapping model names to binary predictions.
        model_scores: Dict mapping model names to anomaly scores.
        feature_names: List of feature names.

    Returns:
        DataFrame comparing failures across models.
    """
    comparison = []

    for model_name, y_pred in model_predictions.items():
        failures = analyze_failures(
            X, y_true, y_pred, model_scores[model_name], feature_names
        )
        summary = summarize_failures(failures)
        summary['model'] = model_name
        comparison.append(summary)

    return pd.DataFrame(comparison)
