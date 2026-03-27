"""
Publication-quality visualizations for imbalanced classification results.

Creates comparison plots, PR curves, ROC curves, and confusion matrices
for the benchmark results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pathlib import Path
import pandas as pd

from src.models.classical.legacy.config import config
from src.models.classical.legacy.experiment import ExperimentResult


# Set publication-quality style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_metrics_comparison(
    results: List[ExperimentResult],
    save_path: Path = None,
    figsize: tuple = None,
):
    """
    Create a bar chart comparing all metrics across techniques.

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    if figsize is None:
        figsize = config.FIGURE_SIZE

    # Prepare data
    techniques = [r.technique for r in results]
    metrics = list(results[0].metrics.keys())

    # Remove recall_at_fpr from bar chart (will plot separately)
    bar_metrics = [m for m in metrics if "recall_at_fpr" not in m]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(bar_metrics):
        ax = axes[idx]

        values = [r.metrics[metric] for r in results]
        errors = [r.std_metrics[metric] for r in results]

        # Create bar plot with error bars
        bars = ax.bar(range(len(techniques)), values, yerr=errors, capsize=3, alpha=0.8)

        # Customize
        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(techniques, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Metrics comparison plot saved to {save_path}")
    else:
        plt.savefig(
            config.RESULTS_DIR / "metrics_comparison.png",
            dpi=config.FIGURE_DPI,
            bbox_inches="tight",
        )

    plt.close()


def plot_recall_at_fpr(
    results: List[ExperimentResult],
    save_path: Path = None,
    figsize: tuple = None,
):
    """
    Plot Recall@FPR comparison (critical for fraud detection).

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    if figsize is None:
        figsize = (10, 6)

    # Extract Recall@FPR metric
    fpr_metric = f"recall_at_fpr_{int(config.FPR_THRESHOLD * 100)}pct"

    techniques = [r.technique for r in results]
    values = [r.metrics[fpr_metric] for r in results]
    errors = [r.std_metrics[fpr_metric] for r in results]

    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar plot for better readability
    y_pos = np.arange(len(techniques))
    bars = ax.barh(y_pos, values, xerr=errors, capsize=5, alpha=0.8)

    # Color bars by value (higher is better)
    colors = plt.cm.RdYlGn(np.array(values))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(techniques)
    ax.set_xlabel(f"Recall at {config.FPR_THRESHOLD*100}% FPR")
    ax.set_title(f"Recall at {config.FPR_THRESHOLD*100}% False Positive Rate")
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(
            val + errors[i],
            i,
            f" {val:.3f} ± {errors[i]:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Recall@FPR plot saved to {save_path}")
    else:
        plt.savefig(
            config.RESULTS_DIR / "recall_at_fpr.png",
            dpi=config.FIGURE_DPI,
            bbox_inches="tight",
        )

    plt.close()


def plot_metrics_heatmap(
    results: List[ExperimentResult],
    save_path: Path = None,
    figsize: tuple = None,
):
    """
    Create a heatmap comparing all metrics across techniques.

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    if figsize is None:
        figsize = (12, 8)

    # Prepare data
    techniques = [r.technique for r in results]
    metrics = list(results[0].metrics.keys())

    # Create matrix
    data_matrix = np.zeros((len(techniques), len(metrics)))
    for i, result in enumerate(results):
        for j, metric in enumerate(metrics):
            data_matrix[i, j] = result.metrics[metric]

    # Create DataFrame
    df = pd.DataFrame(
        data_matrix,
        index=techniques,
        columns=[m.replace("_", " ").title() for m in metrics],
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Score"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Imbalanced Classification Techniques - Metrics Comparison")
    ax.set_ylabel("Technique")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Metrics heatmap saved to {save_path}")
    else:
        plt.savefig(
            config.RESULTS_DIR / "metrics_heatmap.png",
            dpi=config.FIGURE_DPI,
            bbox_inches="tight",
        )

    plt.close()


def plot_ranking(
    results: List[ExperimentResult],
    save_path: Path = None,
    figsize: tuple = None,
):
    """
    Create a ranking plot based on F1 and AUPRC scores.

    Args:
        results: List of experiment results
        save_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    if figsize is None:
        figsize = (10, 6)

    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x.metrics["f1"], reverse=True)

    techniques = [r.technique for r in sorted_results]
    f1_scores = [r.metrics["f1"] for r in sorted_results]
    auprc_scores = [r.metrics["auprc"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(techniques))
    width = 0.35

    bars1 = ax.bar(x - width / 2, f1_scores, width, label="F1 Score", alpha=0.8)
    bars2 = ax.bar(x + width / 2, auprc_scores, width, label="AUPRC", alpha=0.8)

    ax.set_xlabel("Technique")
    ax.set_ylabel("Score")
    ax.set_title("Techniques Ranked by F1 and AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        print(f"Ranking plot saved to {save_path}")
    else:
        plt.savefig(
            config.RESULTS_DIR / "ranking.png",
            dpi=config.FIGURE_DPI,
            bbox_inches="tight",
        )

    plt.close()


def create_all_visualizations(results: List[ExperimentResult]):
    """
    Create all visualizations for the experiment results.

    Args:
        results: List of experiment results
    """
    print("\nCreating visualizations...")

    plot_metrics_comparison(results)
    plot_recall_at_fpr(results)
    plot_metrics_heatmap(results)
    plot_ranking(results)

    print("✓ All visualizations created!")
