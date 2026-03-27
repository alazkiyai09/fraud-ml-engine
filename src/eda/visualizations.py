"""Public EDA visualization surface for the unified fraud ML engine."""

from __future__ import annotations

from typing import Any

from src.eda.dashboard.visualizations import (
    plot_amount_histogram,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_pca_scatter,
    plot_time_patterns,
)

__all__ = [
    "plot_class_distribution",
    "plot_amount_histogram",
    "plot_correlation_heatmap",
    "plot_time_patterns",
    "plot_pca_scatter",
    "build_visualization_bundle",
]


def build_visualization_bundle(df: Any, *, log_scale: bool = False) -> dict[str, Any]:
    """Generate the standard EDA figure set used by the dashboard and notebooks."""
    return {
        "class_distribution": plot_class_distribution(df),
        "amount_histogram": plot_amount_histogram(df, log_scale=log_scale),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "time_patterns": plot_time_patterns(df),
        "pca_scatter": plot_pca_scatter(df),
    }
