"""
Callbacks module for the Fraud Detection EDA Dashboard.

This module contains all Dash callback functions that handle
user interactions and update dashboard components dynamically.
"""

from datetime import datetime
from pathlib import Path

import dash
from dash import Input, Output, State
import pandas as pd

from src.eda.dashboard.visualizations import (
    plot_amount_histogram,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_pca_scatter,
    plot_time_patterns,
)
from src.eda.dashboard.utils import calculate_summary_statistics


def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:
    """
    Register all callbacks with the Dash application.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    df : pd.DataFrame
        The preprocessed fraud detection dataset.

    Returns
    -------
    None

    Examples
    --------
    >>> app = dash.Dash(__name__)
    >>> register_callbacks(app, df)
    """
    @app.callback(
        Output('amount-range-display', 'children'),
        Input('amount-range-slider', 'value')
    )
    def update_amount_range_display(amount_range: list) -> str:
        """
        Update the displayed amount range.

        Parameters
        ----------
        amount_range : list
            List of [min, max] amount values from slider.

        Returns
        -------
        str
            Formatted string displaying the selected range.
        """
        min_amount, max_amount = amount_range
        return f"Selected Range: ${min_amount:.0f} - ${max_amount:.0f}"

    @app.callback(
        [
            Output('class-dist-chart', 'figure'),
            Output('amount-hist-chart', 'figure'),
            Output('correlation-heatmap', 'figure'),
            Output('time-patterns-chart', 'figure'),
            Output('pca-scatter-chart', 'figure'),
        ],
        [
            Input('amount-range-slider', 'value'),
            Input('log-scale-toggle', 'value'),
        ]
    )
    def update_charts(
        amount_range: list,
        log_scale_value: list
    ) -> tuple:
        """
        Update all charts based on filter selections.

        Parameters
        ----------
        amount_range : list
            [min, max] transaction amount range for filtering.
        log_scale_value : list
            List containing 'log' if log scale is enabled, empty otherwise.

        Returns
        -------
        tuple
            Tuple of 5 Plotly figures:
            - Class distribution figure
            - Amount histogram figure
            - Correlation heatmap figure
            - Time patterns figure
            - PCA scatter figure

        Notes
        -----
        Filters data by amount range before generating charts.
        Log scale toggle affects amount histogram display.
        """
        # Unpack amount range
        min_amount, max_amount = amount_range

        # Determine if log scale is enabled
        log_scale_enabled = 'log' in log_scale_value

        # Filter data by amount range
        df_filtered = df[
            (df['Amount'] >= min_amount) &
            (df['Amount'] <= max_amount)
        ].copy()

        # Generate all figures
        # Note: Some figures don't use all data, but filter is applied consistently
        fig_class_dist = plot_class_distribution(df_filtered)

        # Amount histogram respects log scale toggle
        fig_amount_hist = plot_amount_histogram(df_filtered, log_scale=log_scale_enabled)

        # Correlation heatmap
        fig_correlation = plot_correlation_heatmap(df_filtered)

        # Time patterns
        fig_time_patterns = plot_time_patterns(df_filtered)

        # PCA scatter - sample for performance if dataset is large
        sample_size = 5000 if len(df_filtered) > 10000 else None
        fig_pca = plot_pca_scatter(df_filtered, sample_size=sample_size)

        return (
            fig_class_dist,
            fig_amount_hist,
            fig_correlation,
            fig_time_patterns,
            fig_pca,
        )

    @app.callback(
        Output('export-status', 'children'),
        Input('export-button', 'n_clicks'),
        [
            State('class-dist-chart', 'figure'),
            State('amount-hist-chart', 'figure'),
            State('correlation-heatmap', 'figure'),
            State('time-patterns-chart', 'figure'),
            State('pca-scatter-chart', 'figure'),
        ],
        prevent_initial_call=True
    )
    def export_dashboard(
        n_clicks: int,
        fig_class_dist: dict,
        fig_amount_hist: dict,
        fig_correlation: dict,
        fig_time_patterns: dict,
        fig_pca: dict
    ) -> str:
        """
        Export all dashboard charts to individual HTML files.

        Parameters
        ----------
        n_clicks : int
            Number of times export button was clicked.
        fig_class_dist : dict
            Class distribution chart figure.
        fig_amount_hist : dict
            Amount histogram chart figure.
        fig_correlation : dict
            Correlation heatmap figure.
        fig_time_patterns : dict
            Time patterns chart figure.
        fig_pca : dict
            PCA scatter chart figure.

        Returns
        -------
        str
            Status message indicating export success or failure.

        Notes
        -----
        Exports each chart as a separate HTML file in the outputs/ directory.
        Filenames include timestamp to avoid overwriting previous exports.
        """
        if n_clicks == 0:
            return ""

        try:
            # Create outputs directory if it doesn't exist
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export each figure
            from src.eda.dashboard.utils import export_to_html
            import plotly.graph_objects as go

            # Convert dict figures back to Plotly figures
            figures = {
                f'class_distribution_{timestamp}.html': go.Figure(fig_class_dist),
                f'amount_histogram_{timestamp}.html': go.Figure(fig_amount_hist),
                f'correlation_heatmap_{timestamp}.html': go.Figure(fig_correlation),
                f'time_patterns_{timestamp}.html': go.Figure(fig_time_patterns),
                f'pca_scatter_{timestamp}.html': go.Figure(fig_pca),
            }

            # Export all figures
            for filename, fig in figures.items():
                filepath = outputs_dir / filename
                export_to_html(fig, str(filepath))

            return (
                f"✅ Exported {len(figures)} charts to outputs/ directory"
            )

        except Exception as e:
            return f"❌ Export failed: {str(e)}"


def register_summary_card_callback(app: dash.Dash, df: pd.DataFrame) -> None:
    """
    Register callback to update summary statistics card.

    This is a placeholder for future enhancement where filters
    would update the summary card dynamically.

    Parameters
    ----------
    app : dash.Dash
        The Dash application instance.
    df : pd.DataFrame
        The preprocessed fraud detection dataset.

    Returns
    -------
    None

    Examples
    --------
    >>> app = dash.Dash(__name__)
    >>> register_summary_card_callback(app, df)
    """
    # This callback is reserved for future implementation
    # Currently, summary stats are calculated once at initialization
    pass


def get_filtered_stats(df: pd.DataFrame, amount_range: list) -> dict:
    """
    Calculate summary statistics for filtered data.

    Helper function to compute statistics for a subset of data
    based on amount range filtering.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset.
    amount_range : list
        [min, max] transaction amount range for filtering.

    Returns
    -------
    dict
        Dictionary containing summary statistics for filtered data.

    Examples
    --------
    >>> stats = get_filtered_stats(df, [0, 500])
    >>> stats['total_transactions']
    5000
    """
    min_amount, max_amount = amount_range

    # Filter by amount range
    df_filtered = df[
        (df['Amount'] >= min_amount) &
        (df['Amount'] <= max_amount)
    ]

    # Calculate statistics
    return calculate_summary_statistics(df_filtered)
