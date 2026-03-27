"""
Visualization module for the Fraud Detection EDA Dashboard.

This module creates all Plotly figures for the dashboard including:
- Class distribution bar chart
- Transaction amount histogram
- Correlation heatmap
- Time patterns analysis
- PCA scatter plot
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.eda.dashboard.utils import FRAUD_COLOR, LEGIT_COLOR


def plot_class_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing the distribution of fraud vs legitimate transactions.

    Displays the count and percentage of each class with distinct colors
    for easy visual comparison.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset. Must contain 'Class' column.

    Returns
    -------
    go.Figure
        Plotly bar chart figure showing class distribution.

    Raises
    ------
    KeyError
        If 'Class' column is missing from DataFrame.

    Examples
    --------
    >>> fig = plot_class_distribution(df)
    >>> fig.show()
    """
    if 'Class' not in df.columns:
        raise KeyError("DataFrame must contain 'Class' column")

    # Calculate class counts
    class_counts = df['Class'].value_counts().sort_index()
    total = len(df)

    # Create labels and values
    labels = ['Legitimate', 'Fraud']
    values = [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ]
    percentages = [
        round((values[0] / total) * 100, 2) if total > 0 else 0,
        round((values[1] / total) * 100, 2) if total > 0 else 0
    ]
    colors = [LEGIT_COLOR, FRAUD_COLOR]

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[
                f"{values[0]:,}<br>({percentages[0]}%)",
                f"{values[1]:,}<br>({percentages[1]}%)"
            ],
            textposition='auto',
            textfont={'size': 14},
        )
    ])

    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Class Distribution</b><br>Transaction Counts by Class',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Transaction Class',
        yaxis_title='Count',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
    )

    fig.update_yaxis(type="log")  # Log scale due to class imbalance

    return fig


def plot_amount_histogram(
    df: pd.DataFrame,
    log_scale: bool = False
) -> go.Figure:
    """
    Create a histogram showing transaction amount distribution by class.

    Displays separate histograms for fraud and legitimate transactions
    with overlapping distributions for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset. Must contain 'Amount' and 'Class' columns.
    log_scale : bool, optional
        Whether to use logarithmic scale for x-axis (default: False).
        Useful for handling skewed amount distributions.

    Returns
    -------
    go.Figure
        Plotly histogram figure showing amount distribution.

    Raises
    ------
    KeyError
        If required columns ('Amount', 'Class') are missing.

    Examples
    --------
    >>> fig = plot_amount_histogram(df, log_scale=True)
    >>> fig.show()
    """
    required_columns = {'Amount', 'Class'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Separate data by class
    legit_amounts = df[df['Class'] == 0]['Amount'].dropna()
    fraud_amounts = df[df['Class'] == 1]['Amount'].dropna()

    # Create histogram
    fig = go.Figure()

    # Add legitimate transactions
    fig.add_trace(go.Histogram(
        x=legit_amounts,
        name='Legitimate',
        marker_color=LEGIT_COLOR,
        opacity=0.6,
        nbinsx=50,
    ))

    # Add fraudulent transactions
    fig.add_trace(go.Histogram(
        x=fraud_amounts,
        name='Fraud',
        marker_color=FRAUD_COLOR,
        opacity=0.6,
        nbinsx=50,
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Transaction Amount Distribution</b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Transaction Amount ($)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
    )

    # Apply log scale if requested
    if log_scale:
        fig.update_xaxis(type="log")
        fig.update_layout(
            xaxis_title='Transaction Amount ($) - Log Scale'
        )

    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing correlations between PCA features and target.

    Displays correlation matrix with color intensity indicating correlation
    strength. Focuses on V-features, Amount, Time, and Class.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset. Should contain V-features, Amount,
        Time, and Class columns.

    Returns
    -------
    go.Figure
        Plotly heatmap figure showing correlations.

    Examples
    --------
    >>> fig = plot_correlation_heatmap(df)
    >>> fig.show()
    """
    # Select relevant columns for correlation
    v_columns = [col for col in df.columns if col.startswith('V')]
    corr_columns = v_columns + ['Amount', 'Time', 'Class']
    existing_columns = [col for col in corr_columns if col in df.columns]

    # Calculate correlation matrix
    corr_matrix = df[existing_columns].corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title='Correlation',
            titleside='right',
        ),
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Feature Correlation Heatmap</b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600,
        margin=dict(l=100, r=50, t=80, b=100),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
    )

    return fig


def plot_time_patterns(df: pd.DataFrame) -> go.Figure:
    """
    Create visualizations showing temporal patterns in fraud transactions.

    Displays two subplots:
    1. Transactions by hour of day (if Hour column exists)
    2. Fraud ratio by time period

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset. Must contain 'Class' column.
        If 'Hour' column exists, uses it for hourly analysis.
        Otherwise, uses 'Time' column (in seconds).

    Returns
    -------
    go.Figure
        Plotly figure with time pattern subplots.

    Raises
    ------
    KeyError
        If required columns are missing.

    Examples
    --------
    >>> fig = plot_time_patterns(df)
    >>> fig.show()
    """
    if 'Class' not in df.columns:
        raise KeyError("DataFrame must contain 'Class' column")

    # Determine time column to use
    if 'Hour' in df.columns:
        time_col = 'Hour'
        time_label = 'Hour of Day'
        max_time = 24
    elif 'Time' in df.columns:
        time_col = 'Time'
        time_label = 'Time (seconds)'
        max_time = df['Time'].max()
    else:
        raise KeyError("DataFrame must contain either 'Hour' or 'Time' column")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Transaction Count by {time_label}',
            f'Fraud Rate by {time_label}'
        ),
        vertical_spacing=0.15,
    )

    # Calculate transaction counts and fraud rate by time period
    time_stats = df.groupby(time_col)['Class'].agg([
        ('total', 'count'),
        ('fraud', 'sum'),
        ('fraud_rate', lambda x: 100 * x.sum() / len(x) if len(x) > 0 else 0)
    ]).reset_index()

    # Add transaction count trace
    fig.add_trace(
        go.Bar(
            x=time_stats[time_col],
            y=time_stats['total'],
            name='Total Transactions',
            marker_color=LEGIT_COLOR,
            showlegend=False,
        ),
        row=1, col=1
    )

    # Add fraud count trace
    fig.add_trace(
        go.Bar(
            x=time_stats[time_col],
            y=time_stats['fraud'],
            name='Fraudulent',
            marker_color=FRAUD_COLOR,
            showlegend=False,
        ),
        row=1, col=1
    )

    # Add fraud rate trace
    fig.add_trace(
        go.Scatter(
            x=time_stats[time_col],
            y=time_stats['fraud_rate'],
            mode='lines+markers',
            name='Fraud Rate (%)',
            line=dict(color=FRAUD_COLOR, width=2),
            marker=dict(size=4),
            showlegend=False,
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Temporal Fraud Patterns</b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
    )

    # Update x-axes
    fig.update_xaxes(title_text=time_label, row=1, col=1)
    fig.update_xaxes(title_text=time_label, row=2, col=1)

    # Update y-axes
    fig.update_yaxes(title_text='Transaction Count', row=1, col=1)
    fig.update_yaxes(title_text='Fraud Rate (%)', row=2, col=1)

    return fig


def plot_pca_scatter(
    df: pd.DataFrame,
    n_components: int = 2,
    sample_size: Optional[int] = None
) -> go.Figure:
    """
    Create a PCA scatter plot visualizing data separation by class.

    Applies PCA transformation to reduce dimensionality and creates
    an interactive scatter plot colored by transaction class.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset. Must contain V-features and 'Class'.
    n_components : int, optional
        Number of PCA components to compute (default: 2).
    sample_size : int, optional
        If provided, randomly samples this many points for visualization
        to improve performance (default: None).

    Returns
    -------
    go.Figure
        Plotly scatter plot showing PCA-transformed data.

    Raises
    ------
    ValueError
        If n_components is invalid or insufficient features exist.
    KeyError
        If required columns are missing.

    Examples
    --------
    >>> fig = plot_pca_scatter(df, sample_size=5000)
    >>> fig.show()
    """
    # Validate n_components
    if n_components < 2:
        raise ValueError("n_components must be at least 2")

    # Get V-features
    v_columns = sorted([col for col in df.columns if col.startswith('V')])

    if len(v_columns) < n_components:
        raise ValueError(
            f"Need at least {n_components} V-features, found {len(v_columns)}"
        )

    if 'Class' not in df.columns:
        raise KeyError("DataFrame must contain 'Class' column")

    # Sample data if requested
    if sample_size and sample_size < len(df):
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df

    # Prepare features
    X = df_plot[v_columns].values
    y = df_plot['Class'].values

    # Handle missing values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Create scatter plot
    fig = go.Figure()

    # Add legitimate transactions
    legit_mask = y == 0
    fig.add_trace(go.Scatter(
        x=X_pca[legit_mask, 0],
        y=X_pca[legit_mask, 1],
        mode='markers',
        name='Legitimate',
        marker=dict(
            color=LEGIT_COLOR,
            size=5,
            opacity=0.6,
        ),
    ))

    # Add fraudulent transactions
    fraud_mask = y == 1
    fig.add_trace(go.Scatter(
        x=X_pca[fraud_mask, 0],
        y=X_pca[fraud_mask, 1],
        mode='markers',
        name='Fraud',
        marker=dict(
            color=FRAUD_COLOR,
            size=5,
            opacity=0.8,
        ),
    ))

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_ * 100

    # Update layout
    fig.update_layout(
        title={
            'text': (
                f'<b>PCA Scatter Plot</b><br>'
                f'Explained Variance: PC1={explained_variance[0]:.1f}%, '
                f'PC2={explained_variance[1]:.1f}%'
            ),
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=f'Principal Component 1 ({explained_variance[0]:.1f}%)',
        yaxis_title=f'Principal Component 2 ({explained_variance[1]:.1f}%)',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top'),
        hovermode='closest',
    )

    return fig
