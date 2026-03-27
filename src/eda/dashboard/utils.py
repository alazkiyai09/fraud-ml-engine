"""
Utility functions and constants for the Fraud Detection EDA Dashboard.

This module provides color constants, summary statistics calculations,
and export functionality for the dashboard.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Color constants for visualizations
FRAUD_COLOR: str = "#FF6B6B"  # Red/coral for fraud transactions
LEGIT_COLOR: str = "#4ECDC4"  # Teal for legitimate transactions


def calculate_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate comprehensive summary statistics for the fraud dataset.

    Computes key metrics including transaction counts, fraud ratio,
    amount statistics, and feature counts.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset. Must contain 'Class' and 'Amount' columns.

    Returns
    -------
    dict[str, Any]
        Dictionary containing summary statistics with keys:
        - 'total_transactions': Total number of transactions
        - 'fraud_count': Number of fraudulent transactions
        - 'legit_count': Number of legitimate transactions
        - 'fraud_percentage': Percentage of fraudulent transactions
        - 'total_amount': Sum of all transaction amounts
        - 'avg_amount': Average transaction amount
        - 'avg_fraud_amount': Average fraud transaction amount
        - 'avg_legit_amount': Average legitimate transaction amount
        - 'feature_count': Number of features (columns)

    Raises
    ------
    KeyError
        If required columns ('Class', 'Amount') are missing from DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'Class': [0, 1, 0], 'Amount': [100, 50, 200]})
    >>> stats = calculate_summary_statistics(df)
    >>> stats['fraud_percentage']
    33.33
    """
    # Validate required columns exist
    required_columns = {'Class', 'Amount'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Separate fraud and legitimate transactions
    fraud_df = df[df['Class'] == 1]
    legit_df = df[df['Class'] == 0]

    # Calculate statistics
    total_transactions: int = len(df)
    fraud_count: int = len(fraud_df)
    legit_count: int = len(legit_df)

    summary_stats: dict[str, Any] = {
        'total_transactions': total_transactions,
        'fraud_count': fraud_count,
        'legit_count': legit_count,
        'fraud_percentage': round((fraud_count / total_transactions * 100) if total_transactions > 0 else 0, 2),
        'total_amount': float(df['Amount'].sum()),
        'avg_amount': float(df['Amount'].mean()),
        'avg_fraud_amount': float(fraud_df['Amount'].mean()) if len(fraud_df) > 0 else 0.0,
        'avg_legit_amount': float(legit_df['Amount'].mean()) if len(legit_df) > 0 else 0.0,
        'feature_count': len(df.columns) - 1,  # Exclude Class column
    }

    return summary_stats


def export_to_html(fig: go.Figure, filepath: str) -> None:
    """
    Export a Plotly figure to a standalone HTML file.

    Creates a self-contained HTML file that can be opened in any web browser
    without requiring a running Dash server.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to export.
    filepath : str
        Target file path for the HTML export. Will create parent directories
        if they don't exist.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If filepath is empty or invalid.
    IOError
        If unable to write to the specified filepath.

    Examples
    --------
    >>> fig = go.Figure(data=go.Bar(x=[1, 2], y=[3, 4]))
    >>> export_to_html(fig, 'outputs/dashboard.html')
    """
    if not filepath:
        raise ValueError("Filepath cannot be empty")

    # Create parent directories if they don't exist
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_html(
            str(output_path),
            include_plotlyjs=True,
            full_html=True,
            auto_open=False
        )
    except Exception as e:
        raise IOError(f"Failed to export figure to HTML: {e}")


def format_currency(value: float) -> str:
    """
    Format a numeric value as currency string.

    Parameters
    ----------
    value : float
        The numeric value to format.

    Returns
    -------
    str
        Formatted currency string (e.g., '$1,234.56').

    Examples
    --------
    >>> format_currency(1234.56)
    '$1,234.56'
    """
    return f"${value:,.2f}"


def format_number(value: int | float) -> str:
    """
    Format a numeric value with thousands separator.

    Parameters
    ----------
    value : int | float
        The numeric value to format.

    Returns
    -------
    str
        Formatted number string (e.g., '1,234.56').

    Examples
    --------
    >>> format_number(1234.56)
    '1,234.56'
    """
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.2f}"
