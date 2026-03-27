"""
Main application module for the Fraud Detection EDA Dashboard.

This module initializes and configures the Dash application,
loads data, and sets up the server.
"""

from pathlib import Path
from typing import Optional

import dash
from dash import html
import pandas as pd

from src.eda.dashboard.callbacks import register_callbacks
from src.eda.dashboard.data_loader import load_fraud_data, preprocess_data
from src.eda.dashboard.layout import create_dashboard_layout
from src.eda.dashboard.utils import calculate_summary_statistics


# Default data path
DEFAULT_DATA_PATH = "data/creditcard.csv"


def create_app(data_path: Optional[str] = None) -> dash.Dash:
    """
    Create and configure the Dash application.

    Initializes the Dash app, loads and preprocesses the fraud detection
    dataset, creates the layout, and registers all callbacks.

    Parameters
    ----------
    data_path : str, optional
        Path to the CSV file containing the fraud detection dataset.
        If None, uses DEFAULT_DATA_PATH.

    Returns
    -------
    dash.Dash
        Configured Dash application instance ready to run.

    Raises
    ------
    FileNotFoundError
        If the data file cannot be found.
    ValueError
        If the data file is invalid or cannot be processed.

    Examples
    --------
    >>> app = create_app('data/creditcard.csv')
    >>> app.run_server(debug=True)
    """
    # Use default path if not provided
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    # Initialize Dash app
    app = dash.Dash(
        __name__,
        title='Fraud Detection EDA Dashboard',
        suppress_callback_exceptions=True,
        update_title='Loading...',
    )

    # Load and preprocess data
    print(f"Loading data from: {data_path}")
    df = load_fraud_data(data_path, validate=True)
    print(f"Loaded {len(df)} transactions")

    df_processed = preprocess_data(df, normalize_time=True)
    print(f"Preprocessed data: {df_processed.shape}")

    # Calculate summary statistics
    stats = calculate_summary_statistics(df_processed)
    print(f"Summary: {stats['total_transactions']:,} transactions, "
          f"{stats['fraud_count']} fraud cases ({stats['fraud_percentage']}%)")

    # Create layout
    app.layout = html.Div([
        create_dashboard_layout(stats),
        html.Div(id='dummy-div', style={'display': 'none'})  # For storing data
    ])

    # Register callbacks with processed data
    register_callbacks(app, df_processed)

    print("Dashboard initialized successfully!")

    return app


def main(
    data_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = True
) -> None:
    """
    Run the Fraud Detection EDA Dashboard.

    Parameters
    ----------
    data_path : str, optional
        Path to the fraud detection dataset CSV file.
        If None, uses DEFAULT_DATA_PATH.
    host : str, optional
        Host address to bind the server to (default: "127.0.0.1").
    port : int, optional
        Port number to run the server on (default: 8050).
    debug : bool, optional
        Whether to run in debug mode with auto-reload (default: True).

    Returns
    -------
    None

    Examples
    --------
    >>> # Run with default settings
    >>> main()
    >>>
    >>> # Run with custom data path
    >>> main(data_path='path/to/data.csv')
    >>>
    >>> # Run in production mode
    >>> main(debug=False)
    """
    # Create app
    app = create_app(data_path)

    # Print server info
    print("\n" + "="*60)
    print("🚀 Starting Fraud Detection EDA Dashboard")
    print("="*60)
    print(f"Server URL: http://{host}:{port}")
    print(f"Debug mode: {debug}")
    print("="*60 + "\n")

    # Run server
    app.run_server(
        host=host,
        port=port,
        debug=debug,
        dev_tools_hot_reload=debug,
    )


def validate_data_path(data_path: str) -> bool:
    """
    Validate that the data file exists and is readable.

    Parameters
    ----------
    data_path : str
        Path to the data file to validate.

    Returns
    -------
    bool
        True if data file exists and is readable, False otherwise.

    Examples
    --------
    >>> validate_data_path('data/creditcard.csv')
    True
    >>> validate_data_path('nonexistent.csv')
    False
    """
    path = Path(data_path)

    if not path.exists():
        print(f"❌ Data file not found: {data_path}")
        return False

    if not path.is_file():
        print(f"❌ Path is not a file: {data_path}")
        return False

    # Check if file is readable
    try:
        with open(path, 'r') as f:
            f.readline()
        return True
    except Exception as e:
        print(f"❌ Cannot read file: {e}")
        return False


if __name__ == "__main__":
    # Check if data file exists
    if not validate_data_path(DEFAULT_DATA_PATH):
        print("\n" + "="*60)
        print("⚠️  Data file not found!")
        print("="*60)
        print(f"\nPlease download the Kaggle Credit Card Fraud Detection")
        print(f"dataset and place it at: {DEFAULT_DATA_PATH}")
        print(f"\nDownload from:")
        print(f"https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("="*60 + "\n")
        exit(1)

    # Run the app
    main()
