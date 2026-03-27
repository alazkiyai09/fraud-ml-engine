"""
Data loading and preprocessing module for the Fraud Detection EDA Dashboard.

This module handles loading the Kaggle Credit Card Fraud Detection dataset,
validating its structure, and performing necessary preprocessing steps.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Expected schema for Kaggle Credit Card Fraud Detection dataset
EXPECTED_COLUMNS = {
    'Time',  # Seconds elapsed between each transaction and first transaction
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',  # PCA-transformed features
    'Amount',  # Transaction amount
    'Class',  # 1 = fraud, 0 = legitimate
}


def load_fraud_data(
    filepath: str | Path,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load the credit card fraud detection dataset from CSV file.

    Reads the Kaggle Credit Card Fraud Detection dataset and optionally
    validates its structure against the expected schema.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file containing the fraud detection dataset.
        Expected format: Kaggle Credit Card Fraud Detection dataset structure.
    validate : bool, optional
        Whether to validate the dataset schema after loading (default: True).

    Returns
    -------
    pd.DataFrame
        Loaded fraud detection dataset with proper dtypes.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If the file cannot be parsed as CSV or validation fails.

    Examples
    --------
    >>> df = load_fraud_data('data/creditcard.csv')
    >>> len(df)
    284807
    """
    filepath = Path(filepath)

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            f"Please download the Kaggle Credit Card Fraud Detection dataset "
            f"and place it at: {filepath}"
        )

    # Load CSV file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Validate schema if requested
    if validate:
        validate_data(df)

    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the fraud detection dataset structure and content.

    Checks that the DataFrame contains all expected columns with proper
    data types and no missing values in critical fields.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If validation fails with descriptive error message.

    Examples
    --------
    >>> df = pd.DataFrame(columns=['Time', 'V1', 'Amount', 'Class'])
    >>> validate_data(df)
    True
    """
    # Check for required columns
    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Expected columns: {EXPECTED_COLUMNS}"
        )

    # Check data types for key columns
    if not pd.api.types.is_numeric_dtype(df['Class']):
        raise ValueError("'Class' column must be numeric")

    if not pd.api.types.is_numeric_dtype(df['Amount']):
        raise ValueError("'Amount' column must be numeric")

    # Check Class column contains only 0 and 1
    unique_classes = set(df['Class'].unique())
    invalid_classes = unique_classes - {0, 1}
    if invalid_classes:
        raise ValueError(
            f"'Class' column contains invalid values: {invalid_classes}. "
            f"Only 0 (legitimate) and 1 (fraud) are allowed."
        )

    # Check for missing values in critical columns
    critical_columns = {'Class', 'Amount'}
    for col in critical_columns:
        if df[col].isnull().any():
            raise ValueError(f"'{col}' column contains missing values")

    return True


def preprocess_data(
    df: pd.DataFrame,
    normalize_time: bool = True
) -> pd.DataFrame:
    """
    Preprocess the fraud detection dataset for analysis and visualization.

    Performs preprocessing steps including:
    - Converting Time from seconds to hours (optional)
    - Ensuring proper data types
    - Removing any rows with missing values
    - Creating derived features for analysis

    Parameters
    ----------
    df : pd.DataFrame
        Raw fraud detection dataset.
    normalize_time : bool, optional
        Whether to convert Time from seconds to hours since first transaction
        (default: True). Useful for time pattern analysis.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataset ready for visualization.

    Examples
    --------
    >>> df = load_fraud_data('data/creditcard.csv')
    >>> processed_df = preprocess_data(df)
    >>> 'Hour' in processed_df.columns
    True
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # Remove rows with any missing values
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna()
    removed_rows = initial_rows - len(df_processed)

    if removed_rows > 0:
        print(f"Warning: Removed {removed_rows} rows with missing values")

    # Ensure Class is integer type
    df_processed['Class'] = df_processed['Class'].astype(int)

    # Normalize Time if requested and column exists
    if normalize_time and 'Time' in df_processed.columns:
        # Convert seconds to hours (modulo 24 for hour of day)
        df_processed['Hour'] = (df_processed['Time'] // 3600) % 24

    # Add log-transformed Amount to handle skewness
    # Adding small constant to avoid log(0)
    df_processed['Log_Amount'] = np.log1p(df_processed['Amount'])

    return df_processed


def get_data_info(df: pd.DataFrame) -> dict[str, any]:
    """
    Get comprehensive information about the dataset.

    Provides metadata including shape, memory usage, column types,
    and basic statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The fraud detection dataset.

    Returns
    -------
    dict[str, any]
        Dictionary containing dataset information with keys:
        - 'shape': Tuple of (rows, columns)
        - 'memory_usage': Memory usage in MB
        - 'column_types': Dictionary of column names to dtypes
        - 'has_missing': Boolean indicating if any missing values exist
        - 'missing_count': Count of missing values per column

    Examples
    --------
    >>> df = load_fraud_data('data/creditcard.csv')
    >>> info = get_data_info(df)
    >>> info['shape']
    (284807, 31)
    """
    info: dict[str, any] = {
        'shape': df.shape,
        'memory_usage': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'column_types': df.dtypes.astype(str).to_dict(),
        'has_missing': df.isnull().any().any(),
        'missing_count': df.isnull().sum().to_dict(),
    }

    return info
