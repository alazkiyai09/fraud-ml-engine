"""
Data loading and preprocessing utilities.

Handles loading fraud detection datasets and preprocessing them
for the imbalanced classification benchmark.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.classical.legacy.config import config


def load_data(filepath: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load fraud detection data from CSV file.

    Expected format:
    - CSV file with features and target column
    - Target column should be named 'is_fraud' or 'Class' or be the last column

    Args:
        filepath: Path to the CSV file

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df = pd.read_csv(filepath)

    # Detect target column
    if "is_fraud" in df.columns:
        target_col = "is_fraud"
    elif "Class" in df.columns:
        target_col = "Class"
    else:
        # Assume last column is target
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target column: {target_col}")
    print(f"Fraud rate: {y.mean():.4f} ({y.mean()*100:.2f}%)")

    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    scale_features: bool = True,
    drop_na: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for modeling.

    Args:
        X: Feature DataFrame
        y: Target Series
        scale_features: Whether to apply StandardScaler
        drop_na: Whether to drop rows with missing values

    Returns:
        Tuple of (feature array, target array)
    """
    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values

    # Drop missing values if requested
    if drop_na:
        mask = ~np.isnan(X_array).any(axis=1)
        X_array = X_array[mask]
        y_array = y_array[mask]
        print(f"Dropped {X.shape[0] - X_array.shape[0]} rows with missing values")

    # Scale features
    if scale_features:
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X_array)
        print("Features scaled using StandardScaler")

    return X_array, y_array


def generate_synthetic_fraud_data(
    n_samples: int = 100000,
    n_features: int = 20,
    fraud_rate: float = 0.0017,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fraud detection dataset for testing.

    Creates a dataset with similar characteristics to real fraud data:
    - Highly imbalanced (default 0.17% fraud)
    - Mix of continuous and categorical-like features
    - Some predictive power

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        fraud_rate: Target fraud rate
        random_state: Random seed

    Returns:
        Tuple of (feature array, target array)
    """
    if random_state is None:
        random_state = config.RANDOM_STATE

    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate target with some relationship to features
    # First 5 features have predictive power
    log_odds = (
        X[:, 0] * 0.5
        + X[:, 1] * 0.3
        + X[:, 2] * 0.2
        + X[:, 3] * 0.4
        + X[:, 4] * 0.1
        - 2.0
    )

    # Convert to probabilities
    proba = 1 / (1 + np.exp(-log_odds))

    # Adjust to match target fraud rate
    threshold = np.percentile(proba, (1 - fraud_rate) * 100)
    y = (proba > threshold).astype(int)

    print(f"Generated synthetic data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Fraud rate: {y.mean():.4f} ({y.mean()*100:.2f}%)")

    return X, y


def load_or_generate_data(
    filepath: Optional[Path] = None,
    generate_if_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from file or generate synthetic data.

    Args:
        filepath: Path to data file (optional)
        generate_if_missing: Whether to generate synthetic data if file not found

    Returns:
        Tuple of (feature array, target array)
    """
    if filepath is not None and filepath.exists():
        print(f"Loading data from {filepath}...")
        X_df, y_series = load_data(filepath)
        X, y = preprocess_data(X_df, y_series)
        return X, y

    if generate_if_missing:
        print("No data file provided. Generating synthetic fraud data...")
        return generate_synthetic_fraud_data()

    raise FileNotFoundError(
        f"Data file not found: {filepath}. "
        "Set generate_if_missing=True to generate synthetic data."
    )
