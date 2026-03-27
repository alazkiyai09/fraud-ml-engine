"""Data preprocessing utilities for anomaly detection benchmark."""

from typing import Tuple, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


def load_data(
    data_path: str,
    file_type: str = "csv"
) -> pd.DataFrame:
    """
    Load data from file.

    Args:
        data_path: Path to data file.
        file_type: Type of file ('csv', 'parquet', 'pickle').

    Returns:
        Loaded DataFrame.
    """
    path = Path(data_path)
    if file_type == "csv":
        return pd.read_csv(path)
    elif file_type == "parquet":
        return pd.read_parquet(path)
    elif file_type == "pickle":
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def split_data_by_class(
    df: pd.DataFrame,
    label_column: str = "class",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train (class 0 only), validation, and test sets.

    IMPORTANT: Training set contains ONLY class 0 (legitimate) samples.
    Validation and test sets contain both classes.

    Args:
        df: Input DataFrame with features and labels.
        label_column: Name of the label column.
        test_size: Proportion of data for test set.
        val_size: Proportion of data for validation set (from remaining after test).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_test) as DataFrames.
        Note: y_train is not returned as train set is class 0 only.
    """
    # Separate features and labels
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: separate validation from remaining
    val_size_adjusted = val_size / (1 - test_size)
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    # Combine train and validation for training data, KEEP ONLY CLASS 0
    train_df = pd.concat([X_train_val, y_train_val], axis=1)
    train_df_class0 = train_df[train_df[label_column] == 0]

    X_train = train_df_class0.drop(columns=[label_column])

    # Prepare validation and test DataFrames
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"Data split summary:")
    print(f"  Train (class 0 only): {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples (class 0: {sum(y_val==0)}, class 1: {sum(y_val==1)})")
    print(f"  Test: {len(X_test)} samples (class 0: {sum(y_test==0)}, class 1: {sum(y_test==1)})")

    return X_train, X_val, X_test, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "standard",
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:  # type: ignore[valid-type]
    """
    Scale features using fitted scaler on training data only.

    Args:
        X_train: Training features (class 0 only).
        X_val: Validation features.
        X_test: Test features.
        scaler_type: Type of scaler ('standard' or 'minmax').
        save_path: Path to save fitted scaler.

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler).
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    # Fit on training data ONLY
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to {save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def load_and_split_data(
    data_path: str,
    label_column: str = "class",
    test_size: float = 0.2,
    val_size: float = 0.1,
    scaler_type: str = "standard",
    random_state: int = 42,
    file_type: str = "csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[valid-type]
    """
    Complete pipeline: load data, split, and scale.

    Args:
        data_path: Path to data file.
        label_column: Name of the label column.
        test_size: Proportion of data for test set.
        val_size: Proportion of data for validation set.
        scaler_type: Type of scaler ('standard' or 'minmax').
        random_state: Random seed.
        file_type: Type of file ('csv', 'parquet', 'pickle').

    Returns:
        Tuple of (X_train, X_val, X_test, y_val, y_test, feature_names).
        Note: Training data is class 0 only.
    """
    # Load data
    df = load_data(data_path, file_type)

    # Split data
    X_train, X_val, X_test, y_test = split_data_by_class(
        df, label_column, test_size, val_size, random_state
    )

    # Get labels for validation set
    y_val = X_val[label_column].values if label_column in X_val.columns else \
            pd.concat([X_val, y_test], axis=1)[label_column].iloc[:len(X_val)].values

    # Remove label column from features
    if label_column in X_train.columns:
        X_train = X_train.drop(columns=[label_column])
    if label_column in X_val.columns:
        X_val = X_val.drop(columns=[label_column])
    if label_column in X_test.columns:
        X_test = X_test.drop(columns=[label_column])

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test, scaler_type
    )

    feature_names = X_train.columns.tolist()

    return X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test, feature_names


def save_splits(
    X_train: np.ndarray,  # type: ignore[valid-type]
    X_val: np.ndarray,  # type: ignore[valid-type]
    X_test: np.ndarray,  # type: ignore[valid-type]
    y_val: np.ndarray,  # type: ignore[valid-type]
    y_test: np.ndarray,  # type: ignore[valid-type]
    output_dir: str
) -> None:
    """
    Save data splits to disk.

    Args:
        X_train: Training features (class 0 only).
        X_val: Validation features.
        X_test: Test features.
        y_val: Validation labels.
        y_test: Test labels.
        output_dir: Directory to save splits.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_val.npy", y_val)
    np.save(output_path / "y_test.npy", y_test)

    print(f"Data splits saved to {output_dir}")


def load_splits(
    data_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[valid-type]
    """
    Load data splits from disk.

    Args:
        data_dir: Directory containing saved splits.

    Returns:
        Tuple of (X_train, X_val, X_test, y_val, y_test).
    """
    data_path = Path(data_dir)

    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")

    return X_train, X_val, X_test, y_val, y_test
