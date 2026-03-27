"""
Data preprocessing for sequential fraud detection.

Handles sequence creation, temporal splitting, and feature scaling.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_user_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str],
    user_col: str = "user_id",
    time_col: str = "transaction_time",
    label_col: str = "is_fraud",
    min_sequence_length: int = 3
) -> Dict[str, np.ndarray]:
    """
    Create variable-length sequences for each user.

    Args:
        df: Transaction dataframe
        sequence_length: Maximum number of historical transactions per sequence
        feature_columns: List of feature column names
        user_col: Column name for user identifier
        time_col: Column name for timestamp
        label_col: Column name for fraud label
        min_sequence_length: Minimum sequence length (sequences shorter are dropped)

    Returns:
        Dictionary with keys:
            - 'sequences': np.ndarray of shape (num_sequences, max_seq_len, num_features)
            - 'labels': np.ndarray of shape (num_sequences,)
            - 'lengths': np.ndarray of actual sequence lengths
            - 'user_ids': List of user identifiers
    """
    # Ensure data is sorted by time
    df = df.sort_values([user_col, time_col]).copy()

    sequences = []
    labels = []
    lengths = []
    user_ids = []

    # Group by user
    for user_id, user_df in df.groupby(user_col):
        user_df = user_df.sort_values(time_col)

        # Need at least min_sequence_length transactions
        if len(user_df) < min_sequence_length:
            continue

        # Extract features and labels
        user_features = user_df[feature_columns].values
        user_labels = user_df[label_col].values

        # Create sequences with sliding window
        # For transaction t, use transactions [t-seq_len+1, t] to predict
        for i in range(min_sequence_length - 1, len(user_df)):
            # Determine start index (handle beginning of history)
            start_idx = max(0, i - sequence_length + 1)
            actual_length = i - start_idx + 1

            # Extract sequence
            seq = user_features[start_idx:i+1]

            # Pad if necessary to reach sequence_length
            if actual_length < sequence_length:
                padding = np.zeros((sequence_length - actual_length, len(feature_columns)))
                seq = np.vstack([padding, seq])

            sequences.append(seq)
            labels.append(user_labels[i])
            lengths.append(actual_length)
            user_ids.append(user_id)

    return {
        'sequences': np.array(sequences),
        'labels': np.array(labels),
        'lengths': np.array(lengths),
        'user_ids': user_ids
    }


def temporal_split(
    sequences: Dict[str, np.ndarray],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    time_col: Optional[str] = None,
    timestamps: Optional[np.ndarray] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Perform temporal train/validation/test split.

    IMPORTANT: Splits by sequence index to maintain temporal order.
    No data leakage between splits.

    Args:
        sequences: Dictionary from create_user_sequences
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        time_col: Not used, kept for API compatibility
        timestamps: Not used, kept for API compatibility

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    num_sequences = len(sequences['sequences'])

    # Calculate split indices
    train_end = int(num_sequences * train_ratio)
    val_end = int(num_sequences * (train_ratio + val_ratio))

    # Split indices (temporal - no shuffling!)
    train_indices = slice(0, train_end)
    val_indices = slice(train_end, val_end)
    test_indices = slice(val_end, num_sequences)

    # Create split dictionaries
    train_data = {k: v[train_indices] for k, v in sequences.items()}
    val_data = {k: v[val_indices] for k, v in sequences.items()}
    test_data = {k: v[test_indices] for k, v in sequences.items()}

    return train_data, val_data, test_data


def scale_features(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict
) -> Tuple[Dict, Dict, Dict, StandardScaler]:
    """
    Scale features using StandardScaler fit on training data only.

    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary

    Returns:
        Tuple of (scaled_train, scaled_val, scaled_test, scaler)
    """
    # Flatten sequences for fitting scaler: (num_samples * seq_len, num_features)
    train_sequences_flat = train_data['sequences'].reshape(-1, train_data['sequences'].shape[-1])

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_sequences_flat)

    # Scale each split
    def scale_split(data: Dict) -> Dict:
        sequences = data['sequences']
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = scaler.transform(sequences_flat)
        return {
            **data,
            'sequences': sequences_scaled.reshape(original_shape)
        }

    train_scaled = scale_split(train_data)
    val_scaled = scale_split(val_data)
    test_scaled = scale_split(test_data)

    return train_scaled, val_scaled, test_scaled, scaler


def compute_class_weights(
    labels: np.ndarray,
    method: str = "inverse"
) -> np.ndarray:
    """
    Compute class weights for imbalanced data.

    Args:
        labels: Array of binary labels (0 or 1)
        method: "inverse" for 1/frequency, "balanced" for sklearn balanced

    Returns:
        Array of shape (2,) with weights for [class_0, class_1]
    """
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)
    total = len(labels)

    if method == "inverse":
        # Simple inverse frequency weighting
        weight_pos = total / (2 * pos_count) if pos_count > 0 else 1.0
        weight_neg = total / (2 * neg_count) if neg_count > 0 else 1.0
    elif method == "balanced":
        # sklearn balanced mode
        weight_pos = total / (2 * pos_count) if pos_count > 0 else 1.0
        weight_neg = total / (2 * neg_count) if neg_count > 0 else 1.0
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.array([weight_neg, weight_pos])


def prepare_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    config: Dict
) -> Tuple[Dict, Dict, Dict, StandardScaler, np.ndarray]:
    """
    End-to-end data preparation pipeline.

    Args:
        df: Raw transaction dataframe
        feature_columns: List of feature column names
        config: Configuration dictionary

    Returns:
        Tuple of (train_data, val_data, test_data, scaler, class_weights)
    """
    # Create sequences
    sequences = create_user_sequences(
        df=df,
        sequence_length=config['sequence']['max_sequence_length'],
        feature_columns=feature_columns,
        min_sequence_length=config['sequence']['min_sequence_length']
    )

    print(f"Created {len(sequences['sequences'])} sequences")
    print(f"Positive class ratio: {np.mean(sequences['labels']):.4f}")

    # Temporal split
    train_data, val_data, test_data = temporal_split(
        sequences,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio']
    )

    print(f"Train: {len(train_data['sequences'])}, "
          f"Val: {len(val_data['sequences'])}, "
          f"Test: {len(test_data['sequences'])}")

    # Scale features
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_data, val_data, test_data
    )

    # Compute class weights
    if config['class_weights']['enabled']:
        class_weights = compute_class_weights(
            train_scaled['labels'],
            method=config['class_weights']['method']
        )
        print(f"Class weights: {class_weights}")
    else:
        class_weights = np.array([1.0, 1.0])

    return train_scaled, val_scaled, test_scaled, scaler, class_weights
