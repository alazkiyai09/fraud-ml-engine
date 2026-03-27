"""
PyTorch Dataset for sequential fraud detection.

Handles variable-length sequences with pack_padded_sequence.
"""

from typing import Dict, Tuple, List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FraudSequenceDataset(Dataset):
    """
    Dataset for fraud detection with variable-length sequences.

    Each item returns:
        - sequence: Tensor of shape (seq_len, num_features)
        - label: Tensor of shape (1,)
        - length: Actual sequence length (for packing)
    """

    def __init__(self, data: Dict[str, torch.Tensor]):
        """
        Args:
            data: Dictionary with keys:
                - 'sequences': np.ndarray of shape (N, max_seq_len, num_features)
                - 'labels': np.ndarray of shape (N,)
                - 'lengths': np.ndarray of shape (N,)
        """
        self.sequences = torch.tensor(data['sequences'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        self.lengths = torch.tensor(data['lengths'], dtype=torch.long)

        assert len(self.sequences) == len(self.labels) == len(self.lengths), \
            "Mismatch in data dimensions"

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sequence-label pair.

        Returns:
            sequence: (max_seq_len, num_features)
            label: scalar tensor
            length: actual sequence length (int)
        """
        return self.sequences[idx], self.labels[idx], self.lengths[idx].item()


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader with pack_padded_sequence.

    This function:
    1. Sorts batch by sequence length (descending) - required for pack_padded_sequence
    2. Pads sequences to the same length
    3. Returns padded sequences, labels, and lengths

    Args:
        batch: List of (sequence, label, length) tuples

    Returns:
        padded_sequences: (batch_size, max_seq_len, num_features)
        labels: (batch_size,)
        lengths: (batch_size,) - sorted in descending order
    """
    # Sort batch by sequence length (descending)
    sorted_batch = sorted(batch, key=lambda x: x[2], reverse=True)

    # Extract sequences, labels, and lengths
    sequences = torch.stack([item[0] for item in sorted_batch])
    labels = torch.tensor([item[1] for item in sorted_batch], dtype=torch.float32)
    lengths = torch.tensor([item[2] for item in sorted_batch], dtype=torch.long)

    # sequences is already padded (all have same max_seq_len from preprocessing)
    # Shape: (batch_size, max_seq_len, num_features)

    return sequences, labels, lengths


def create_packed_sequence(
    padded_sequences: torch.Tensor,
    lengths: torch.Tensor
) -> torch.nn.utils.rnn.PackedSequence:
    """
    Create a PackedSequence from padded sequences.

    Args:
        padded_sequences: (batch_size, max_seq_len, num_features)
        lengths: (batch_size,) - actual lengths (must be sorted descending)

    Returns:
        PackedSequence for efficient LSTM processing
    """
    # Ensure sequences are sorted by length (descending)
    # This should already be done by collate_fn
    batch_size = padded_sequences.size(0)

    # Transpose to (max_seq_len, batch_size, num_features) for pack_padded_sequence
    padded_sequences = padded_sequences.transpose(0, 1)

    # Create packed sequence
    packed = pack_padded_sequence(
        padded_sequences,
        lengths.cpu(),
        batch_first=False,
        enforce_sorted=True
    )

    return packed


def unpack_sequence(
    packed_sequence: torch.nn.utils.rnn.PackedSequence,
    max_length: int
) -> torch.Tensor:
    """
    Unpack a PackedSequence back to padded format.

    Args:
        packed_sequence: PackedSequence from LSTM output
        max_length: Maximum sequence length for padding

    Returns:
        Unpacked tensor of shape (batch_size, max_length, hidden_dim)
    """
    # pad_packed_sequence returns (padded_output, lengths)
    padded_output, _ = pad_packed_sequence(
        packed_sequence,
        batch_first=False,
        total_length=max_length
    )

    # Transpose back to (batch_size, max_length, hidden_dim)
    return padded_output.transpose(0, 1)
