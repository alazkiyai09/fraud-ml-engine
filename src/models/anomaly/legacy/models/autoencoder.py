"""Autoencoder implementation for anomaly detection using PyTorch."""

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import AnomalyDetector


class Autoencoder(nn.Module):
    """PyTorch Autoencoder model."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        """
        Initialize Autoencoder architecture.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions for encoder.
            latent_dim: Dimension of latent representation.
        """
        super(Autoencoder, self).__init__()

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor.

        Returns:
            Reconstructed tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder based anomaly detector using reconstruction error."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 16,
        contamination: float = 0.1,
        device: str = "cuda"
    ):
        """
        Initialize Autoencoder detector.

        Args:
            input_dim: Number of input features.
            hidden_dims: Hidden layer dimensions for encoder.
            latent_dim: Latent dimension.
            contamination: Expected proportion of outliers.
            device: Device for training ('cuda' or 'cpu').
        """
        super().__init__(contamination=contamination)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = Autoencoder(input_dim, hidden_dims, latent_dim).to(self.device)
        self.training_losses = []

    def fit(
        self,
        X: np.ndarray,  # type: ignore[valid-type]
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> None:
        """
        Train Autoencoder on training data.

        IMPORTANT: Should only be trained on Class=0 (legitimate) data.

        Args:
            X: Training data of shape (n_samples, n_features).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            early_stopping_patience: Patience for early stopping.
            verbose: Whether to print training progress.
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)  # Input = Target for reconstruction
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, _ in dataloader:
                # Forward pass
                reconstructions = self.model(batch_X)
                loss = criterion(reconstructions, batch_X)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.training_losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_fitted = True

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:  # type: ignore[valid-type]
        """
        Compute anomaly scores based on reconstruction error.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
            Higher scores indicate higher reconstruction error (more anomalous).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructions = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1)

        return reconstruction_errors.cpu().numpy()

    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'latent_dim': self.latent_dim,
            'contamination': self.contamination
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.latent_dim = checkpoint['latent_dim']
        self.contamination = checkpoint['contamination']
        self.is_fitted = True
        print(f"Model loaded from {path}")
