"""
Focal Loss implementation in PyTorch for imbalanced classification.

Focal Loss focuses training on hard examples by down-weighting easy examples.
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from src.models.classical.legacy.config import config


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter, higher increases down-weighting of easy examples
               (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: float = None,
        gamma: float = None,
        reduction: str = "mean",
    ):
        super().__init__()

        if alpha is None:
            alpha = config.FOCAL_ALPHA
        if gamma is None:
            gamma = config.FOCAL_GAMMA

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss with numerical stability.

        Args:
            inputs: Raw logits from model (shape: [batch_size, 1] or [batch_size])
            targets: Ground truth labels (0 or 1, shape: [batch_size])

        Returns:
            Focal loss value
        """
        # Ensure inputs have correct shape
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)

        # Calculate binary cross entropy with logits for numerical stability
        # bce_loss = -[y * log(sigma(x)) + (1-y) * log(1-sigma(x))]
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.unsqueeze(1).float(), reduction="none"
        )

        # Calculate probability with numerical stability
        # p_t = p if y == 1 else 1-p
        p_t = torch.sigmoid(inputs)
        p_t = torch.where(
            targets.unsqueeze(1) == 1, p_t, 1 - p_t
        )

        # Calculate alpha_t
        alpha_t = torch.where(targets.unsqueeze(1) == 1, self.alpha, 1 - self.alpha)

        # Focal loss: alpha_t * (1 - p_t)^gamma * BCE
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss.squeeze(1)


class FocalLossClassifier:
    """
    Neural network classifier with Focal Loss.

    Implements a simple feedforward network with Focal Loss
    for imbalanced binary classification.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        alpha: Alpha parameter for Focal Loss
        gamma: Gamma parameter for Focal Loss
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        alpha: float = None,
        gamma: float = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 256,
        random_state: int = None,
    ):
        if random_state is None:
            random_state = config.RANDOM_STATE

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.alpha = alpha if alpha is not None else config.FOCAL_ALPHA
        self.gamma = gamma if gamma is not None else config.FOCAL_GAMMA
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers).to(self.device)

        self.criterion = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def _create_data_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader."""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FocalLossClassifier":
        """
        Train the neural network with Focal Loss.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        train_loader = self._create_data_loader(X, y, shuffle=True)

        self.network.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.network(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates.

        Args:
            X: Test features

        Returns:
            Probability estimates for both classes (n_samples, 2)
        """
        self.network.eval()

        # Process in batches to avoid memory issues
        probabilities = []
        batch_size = self.batch_size

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
                outputs = self.network(batch_X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                probabilities.append(probs)

        all_probs = np.vstack(probabilities)

        # Convert to [n_samples, 2] format (class 0, class 1)
        proba = np.column_stack([1 - all_probs, all_probs])

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Test features

        Returns:
            Predicted binary labels
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
