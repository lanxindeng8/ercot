"""
LSTM Model for Wind Forecasting

Sequence-to-sequence LSTM with quantile outputs.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseWindForecastModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Args:
        X: Features array (n_samples, n_features)
        y: Target array (n_samples,)
        seq_length: Length of input sequences
        horizon: Forecast horizon

    Returns:
        Tuple of (X_seq, y_seq)
    """
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length - horizon + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length + horizon - 1])

    return np.array(X_seq), np.array(y_seq)


# Define placeholder classes for when torch is not available
QuantileLoss = None
LSTMNetwork = None

if TORCH_AVAILABLE:
    class QuantileLoss(nn.Module):
        """Quantile loss (pinball loss) for quantile regression."""

        def __init__(self, quantiles: List[float]):
            super().__init__()
            self.quantiles = quantiles

        def forward(self, y_pred, y_true):
            """
            Compute quantile loss.

            Args:
                y_pred: Predictions (batch, n_quantiles)
                y_true: Targets (batch,)

            Returns:
                Total quantile loss
            """
            losses = []
            y_true = y_true.unsqueeze(-1)

            for i, q in enumerate(self.quantiles):
                errors = y_true - y_pred[:, i:i+1]
                loss = torch.max(q * errors, (q - 1) * errors)
                losses.append(loss)

            return torch.mean(torch.cat(losses, dim=1))

    class LSTMNetwork(nn.Module):
        """LSTM network with multi-quantile output."""

        def __init__(
            self,
            n_features: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            n_quantiles: int = 3,
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_quantiles),
            )

        def forward(self, x):
            """
            Forward pass.

            Args:
                x: Input tensor (batch, seq_len, n_features)

            Returns:
                Output tensor (batch, n_quantiles)
            """
            lstm_out, _ = self.lstm(x)
            # Use last hidden state
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)


class LSTMWindModel(BaseWindForecastModel):
    """
    LSTM model for wind generation forecasting.

    Uses sequence-to-sequence learning with quantile outputs.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        seq_length: int = 24,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        """
        Initialize LSTM model.

        Args:
            quantiles: Quantile levels to predict
            seq_length: Input sequence length (hours)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.quantiles = sorted(quantiles)
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[LSTMNetwork] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-8

        return (X - self.feature_means) / self.feature_stds

    def _normalize_target(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize target."""
        if fit:
            self.target_mean = np.mean(y)
            self.target_std = np.std(y) + 1e-8

        return (y - self.target_mean) / self.target_std

    def _denormalize_target(self, y: np.ndarray) -> np.ndarray:
        """Denormalize target."""
        return y * self.target_std + self.target_mean

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> 'LSTMWindModel':
        """
        Train LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)
        n_features = len(self.feature_names)

        # Convert to numpy
        X_train_np = X_train.values
        y_train_np = y_train.values

        # Normalize
        X_train_norm = self._normalize_features(X_train_np, fit=True)
        y_train_norm = self._normalize_target(y_train_np, fit=True)

        # Create sequences
        X_seq, y_seq = create_sequences(X_train_norm, y_train_norm, self.seq_length)

        logger.info(f"Training sequences: {X_seq.shape}")

        # Create tensors
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = self._normalize_features(X_val.values)
            y_val_norm = self._normalize_target(y_val.values)
            X_val_seq, y_val_seq = create_sequences(X_val_norm, y_val_norm, self.seq_length)

            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq),
                    torch.FloatTensor(y_val_seq)
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize model
        self.model = LSTMNetwork(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_quantiles=len(self.quantiles),
        ).to(self.device)

        # Loss and optimizer
        criterion = QuantileLoss(self.quantiles)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        y_pred = self.model(X_batch)
                        val_loss += criterion(y_pred, y_batch).item()

                val_loss /= len(val_loader)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point forecast (median).

        Args:
            X: Features

        Returns:
            Point predictions (MW)
        """
        quantile_preds = self.predict_quantiles(X)
        return quantile_preds.get(0.5, list(quantile_preds.values())[0])

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """
        Generate quantile predictions.

        Args:
            X: Features
            quantiles: Quantile levels (default: all trained)

        Returns:
            Dict mapping quantile to predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")

        quantiles = quantiles or self.quantiles

        # Normalize features
        X_norm = self._normalize_features(X.values)

        # Need to handle sequence creation for prediction
        # For simplicity, pad with zeros if needed
        if len(X_norm) < self.seq_length:
            pad_size = self.seq_length - len(X_norm)
            X_norm = np.vstack([np.zeros((pad_size, X_norm.shape[1])), X_norm])

        # Create sequences for each position
        predictions = {q: [] for q in quantiles}

        self.model.eval()
        with torch.no_grad():
            for i in range(self.seq_length - 1, len(X_norm)):
                seq = X_norm[i - self.seq_length + 1:i + 1]
                X_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)

                output = self.model(X_tensor).cpu().numpy()[0]

                for j, q in enumerate(self.quantiles):
                    if q in quantiles:
                        # Denormalize
                        pred = output[j] * self.target_std + self.target_mean
                        predictions[q].append(pred)

        # Convert to arrays
        return {q: np.array(preds) for q, preds in predictions.items()}

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path (will save as .pt)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        state = {
            'model_state': self.model.state_dict(),
            'quantiles': self.quantiles,
            'seq_length': self.seq_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'feature_names': self.feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
        }

        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'LSTMWindModel':
        """
        Load model from disk.

        Args:
            path: File path

        Returns:
            Loaded model
        """
        state = torch.load(path, map_location='cpu')

        model = cls(
            quantiles=state['quantiles'],
            seq_length=state['seq_length'],
            hidden_dim=state['hidden_dim'],
            num_layers=state['num_layers'],
            dropout=state['dropout'],
        )

        model.feature_names = state['feature_names']
        model.feature_means = state['feature_means']
        model.feature_stds = state['feature_stds']
        model.target_mean = state['target_mean']
        model.target_std = state['target_std']

        # Reconstruct model
        n_features = len(model.feature_names)
        model.model = LSTMNetwork(
            n_features=n_features,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
            dropout=model.dropout,
            n_quantiles=len(model.quantiles),
        ).to(model.device)

        model.model.load_state_dict(state['model_state'])
        model.model.eval()

        logger.info(f"Model loaded from {path}")
        return model

    @property
    def name(self) -> str:
        return "LSTM"
