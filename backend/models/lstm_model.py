"""
LSTM Model Architecture for Fall Detection

Two-layer LSTM network for temporal pattern recognition.
"""

import torch
import torch.nn as nn
from typing import Tuple


class FallDetectionLSTM(nn.Module):
    """
    Bidirectional LSTM model for fall detection.
    
    Architecture:
        Input: (batch, time_steps=30, features=7)
        ↓
        LSTM(128 units) + Dropout(0.3)
        ↓
        LSTM(64 units) + Dropout(0.3)
        ↓
        Dense(1) + Sigmoid
        ↓
        Output: Fall probability
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Feature dimension (default: 7)
            hidden_dim1: First LSTM hidden dimension
            hidden_dim2: Second LSTM hidden dimension
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM (doubles computation)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.bidirectional = bidirectional
        
        # Multiplier for bidirectional
        self.multiplier = 2 if bidirectional else 1
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1 * self.multiplier,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim2 * self.multiplier, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time_steps, features)
        
        Returns:
            Output tensor of shape (batch, 1) with fall probabilities
        """
        # First LSTM layer
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Second LSTM layer
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Use last time step
        out = out[:, -1, :]
        
        # Output layer
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_dim": self.input_dim,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
            "bidirectional": self.bidirectional,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/F1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def create_model(
    input_dim: int = 7,
    hidden_dim1: int = 128,
    hidden_dim2: int = 64,
    dropout: float = 0.3,
    bidirectional: bool = False,
    device: str = 'cuda'
) -> FallDetectionLSTM:
    """
    Create and initialize fall detection model.
    
    Args:
        input_dim: Feature dimension
        hidden_dim1: First LSTM hidden dimension
        hidden_dim2: Second LSTM hidden dimension
        dropout: Dropout rate
        bidirectional: Use bidirectional LSTM
        device: Device to put model on
    
    Returns:
        Initialized model
    """
    model = FallDetectionLSTM(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model(device='cpu')
    
    # Print model info
    info = model.get_model_info()
    print("="*50)
    print("LSTM Model Architecture")
    print("="*50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("="*50)
    
    # Test forward pass
    batch_size = 4
    time_steps = 30
    features = 7
    
    x = torch.randn(batch_size, time_steps, features)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
