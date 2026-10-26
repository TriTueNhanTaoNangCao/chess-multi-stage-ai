"""
Stage 4: Convolutional Policy Network for Chess (SL Policy & Fast Rollout).
Architectural reference: Silver et al. (Nature 2016), Mark Liu (2025) Chapter 10 & 11.
"""
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from games.chess.encoder import NUM_ACTIONS, NUM_CHANNELS


class CNNPolicyNetwork(nn.Module):
    """
    Convolutional Neural Network predicting move probability distribution p = Pr(a|s).
    Input: (B, 14, 8, 8) float tensor in NCHW format.
    Output: (B, 4672) raw logits over AlphaZero action space.
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 128,
        fc_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "fc_dim": fc_dim,
            "dropout": dropout,
        }
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.dropout_rate = dropout

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim * 8 * 8, fc_dim)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(fc_dim, num_actions)

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration for checkpoint serialization."""
        return dict(self.config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CNNPolicyNetwork":
        """Instantiate network from configuration dictionary."""
        return cls(**config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw unnormalized logits.
        Args:
            x: Float tensor of shape (B, 14, 8, 8) in NCHW format.
        Returns:
            Raw logits of shape (B, 4672).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {x.size(1)}")

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout_layer(out)
        logits = self.fc2(out)
        return logits
