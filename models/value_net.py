"""
Stage 5: Value Network for Chess Position Evaluation.
Architectural reference: Silver et al. (Nature 2016, 2017), Mark Liu (2025) Chapter 13 & 15.
"""
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from games.chess.encoder import NUM_CHANNELS


class ValueNetwork(nn.Module):
    """
    Evaluates board position s to output scalar value v in [-1, 1]
    representing expected game outcome z from current side-to-move's perspective.
    Input: (B, 14, 8, 8) in NCHW format.
    Output: (B, 1) scalar value in [-1.0, 1.0].
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden_dim: int = 128,
        conv2_dim: int = 64,
        fc1_dim: int = 256,
        fc2_dim: int = 64,
    ):
        super().__init__()
        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "hidden_dim": hidden_dim,
            "conv2_dim": conv2_dim,
            "fc1_dim": fc1_dim,
            "fc2_dim": fc2_dim,
        }
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.conv2_dim = conv2_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, conv2_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_dim)

        self.conv_val = nn.Conv2d(conv2_dim, 1, kernel_size=1)
        self.bn_val = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(8 * 8, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, 1)

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration for checkpoint serialization."""
        return dict(self.config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ValueNetwork":
        """Instantiate network from configuration dictionary."""
        return cls(**config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing scalar value evaluation in range [-1.0, 1.0].
        Args:
            x: Float tensor of shape (B, 14, 8, 8) in NCHW format.
        Returns:
            Position value of shape (B, 1) via tanh.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {x.size(1)}")

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        out = F.relu(self.bn_val(self.conv_val(out)))
        out = out.reshape(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        val = torch.tanh(self.fc3(out))
        return val
