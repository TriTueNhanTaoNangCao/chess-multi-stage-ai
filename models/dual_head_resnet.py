"""
Stage 7: Dual-Head ResNet Architecture for AlphaZero Chess.
Jointly predicts move policy distribution p = Pr(a|s) and position value v in [-1, 1].

Architectural reference:
- Silver et al., "A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play", Science 362, 1140-1144 (2018).
- Mark Liu, "AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games", CRC Press.
"""
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from games.chess.encoder import NUM_ACTIONS, NUM_CHANNELS


class ResidualBlock(nn.Module):
    """
    Standard AlphaZero residual block with two 3x3 convolutions, batch normalization,
    and a skip-connection.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class DualHeadResNet(nn.Module):
    """
    Dual-Head Residual Neural Network for AlphaZero Chess.

    - Input: Tensor of shape (B, 14, 8, 8) representing board state from side to move.
    - Shared Backbone: Initial 3x3 conv followed by a stack of residual blocks.
    - Policy Head: 1x1 conv + BN + ReLU + FC -> (B, 4672) raw logits over AlphaZero action space.
    - Value Head: 1x1 conv + BN + ReLU + FC(128) + ReLU + FC(1) + Tanh -> (B, 1) scalar in [-1, 1].
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 128,
        num_res_blocks: int = 4,
        policy_channels: int = 16,
        fc_val_dim: int = 128,
    ):
        super().__init__()
        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "num_actions": num_actions,
            "hidden_dim": hidden_dim,
            "num_res_blocks": num_res_blocks,
            "policy_channels": policy_channels,
            "fc_val_dim": fc_val_dim,
        }
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        self.policy_channels = policy_channels
        self.fc_val_dim = fc_val_dim

        # Initial representation convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_dim)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(hidden_dim, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * 8 * 8, num_actions)

        # Value Head
        self.value_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, fc_val_dim)
        self.value_fc2 = nn.Linear(fc_val_dim, 1)

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration for checkpoint serialization."""
        return dict(self.config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DualHeadResNet":
        """Instantiate network from configuration dictionary."""
        return cls(**config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing policy logits and position evaluation.

        Args:
            x: Float tensor of shape (B, 14, 8, 8) in NCHW format.
        Returns:
            Tuple of:
            - policy_logits: Raw unnormalized logits of shape (B, 4672).
            - value: Position evaluation of shape (B, 1) in [-1.0, 1.0].
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D tensor with shape {tuple(x.shape)}"
            )
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected input channels {self.in_channels}, got {x.size(1)}")

        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper returning normalized action probabilities and value.

        Args:
            x: Float tensor of shape (B, 14, 8, 8).
        Returns:
            Tuple of (policy_probs, value).
        """
        self.eval()
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs, value
