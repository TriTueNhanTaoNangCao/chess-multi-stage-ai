"""Tùng: (N,14,8,8) -> (N,4672) logits. Coordinate design with Thái."""
import torch.nn as nn

class CNNPolicyNetwork(nn.Module):
    def __init__(self, in_channels=14):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        raise NotImplementedError("(N,14,8,8) -> (N,4672) logits; network pending implementation")
