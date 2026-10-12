"""Tùng: (N,14,8,8) -> (N,1) value in [-1,1], side-to-move. Coordinate design with Thái."""
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, in_channels=14):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        raise NotImplementedError("(N,14,8,8) -> (N,1) value in [-1,1], side-to-move; network pending implementation")
