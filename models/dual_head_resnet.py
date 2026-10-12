"""Tùng: (N,14,8,8) -> policy logits (N,4672), value (N,1). Coordinate design with Thái."""
import torch.nn as nn

class DualHeadResNet(nn.Module):
    def __init__(self, in_channels=14):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        raise NotImplementedError("(N,14,8,8) -> policy logits (N,4672), value (N,1); network pending implementation")
