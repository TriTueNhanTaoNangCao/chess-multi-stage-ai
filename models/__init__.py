"""Stage 4-5 Neural Network architectures for Chess."""
from .cnn_policy import CNNPolicyNetwork
from .value_net import ValueNetwork

__all__ = ["CNNPolicyNetwork", "ValueNetwork"]
