"""
Unit tests for DualHeadResNet architecture (Stage 7 AlphaZero).
"""
import pytest
import torch

from models.dual_head_resnet import DualHeadResNet, ResidualBlock
from games.chess.encoder import NUM_ACTIONS, NUM_CHANNELS


def test_residual_block():
    """Verify ResidualBlock maintains spatial dimensions and channel depth."""
    channels = 64
    block = ResidualBlock(channels)
    x = torch.randn(2, channels, 8, 8)
    out = block(x)
    assert out.shape == (2, channels, 8, 8)
    assert not torch.isnan(out).any()


def test_forward_pass_shapes():
    """Verify DualHeadResNet forward output shapes for batch sizes 1 and 4."""
    model = DualHeadResNet(num_res_blocks=2, hidden_dim=64, policy_channels=8)
    
    # Single sample
    x1 = torch.randn(1, NUM_CHANNELS, 8, 8)
    logits1, val1 = model(x1)
    assert logits1.shape == (1, NUM_ACTIONS)
    assert val1.shape == (1, 1)

    # Batch of 4
    x4 = torch.randn(4, NUM_CHANNELS, 8, 8)
    logits4, val4 = model(x4)
    assert logits4.shape == (4, NUM_ACTIONS)
    assert val4.shape == (4, 1)


def test_value_range_tanh():
    """Verify that value head is strictly bounded within [-1.0, 1.0]."""
    model = DualHeadResNet(num_res_blocks=1, hidden_dim=32, policy_channels=4)
    x = torch.randn(10, NUM_CHANNELS, 8, 8) * 10.0  # Large inputs to saturate
    _, val = model(x)
    assert torch.all(val >= -1.0)
    assert torch.all(val <= 1.0)


def test_gradient_flow_both_heads():
    """Verify backpropagation computes valid gradients for both policy and value heads."""
    model = DualHeadResNet(num_res_blocks=2, hidden_dim=32, policy_channels=4)
    x = torch.randn(2, NUM_CHANNELS, 8, 8)
    target_policy = torch.randint(0, NUM_ACTIONS, (2,))
    target_value = torch.tensor([[1.0], [-1.0]])

    logits, val = model(x)
    loss_p = torch.nn.functional.cross_entropy(logits, target_policy)
    loss_v = torch.nn.functional.mse_loss(val, target_value)
    total_loss = loss_p + loss_v

    total_loss.backward()

    # Check gradients in shared backbone and both heads
    assert model.conv_in.weight.grad is not None
    assert not torch.isnan(model.conv_in.weight.grad).any()
    assert model.policy_fc.weight.grad is not None
    assert model.value_fc2.weight.grad is not None


def test_config_serialization():
    """Verify get_config and from_config factory contract."""
    model = DualHeadResNet(
        in_channels=14,
        num_actions=4672,
        hidden_dim=64,
        num_res_blocks=3,
        policy_channels=8,
        fc_val_dim=64,
    )
    cfg = model.get_config()
    assert cfg["hidden_dim"] == 64
    assert cfg["num_res_blocks"] == 3
    assert cfg["policy_channels"] == 8

    reconstructed = DualHeadResNet.from_config(cfg)
    assert reconstructed.hidden_dim == 64
    assert reconstructed.num_res_blocks == 3
    assert len(reconstructed.res_blocks) == 3


def test_predict_helper():
    """Verify predict helper returns valid probabilities summing to 1.0."""
    model = DualHeadResNet(num_res_blocks=1, hidden_dim=32, policy_channels=4)
    x = torch.randn(2, NUM_CHANNELS, 8, 8)
    probs, val = model.predict(x)

    assert probs.shape == (2, NUM_ACTIONS)
    assert val.shape == (2, 1)
    sums = torch.sum(probs, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_invalid_input_shapes():
    """Verify that ValueError is raised on non-4D inputs or wrong channel count."""
    model = DualHeadResNet()
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model(torch.randn(14, 8, 8))

    with pytest.raises(ValueError, match="Expected input channels 14"):
        model(torch.randn(1, 10, 8, 8))
