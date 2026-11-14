"""
Unit and integration tests for AlphaZero training pipeline (Stage 7 AlphaZero).
"""
from pathlib import Path
import numpy as np
import pytest
import torch

from models.dual_head_resnet import DualHeadResNet
from games.chess.encoder import NUM_ACTIONS, NUM_CHANNELS
from agents.stage7_alphazero import AlphaZeroAgent
from training.stage7_train_alphazero import (
    AlphaZeroTrainConfig,
    AlphaZeroReplayBuffer,
    SelfPlaySample,
    compute_alphazero_loss,
    play_self_play_game,
    save_checkpoint,
    train,
)


def test_replay_buffer_operations():
    """Verify push, capacity, and minibatch sampling of AlphaZeroReplayBuffer."""
    buffer = AlphaZeroReplayBuffer(capacity=10)
    for i in range(12):
        s = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.float32)
        pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
        pi[i] = 1.0
        buffer.push(SelfPlaySample(state=s, pi=pi, value=float(i % 2)))

    assert len(buffer) == 10  # Capacity limit adhered to

    states, pis, values = buffer.sample_batch(batch_size=4)
    assert states.shape == (4, NUM_CHANNELS, 8, 8)
    assert pis.shape == (4, NUM_ACTIONS)
    assert values.shape == (4, 1)


def test_compute_alphazero_loss():
    """Verify loss function computes valid policy cross-entropy and value MSE."""
    logits = torch.randn(4, NUM_ACTIONS, requires_grad=True)
    val_raw = torch.randn(4, 1, requires_grad=True)
    val = torch.tanh(val_raw)

    target_pi = torch.zeros(4, NUM_ACTIONS)
    target_pi[:, 0] = 0.5
    target_pi[:, 1] = 0.5

    target_z = torch.tensor([[1.0], [-1.0], [0.0], [1.0]])

    tot, pol, v_loss = compute_alphazero_loss(logits, val, target_pi, target_z)

    assert tot.item() > 0.0
    assert pol.item() > 0.0
    assert v_loss.item() >= 0.0

    tot.backward()
    assert logits.grad is not None
    assert val_raw.grad is not None


def test_play_self_play_game():
    """Verify that a self-play game produces valid trajectories."""
    model = DualHeadResNet(num_res_blocks=1, hidden_dim=32, policy_channels=4)
    model.eval()

    cfg = AlphaZeroTrainConfig(
        mcts_simulations=5,
        max_plies_per_game=6,
        temp_threshold_plies=2,
    )

    samples = play_self_play_game(model, cfg, seed=42)
    assert len(samples) > 0
    assert len(samples) <= 6

    for s in samples:
        assert s.state.shape == (NUM_CHANNELS, 8, 8)
        assert s.pi.shape == (NUM_ACTIONS,)
        assert sum(s.pi) == pytest.approx(1.0, abs=1e-5)
        assert s.value in [-1.0, 0.0, 1.0]


def test_save_and_load_checkpoint_contract(tmp_path: Path):
    """Verify save_checkpoint generates payload matching Appendix B contract."""
    model = DualHeadResNet(num_res_blocks=1, hidden_dim=32, policy_channels=4)
    cfg = AlphaZeroTrainConfig(checkpoint_dir=str(tmp_path), seed=123)

    saved_file = save_checkpoint(model, cfg, step=1, metrics={"loss": 0.5}, filename="test_az.pt")
    assert saved_file.is_file()

    payload = torch.load(saved_file, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["stage"] == 7
    assert payload["model_class"] == "DualHeadResNet"
    assert payload["num_actions"] == 4672
    assert payload["claim_draw"] is True
    assert "state_dict" in payload

    # Ensure AlphaZeroAgent can load it seamlessly
    agent = AlphaZeroAgent(checkpoint=saved_file, simulations=5)
    assert agent.network is not None


def test_train_pipeline_smoke(tmp_path: Path):
    """Smoke test running 1 iteration of the closed-loop AlphaZero training pipeline."""
    cfg = AlphaZeroTrainConfig(
        num_iterations=1,
        games_per_iteration=1,
        mcts_simulations=5,
        max_plies_per_game=4,
        batch_size=4,
        checkpoint_dir=str(tmp_path),
        seed=42,
    )

    result = train(cfg)
    assert result["status"] == "success"
    assert result["total_samples"] > 0
    assert Path(result["checkpoint_path"]).is_file()
