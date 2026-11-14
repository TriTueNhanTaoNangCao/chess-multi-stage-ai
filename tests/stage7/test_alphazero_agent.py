"""
Unit tests for AlphaZeroAgent (Stage 7 AlphaZero).
"""
from pathlib import Path
import chess
import pytest
import torch

from agents.stage7_alphazero import AlphaZeroAgent
from games.chess.board import ChessBoard
from models.dual_head_resnet import DualHeadResNet


def test_agent_initialization_fallback():
    """Verify agent initializes with fallback DualHeadResNet when no checkpoint is given."""
    agent = AlphaZeroAgent(simulations=20, seed=42)
    assert agent.name == "AlphaZero"
    assert agent.simulations == 20
    assert agent.c_puct == 1.5
    assert isinstance(agent.network, DualHeadResNet)


def test_agent_act_returns_legal_move():
    """Verify act() returns a strictly legal move on starting board."""
    agent = AlphaZeroAgent(simulations=10, seed=123)
    b = ChessBoard()
    move = agent.act(b)

    assert isinstance(move, chess.Move)
    assert move in b.get_legal_actions()


def test_agent_preserves_board_immutability():
    """Verify act() does not mutate caller's board or move stack."""
    agent = AlphaZeroAgent(simulations=10, seed=42)
    b = ChessBoard()
    orig_fen = b.fen()
    orig_move_count = len(b.board.move_stack)

    _ = agent.act(b)

    assert b.fen() == orig_fen
    assert len(b.board.move_stack) == orig_move_count


def test_agent_act_terminal_raises_error():
    """Verify act() raises ValueError on a terminal position."""
    b = ChessBoard("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_game_over()

    agent = AlphaZeroAgent(simulations=5)
    with pytest.raises(ValueError, match="Cannot act on terminal position"):
        agent.act(b)


def test_agent_get_action_probs():
    """Verify get_action_probs returns valid probability distribution."""
    agent = AlphaZeroAgent(simulations=20, seed=777)
    b = ChessBoard()
    probs = agent.get_action_probs(b, temperature=1.0)

    assert len(probs) == 20
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    for p in probs.values():
        assert p >= 0.0


def test_agent_checkpoint_loading(tmp_path: Path):
    """Verify agent can load a checkpoint complying with Appendix B contract."""
    model = DualHeadResNet(num_res_blocks=1, hidden_dim=32, policy_channels=4)
    ckpt_path = tmp_path / "mock_alphazero.pt"

    payload = {
        "format_version": 1,
        "stage": 7,
        "model_class": "DualHeadResNet",
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, ckpt_path)

    loaded_agent = AlphaZeroAgent(checkpoint=ckpt_path, simulations=10)
    assert loaded_agent.network is not None
    assert isinstance(loaded_agent.network, DualHeadResNet)
    assert loaded_agent.network.num_res_blocks == 1
    assert loaded_agent.network.hidden_dim == 32

    # Verify it can act properly
    b = ChessBoard()
    move = loaded_agent.act(b)
    assert move in b.get_legal_actions()


def test_agent_reset():
    """Verify reset() can be called cleanly between games."""
    agent = AlphaZeroAgent(simulations=10, seed=42)
    b = ChessBoard()
    _ = agent.act(b)
    agent.reset()
    # Ensure next act works seamlessly after reset
    move2 = agent.act(b)
    assert move2 in b.get_legal_actions()
