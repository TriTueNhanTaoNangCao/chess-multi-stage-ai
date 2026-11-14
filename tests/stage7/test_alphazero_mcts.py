"""
Unit tests for AlphaZeroMCTS search engine (Stage 7 AlphaZero).
"""
import chess
import pytest
import torch

from agents.stage7_alphazero.mcts import AlphaZeroMCTS
from games.chess.board import ChessBoard
from models.dual_head_resnet import DualHeadResNet


def test_mcts_single_legal_move():
    """Verify that a position with only one legal move returns immediately without simulation."""
    # Custom FEN where Black has only one legal king escape
    # White King on h6, pawn on g7, Black King on h8
    b = ChessBoard("7k/6P1/7K/8/8/8/8/8 b - - 0 1")
    assert len(b.get_legal_actions()) == 1

    mcts = AlphaZeroMCTS(simulations=50)
    move = mcts.search(b)
    assert move == chess.Move.from_uci("h8g8")


def test_mcts_search_runs_simulations():
    """Verify that MCTS runs exactly the configured number of simulations."""
    b = ChessBoard()
    mcts = AlphaZeroMCTS(simulations=25, seed=42)
    root = mcts._build_and_run_mcts(b, add_noise=False)

    total_visits = sum(child.visit_count for child in root.children.values())
    assert total_visits == 25
    assert root.visit_count == 25


def test_mcts_action_probs_greedy_vs_temperature():
    """Verify action distribution for greedy (tau=0.0) vs exploration (tau=1.0)."""
    b = ChessBoard()
    mcts = AlphaZeroMCTS(simulations=30, seed=123)

    # Greedy tau <= 1e-2
    pi_greedy = mcts.get_action_probs(b, temperature=0.0)
    assert sum(pi_greedy.values()) == pytest.approx(1.0)
    # Greedy should have exactly one 1.0 and rest 0.0
    non_zeros = [v for v in pi_greedy.values() if v > 0.0]
    assert len(non_zeros) == 1
    assert non_zeros[0] == pytest.approx(1.0)

    # Temperature tau = 1.0
    pi_soft = mcts.get_action_probs(b, temperature=1.0)
    assert sum(pi_soft.values()) == pytest.approx(1.0)
    # Proportional visits should distribute over visited children
    assert len([v for v in pi_soft.values() if v > 0.0]) > 1


def test_mcts_with_dual_head_resnet_integration():
    """Verify end-to-end integration between AlphaZeroMCTS and DualHeadResNet."""
    model = DualHeadResNet(num_res_blocks=1, hidden_dim=32, policy_channels=4)
    model.eval()

    mcts = AlphaZeroMCTS(network=model, simulations=15, seed=99)
    b = ChessBoard()
    move = mcts.search(b, add_noise=True, temperature=0.0)

    assert isinstance(move, chess.Move)
    assert move in b.get_legal_actions()


def test_mcts_terminal_raises_error():
    """Verify searching in a terminal position raises ValueError."""
    b = ChessBoard("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_game_over()

    mcts = AlphaZeroMCTS(simulations=10)
    with pytest.raises(ValueError, match="Cannot search in terminal position"):
        mcts.search(b)
