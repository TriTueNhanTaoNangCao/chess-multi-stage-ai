"""
Integration tests for AlphaZero in Arena and CLI (Stage 7 AlphaZero).
"""
import chess
import pytest

from agents import AGENT_REGISTRY, CLASSIC_AGENTS, build_agent
from agents.stage7_alphazero import AlphaZeroAgent
from arena.evaluator import play_game
from games.chess.board import ChessBoard


def test_alphazero_registration():
    """Verify alphazero is properly registered in AGENT_REGISTRY and CLASSIC_AGENTS."""
    assert "alphazero" in AGENT_REGISTRY
    assert "alphazero" in CLASSIC_AGENTS
    assert AGENT_REGISTRY["alphazero"] is AlphaZeroAgent

    agent = build_agent("alphazero", depth=2, seed=42)
    assert isinstance(agent, AlphaZeroAgent)
    # depth=2 maps to max(10, 2 * 15) = 30 simulations
    assert agent.simulations == 30


def test_alphazero_arena_two_game_match():
    """Verify AlphaZero plays a balanced two-game match (as White and Black) in Arena."""
    az_agent = build_agent("alphazero", depth=1, seed=10)
    rand_agent = build_agent("random", seed=20)

    # Game 1: AlphaZero is White
    result_w = play_game(az_agent, rand_agent, max_plies=20)
    assert len(result_w.moves) > 0
    assert result_w.white == "AlphaZero"
    assert result_w.black == "Random"
    assert result_w.result in ["1-0", "0-1", "1/2-1/2"]

    # Game 2: AlphaZero is Black (color balance)
    az_agent.reset()
    rand_agent.reset()
    result_b = play_game(rand_agent, az_agent, max_plies=20)
    assert len(result_b.moves) > 0
    assert result_b.white == "Random"
    assert result_b.black == "AlphaZero"
    assert result_b.result in ["1-0", "0-1", "1/2-1/2"]


def test_alphazero_tactical_positions():
    """Verify AlphaZero handles tactical states: pawn promotion and check evasion."""
    agent = build_agent("alphazero", depth=1, seed=42)

    # 1. Pawn promotion: White pawn on c7 about to promote
    b_promo = ChessBoard("k7/2P5/K7/8/8/8/8/8 w - - 0 1")
    move_promo = agent.act(b_promo)
    assert move_promo.promotion is not None  # Must choose promotion

    # 2. Check evasion: White King under check from Black Rook
    b_check = ChessBoard("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    assert b_check.is_check()
    move_evade = agent.act(b_check)
    # Check that king escaped
    b_clone = b_check.clone()
    b_clone.push(move_evade)
    assert not b_clone.is_check()
