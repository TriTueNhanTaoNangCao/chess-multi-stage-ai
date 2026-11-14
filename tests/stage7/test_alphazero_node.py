"""
Unit tests for AlphaZeroNode (Stage 7 AlphaZero).
"""
import math
import chess
import numpy as np
import pytest

from agents.stage7_alphazero.node import AlphaZeroNode
from games.chess.board import ChessBoard


def test_node_initialization():
    """Verify initial properties of an AlphaZeroNode."""
    b = ChessBoard()
    node = AlphaZeroNode(board=b, prior=0.75)
    assert node.parent is None
    assert node.move is None
    assert node.prior == 0.75
    assert node.visit_count == 0
    assert node.mean_value == 0.0
    assert not node.is_expanded
    assert not node.is_terminal


def test_node_expansion():
    """Verify node expansion instantiates children for all legal moves."""
    b = ChessBoard()
    node = AlphaZeroNode(board=b)
    legal_moves = b.get_legal_actions()
    assert len(legal_moves) == 20

    priors = {m: 1.0 / len(legal_moves) for m in legal_moves}
    node.expand(priors)

    assert node.is_expanded
    assert len(node.children) == 20
    for m in legal_moves:
        assert m in node.children
        child = node.children[m]
        assert child.parent is node
        assert child.move == m
        assert child.prior == pytest.approx(1.0 / 20)
        assert child.visit_count == 0


def test_puct_score_calculation():
    """
    Verify exact PUCT score calculation:
      score = (-child.mean_value) + c_puct * child.prior * sqrt(N_parent) / (1 + N_child)
    """
    b = ChessBoard()
    parent = AlphaZeroNode(board=b)
    parent.visit_count = 100

    m = chess.Move.from_uci("e2e4")
    child = AlphaZeroNode(board=b.clone(), parent=parent, move=m, prior=0.4)
    child.visit_count = 15
    child.total_value = -3.0
    child.mean_value = -3.0 / 15.0  # -0.2

    # score = -(-0.2) + 1.5 * 0.4 * sqrt(100) / (1 + 15)
    #       = +0.2 + 0.6 * 10 / 16 = 0.2 + 6.0 / 16 = 0.2 + 0.375 = 0.575
    expected_score = 0.2 + 1.5 * 0.4 * 10.0 / 16.0
    calc_score = parent.puct_score(child, c_puct=1.5)
    assert calc_score == pytest.approx(expected_score, abs=1e-6)


def test_child_selection():
    """Verify select_child picks child with highest PUCT score."""
    b = ChessBoard()
    parent = AlphaZeroNode(board=b)
    parent.visit_count = 10

    m1 = chess.Move.from_uci("e2e4")
    m2 = chess.Move.from_uci("d2d4")

    # m1 has higher exploitation (-Q) and higher prior
    c1 = AlphaZeroNode(board=b.clone(), parent=parent, move=m1, prior=0.6)
    c1.visit_count = 2
    c1.mean_value = -0.5  # -Q = +0.5

    c2 = AlphaZeroNode(board=b.clone(), parent=parent, move=m2, prior=0.2)
    c2.visit_count = 5
    c2.mean_value = +0.5  # -Q = -0.5

    parent.children = {m1: c1, m2: c2}
    parent.is_expanded = True

    chosen_move, chosen_child = parent.select_child(c_puct=1.5)
    assert chosen_move == m1
    assert chosen_child is c1


def test_dirichlet_noise_injection():
    """Verify Dirichlet noise perturbs priors while preserving valid distribution."""
    b = ChessBoard()
    node = AlphaZeroNode(board=b)
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    node.expand({moves[0]: 0.5, moves[1]: 0.5})

    orig_priors = [child.prior for child in node.children.values()]
    rng = np.random.default_rng(seed=42)
    node.add_dirichlet_noise(alpha=0.3, epsilon=0.25, rng=rng)

    new_priors = [child.prior for child in node.children.values()]
    assert new_priors != orig_priors
    assert sum(new_priors) == pytest.approx(1.0, abs=1e-5)
    for p in new_priors:
        assert p > 0.0


def test_terminal_value_checkmate():
    """Verify terminal value is -1.0 for player whose turn it is in checkmate."""
    # Fool's mate: White is checkmated, turn=White
    b = ChessBoard("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_game_over()

    node = AlphaZeroNode(board=b)
    assert node.is_terminal
    assert node.terminal_value() == -1.0
