"""
Stage 7: AlphaZero Node Architecture with PUCT and Dirichlet Noise.
Author: Phan Thanh Tung (Tung).

Architectural Reference:
- Silver et al., "A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play", Science 362, 1140-1144 (2018).
- Mark Liu, "AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games", CRC Press.

Technical Invariants:
- Value Q(s) is stored from the perspective of the player to move at *this* node.
- In zero-sum alternating play, selection score from parent's perspective is:
    score(child) = (-Q_child) + u(s, a)
  where u(s, a) = c_puct * P(s, a) * sqrt(max(1, N_parent)) / (1 + N_child).
- Dirichlet noise is injected into the root's prior probabilities during self-play:
    P'(s_root, a) = (1 - epsilon) * P(s_root, a) + epsilon * eta_a
  where eta ~ Dir(alpha) with alpha = 0.3 for chess, epsilon = 0.25.
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple
import chess
import numpy as np

from games.chess.board import ChessBoard


class AlphaZeroNode:
    """
    MCTS node for AlphaZero Chess.
    Tracks visit count N, total value W, mean action-value Q,
    and prior probability P predicted by DualHeadResNet's policy head.
    """

    def __init__(
        self,
        board: ChessBoard,
        parent: Optional[AlphaZeroNode] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ):
        self.board: ChessBoard = board
        self.parent: Optional[AlphaZeroNode] = parent
        self.move: Optional[chess.Move] = move
        self.prior: float = float(prior)

        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.mean_value: float = 0.0

        self.children: Dict[chess.Move, AlphaZeroNode] = {}
        self.is_expanded: bool = False

    @property
    def is_terminal(self) -> bool:
        """Return True if the node's position is game over."""
        return self.board.is_game_over()

    def terminal_value(self) -> Optional[float]:
        """
        Return the terminal outcome from the perspective of the player to move at this node:
        - -1.0 if current player is checkmated (loss)
        - 0.0 if draw (stalemate, repetition, 50-move rule, insufficient material)
        - +1.0 if opposing player was checkmated (should not happen at start of turn, but handled)
        - None if position is not terminal
        """
        if not self.is_terminal:
            return None
        res = self.board.outcome()
        if res is None or res.winner is None:
            return 0.0
        # If winner is the player whose turn it is, +1.0; if opponent won, -1.0
        current_turn = self.board.turn
        return 1.0 if res.winner == current_turn else -1.0

    def puct_score(self, child: AlphaZeroNode, c_puct: float = 1.5) -> float:
        """
        Compute the AlphaZero PUCT selection score for a child node.

        From parent's perspective:
          score = (-child.mean_value) + u(parent, child)
          u = c_puct * child.prior * sqrt(max(1, parent.visit_count)) / (1 + child.visit_count)
        """
        n_parent = max(1, self.visit_count)
        n_child = child.visit_count
        u = c_puct * child.prior * (math.sqrt(n_parent) / (1.0 + n_child))
        # -child.mean_value because zero-sum alternating game
        exploitation = -child.mean_value
        return exploitation + u

    def select_child(
        self,
        c_puct: float = 1.5,
        rng: Optional[random.Random] = None,
    ) -> Tuple[chess.Move, AlphaZeroNode]:
        """
        Select child with highest PUCT score.
        Ties are broken using random choice from provided or local RNG.
        """
        if not self.children:
            raise ValueError("Cannot select child from a node with no children.")

        best_score = -float("inf")
        best_candidates: List[Tuple[chess.Move, AlphaZeroNode]] = []

        for move, child in self.children.items():
            score = self.puct_score(child, c_puct=c_puct)
            if score > best_score + 1e-9:
                best_score = score
                best_candidates = [(move, child)]
            elif abs(score - best_score) <= 1e-9:
                best_candidates.append((move, child))

        if len(best_candidates) == 1:
            return best_candidates[0]
        chooser = rng.choice if rng is not None else random.choice
        return chooser(best_candidates)

    def expand(self, action_priors: Dict[chess.Move, float]) -> None:
        """
        Expand the node by instantiating children for all legal moves
        with their corresponding prior probabilities.
        """
        if self.is_expanded:
            return

        for move, prior in action_priors.items():
            child_board = self.board.clone()
            child_board.push(move)
            self.children[move] = AlphaZeroNode(
                board=child_board,
                parent=self,
                move=move,
                prior=prior,
            )
        self.is_expanded = True

    def add_dirichlet_noise(
        self,
        alpha: float = 0.3,
        epsilon: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Inject Dirichlet noise into children priors at root node to ensure exploration:
            P'(s_root, a) = (1 - epsilon) * P(s_root, a) + epsilon * eta_a
        where eta ~ Dir(alpha).
        """
        if not self.children:
            return

        num_moves = len(self.children)
        gen = rng if rng is not None else np.random.default_rng()
        noise = gen.dirichlet([alpha] * num_moves)

        for (move, child), eta in zip(self.children.items(), noise):
            child.prior = float((1.0 - epsilon) * child.prior + epsilon * eta)

    def update(self, value: float) -> None:
        """
        Update visit count and value accumulator with new evaluation.
        """
        self.visit_count += 1
        self.total_value += float(value)
        self.mean_value = self.total_value / self.visit_count
