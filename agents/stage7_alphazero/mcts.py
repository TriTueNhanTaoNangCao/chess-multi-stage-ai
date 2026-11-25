"""
Stage 7: AlphaZero MCTS Engine (PUCT without Rollout, Dirichlet Noise, and Temperature).
Author: Phan Thanh Tung (Tung).

Architectural Reference:
- Silver et al., "A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play", Science 362, 1140-1144 (2018).
- Mark Liu, "AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games", CRC Press.

Key Principles of AlphaZero MCTS:
1. Selection: Traverse tree using PUCT score = (-Q_child) + u(s, a).
2. Pure Neural Leaf Evaluation: Leaf position is evaluated directly by DualHeadResNet
   Value Head v(s) in [-1.0, 1.0]. Zero random rollout is performed (lambda = 0).
3. Root Dirichlet Noise: During self-play, inject Dir(alpha=0.3) noise with weight
   epsilon=0.25 into the root's prior probabilities.
4. Backpropagation: Cumulative updates N += 1, W += v, Q = W / N, inverting sign v = -v
   at each step up the tree to reflect zero-sum alternating play.
5. Action Probabilities: Move selection follows pi(a|s) proportional to N(s, a)^(1/tau).
"""
from __future__ import annotations
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import chess
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder
from .node import AlphaZeroNode


class AlphaZeroMCTS:
    """
    Search engine for AlphaZero.
    Integrates DualHeadResNet inference with PUCT tree search.
    """

    def __init__(
        self,
        network: Optional[Any] = None,
        predict_fn: Optional[Callable[[ChessBoard], Tuple[Dict[chess.Move, float], float]]] = None,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.network = network
        self.predict_fn = predict_fn
        self.simulations = max(1, int(simulations))
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.device = device
        self.seed = seed

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset search engine state and RNG seeds."""
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    def _predict(self, board: ChessBoard) -> Tuple[Dict[chess.Move, float], float]:
        """
        Evaluate board state using DualHeadResNet or custom predict_fn.
        Returns:
            Tuple of (priors_dict for legal moves, scalar_value in [-1, 1]).
        """
        if self.predict_fn is not None:
            return self.predict_fn(board)

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            return {}, 0.0

        if self.network is None or torch is None:
            # Fallback uniform prior and neutral evaluation
            n = len(legal_moves)
            uniform_priors = {m: 1.0 / n for m in legal_moves}
            return uniform_priors, 0.0

        # Encode state: HWC (8, 8, 14) -> NCHW (1, 14, 8, 8)
        hwc_state = ChessEncoder.encode_board(board.board)
        chw_state = np.transpose(hwc_state, (2, 0, 1)).copy()
        tensor = torch.from_numpy(chw_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_tensor = self.network(tensor)

        logits = policy_logits[0]  # shape (4672,)
        val = float(value_tensor.item())

        # Mask and compute softmax over legal moves
        turn = board.turn
        move_logits: Dict[chess.Move, float] = {}
        for move in legal_moves:
            action_idx = ChessEncoder.encode_move(move, turn)
            move_logits[move] = float(logits[action_idx].item())

        max_l = max(move_logits.values())
        exp_vals = {m: math.exp(l - max_l) for m, l in move_logits.items()}
        sum_exp = sum(exp_vals.values())

        if sum_exp <= 0 or math.isnan(sum_exp):
            n = len(legal_moves)
            priors = {m: 1.0 / n for m in legal_moves}
        else:
            priors = {m: v / sum_exp for m, v in exp_vals.items()}

        return priors, val

    def search(
        self,
        board: ChessBoard,
        add_noise: bool = False,
        temperature: float = 1.0,
    ) -> chess.Move:
        """
        Run AlphaZero MCTS search and select action according to temperature tau.

        Args:
            board: Current board state.
            add_noise: Whether to inject Dirichlet noise at root (True during self-play).
            temperature: Softmax sampling temperature tau (tau -> 0 for greedy/competitive play).
        Returns:
            Selected legal chess.Move.
        """
        if board.is_game_over():
            raise ValueError("Cannot search in terminal position.")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            raise ValueError("No legal moves available.")
        if len(legal_moves) == 1:
            return legal_moves[0]

        root = self._build_and_run_mcts(board, add_noise=add_noise)
        return self._select_move_from_root(root, temperature=temperature)

    def get_action_probs(
        self,
        board: ChessBoard,
        add_noise: bool = False,
        temperature: float = 1.0,
    ) -> Dict[chess.Move, float]:
        """
        Run MCTS search and return action probability distribution pi(a|s) proportional
        to N(s, a)^(1/tau).
        """
        if board.is_game_over():
            raise ValueError("Cannot compute action probabilities for terminal position.")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            return {}
        if len(legal_moves) == 1:
            return {legal_moves[0]: 1.0}

        root = self._build_and_run_mcts(board, add_noise=add_noise)
        return self._compute_pi(root, temperature=temperature)

    def _build_and_run_mcts(self, board: ChessBoard, add_noise: bool) -> AlphaZeroNode:
        """Execute simulation iterations from root."""
        root = AlphaZeroNode(board=board.clone())
        priors, _ = self._predict(root.board)
        root.expand(priors)

        if add_noise and root.children:
            root.add_dirichlet_noise(
                alpha=self.dirichlet_alpha,
                epsilon=self.dirichlet_epsilon,
                rng=self.np_rng,
            )

        for _ in range(self.simulations):
            self._simulate(root)

        return root

    def _simulate(self, root: AlphaZeroNode) -> None:
        """Single MCTS simulation: Selection -> Expansion/Evaluation -> Backpropagation."""
        node = root

        # 1. Selection
        while node.is_expanded and not node.is_terminal:
            _, node = node.select_child(c_puct=self.c_puct, rng=self.rng)

        # 2. Expansion and Evaluation
        if node.is_terminal:
            v = node.terminal_value()
            if v is None:
                v = 0.0
        else:
            priors, v = self._predict(node.board)
            node.expand(priors)

        # 3. Backpropagation (Negamax inversion)
        curr: Optional[AlphaZeroNode] = node
        val = v
        while curr is not None:
            curr.update(val)
            val = -val
            curr = curr.parent

    def search_timed(self, board, limits):
        """Competitive best-so-far selection; training search remains unchanged."""
        limits.check()
        moves = board.get_legal_actions()
        if len(moves) == 1:
            limits.publish(moves[0], simulations_completed=0)
            return moves[0]
        root = AlphaZeroNode(board=board.clone())
        priors, _ = self._predict(root.board)
        limits.check()
        root.expand(priors)
        for completed in range(1, self.simulations + 1):
            limits.check()
            node = root
            while node.is_expanded and not node.is_terminal:
                limits.check()
                _, node = node.select_child(c_puct=self.c_puct, rng=self.rng)
            if node.is_terminal:
                value = node.terminal_value() or 0.0
            else:
                priors, value = self._predict(node.board)
                limits.check()
                node.expand(priors)
            limits.check()
            current = node
            while current is not None:
                current.update(value)
                value = -value
                current = current.parent
            best = max(root.children, key=lambda m: (root.children[m].visit_count, -root.children[m].mean_value))
            limits.publish(best, simulations_completed=completed, root_visits=root.visit_count)
        return limits.best_move

    def _compute_pi(self, root: AlphaZeroNode, temperature: float) -> Dict[chess.Move, float]:
        """Compute pi(a|s) based on visit counts and temperature tau."""
        if not root.children:
            return {}

        if temperature <= 1e-2:
            # Greedy: 1.0 for most visited child
            max_n = max(child.visit_count for child in root.children.values())
            best_moves = [m for m, child in root.children.items() if child.visit_count == max_n]
            chosen = self.rng.choice(best_moves)
            return {m: (1.0 if m == chosen else 0.0) for m in root.children.keys()}

        # Scaled visits: N^(1/tau)
        inv_tau = 1.0 / temperature
        scaled = {
            m: (child.visit_count ** inv_tau) if child.visit_count > 0 else 0.0
            for m, child in root.children.items()
        }
        total = sum(scaled.values())

        if total <= 0:
            n = len(root.children)
            return {m: 1.0 / n for m in root.children.keys()}

        return {m: v / total for m, v in scaled.items()}

    def _select_move_from_root(self, root: AlphaZeroNode, temperature: float) -> chess.Move:
        """Pick move according to action distribution."""
        pi = self._compute_pi(root, temperature=temperature)
        if not pi:
            raise RuntimeError("Root node has no actions.")

        moves = list(pi.keys())
        probs = list(pi.values())

        if temperature <= 1e-2:
            # Exactly argmax
            return max(moves, key=lambda m: pi[m])

        return self.rng.choices(moves, weights=probs, k=1)[0]
