"""
Stage 7: AlphaZero Chess Agent with Dual-Head ResNet and PUCT Search.
Author: Phan Thanh Tung (Tung).

Architectural Reference:
- Silver et al., "A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play", Science 362, 1140-1144 (2018).
- Mark Liu, "AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games", CRC Press.

Technical Invariants:
- Implements the unified BaseAgent contract.
- Guarantees immutability of the caller's ChessBoard instance (clones for search).
- Raises ValueError if invoked on a terminal position or when no legal moves exist.
- Returns a strictly verified legal chess.Move.
- Supports checkpoint deserialization following Appendix B contract or falls back
  safely to an initialized DualHeadResNet so CLI/GUI/Arena never crash.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import chess

try:
    import torch
    from models.dual_head_resnet import DualHeadResNet
    from models.initialization import initialize_model
except ImportError:
    torch = None
    DualHeadResNet = None

from agents.base import BaseAgent
from games.chess.board import ChessBoard
from .mcts import AlphaZeroMCTS


class AlphaZeroAgent(BaseAgent):
    """
    AlphaZero Agent for Chess.
    Combines a Dual-Head ResNet (joint policy-value representation) with PUCT tree search.
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        network: Optional[Any] = None,
        simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 0.0,
        add_noise: bool = False,
        player_id: Optional[int] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__(name="AlphaZero", player_id=player_id)
        self.seed = seed
        self.device = device
        self.simulations = max(1, int(simulations))
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.temperature = float(temperature)
        self.add_noise = bool(add_noise)

        # 1. Initialize or load DualHeadResNet model
        self.network = network
        if self.network is None and checkpoint is not None:
            self.network = self._load_checkpoint(checkpoint)

        # Fallback mode: instantiate default DualHeadResNet if PyTorch is available
        if self.network is None and torch is not None and DualHeadResNet is not None:
            self.network = initialize_model(DualHeadResNet, seed)
            self.network.to(self.device)
            self.network.eval()

        # 2. Initialize AlphaZero MCTS search engine
        self.mcts = AlphaZeroMCTS(
            network=self.network,
            simulations=self.simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            device=self.device,
            seed=self.seed,
        )

    def _load_checkpoint(self, path: Union[str, Path]) -> Optional[Any]:
        """
        Load DualHeadResNet weights from checkpoint path following Appendix B schema.
        """
        if torch is None or DualHeadResNet is None:
            return None

        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"AlphaZero checkpoint does not exist: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            model_config = payload.get("model_config", {})
            if model_config:
                model = DualHeadResNet.from_config(model_config)
            else:
                model = DualHeadResNet()
            model.load_state_dict(payload["state_dict"], strict=True)
        elif isinstance(payload, dict):
            model = DualHeadResNet()
            model.load_state_dict(payload, strict=True)
        elif isinstance(payload, DualHeadResNet):
            model = payload
        else:
            raise ValueError(f"Unrecognized checkpoint payload format in {ckpt_path}")

        model.to(self.device)
        model.eval()
        return model

    def act(self, board: ChessBoard) -> chess.Move:
        """
        Select best legal move on the given board using AlphaZero MCTS.

        Args:
            board: Current ChessBoard state.
        Returns:
            Selected legal chess.Move.
        Raises:
            ValueError: If position is terminal or has no legal moves.
        """
        if board.is_terminal() or board.is_game_over():
            raise ValueError(f"Cannot act on terminal position: {board.fen()}")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            raise ValueError("No legal moves available on board.")

        # Safety: Clone board before passing to search engine
        search_board = board.clone()
        chosen_move = self.mcts.search(
            search_board,
            add_noise=self.add_noise,
            temperature=self.temperature,
        )

        if not board.is_legal(chosen_move):
            raise RuntimeError(f"AlphaZeroAgent generated illegal move: {chosen_move}")

        return chosen_move

    def get_action_probs(
        self,
        board: ChessBoard,
        temperature: Optional[float] = None,
        add_noise: Optional[bool] = None,
    ) -> Dict[chess.Move, float]:
        """
        Compute MCTS action probability distribution pi(a|s) proportional to N(s,a)^(1/tau).
        """
        if board.is_terminal() or board.is_game_over():
            raise ValueError(f"Cannot compute action probabilities for terminal position: {board.fen()}")

        temp = self.temperature if temperature is None else float(temperature)
        noise = self.add_noise if add_noise is None else bool(add_noise)

        search_board = board.clone()
        return self.mcts.get_action_probs(search_board, add_noise=noise, temperature=temp)

    def reset(self) -> None:
        """Reset search state between games."""
        self.mcts.reset()
