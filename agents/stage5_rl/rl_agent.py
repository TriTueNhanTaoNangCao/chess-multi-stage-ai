"""Stage 5 REINFORCE policy agent for checkpoint inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess

try:
    import torch
    from training.stage5.checkpoint import load_stage5_policy_checkpoint
except ModuleNotFoundError as err:
    if err.name == "torch" or (err.name and err.name.startswith("torch.")):
        torch = None
        load_stage5_policy_checkpoint = None
    else:
        raise

from agents.base import BaseAgent
from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS


class RLPolicyAgent(BaseAgent):
    """Greedy legal-masked Policy inference from a Stage 5 checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        seed: Optional[int] = None,
        name: str = "RLPolicyAgent",
        player_id: Optional[int] = None,
    ):
        super().__init__(name=name, player_id=player_id)
        if torch is None or load_stage5_policy_checkpoint is None:
            raise RuntimeError(
                "PyTorch is required to run RLPolicyAgent but is not installed."
            )
        if not checkpoint:
            raise ValueError("RLPolicyAgent requires a valid checkpoint path.")
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device)
        self.seed = seed
        self.model, self.metadata = load_stage5_policy_checkpoint(
            self.checkpoint_path, device=self.device
        )

    def act(self, board: ChessBoard) -> chess.Move:
        if not isinstance(board, ChessBoard):
            raise TypeError("RLPolicyAgent expects a ChessBoard")
        if board.is_terminal():
            raise ValueError(f"Cannot act on terminal position: {board.fen()}")

        state = ChessEncoder.encode_board(board).transpose(2, 0, 1).copy()
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        legal_mask = ChessEncoder.get_legal_action_mask(
            board, claim_draw=board.claim_draw
        ).astype(bool, copy=False)
        if not legal_mask.any():
            raise ValueError("No legal actions available in current position")
        mask_tensor = torch.from_numpy(legal_mask).to(
            device=self.device, dtype=torch.bool
        )
        with torch.no_grad():
            logits = self.model(state_tensor)
        if logits.shape != (1, NUM_ACTIONS):
            raise RuntimeError(
                f"Policy returned shape {tuple(logits.shape)}, "
                f"expected (1, {NUM_ACTIONS})"
            )
        action = int(
            logits[0].masked_fill(~mask_tensor, -torch.inf).argmax().item()
        )
        move = ChessEncoder.decode_move(
            action, board, claim_draw=board.claim_draw
        )
        if move is None or not board.is_legal(move):
            raise RuntimeError(
                f"RLPolicyAgent produced an illegal move at action {action}"
            )
        return move

    def reset(self) -> None:
        pass
