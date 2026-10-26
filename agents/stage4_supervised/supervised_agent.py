"""
Stage 4: Supervised Learning Agent using CNNPolicyNetwork.
Selects action with maximum logit among legal moves, evaluated without gradient.
Strictly adheres to docs/HANDOFF.md schema.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    import torch
    from models.cnn_policy import CNNPolicyNetwork
except ModuleNotFoundError as err:
    if err.name == "torch" or (err.name and err.name.startswith("torch.")):
        torch = None
        CNNPolicyNetwork = None
    else:
        raise

import chess
from agents.base import BaseAgent
from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS


class SupervisedAgent(BaseAgent):
    """
    Supervised Learning Chess Agent based on CNNPolicyNetwork.
    Evaluates current board with encoder, masks illegal moves, and selects argmax action.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: str = "cpu",
        seed: Optional[int] = None,
        name: str = "SupervisedAgent",
        player_id: Optional[int] = None,
    ):
        super().__init__(name=name, player_id=player_id)
        if torch is None or CNNPolicyNetwork is None:
            raise RuntimeError("PyTorch is required to run SupervisedAgent but is not installed.")

        if not checkpoint:
            raise ValueError("SupervisedAgent requires a valid checkpoint path.")

        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device)
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.model, self.metadata = self._load_and_validate_checkpoint(self.checkpoint_path, self.device)

    @staticmethod
    def _load_and_validate_checkpoint(
        path: Path,
        device: torch.device,
    ) -> tuple[CNNPolicyNetwork, Dict[str, Any]]:
        """Validate checkpoint against docs/HANDOFF.md schema and return model in eval mode."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:
            raise ValueError(f"Failed to load checkpoint file {path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid checkpoint format in {path}: expected dictionary payload")

        # Validate HANDOFF.md metadata contract
        if payload.get("format_version") != 1:
            raise ValueError(f"Incompatible format_version: {payload.get('format_version')} (expected 1)")
        if payload.get("stage") != 4:
            raise ValueError(f"Incompatible stage: {payload.get('stage')} (expected 4)")
        if payload.get("model_class") != "CNNPolicyNetwork":
            raise ValueError(f"Incompatible model_class: {payload.get('model_class')} (expected CNNPolicyNetwork)")
        if payload.get("encoder_version") != "hwc14-white-v1":
            raise ValueError(f"Incompatible encoder_version: {payload.get('encoder_version')} (expected hwc14-white-v1)")
        if payload.get("action_version") != "az73-rankflip-v1":
            raise ValueError(f"Incompatible action_version: {payload.get('action_version')} (expected az73-rankflip-v1)")
        if payload.get("num_actions") != NUM_ACTIONS:
            raise ValueError(f"Incompatible num_actions: {payload.get('num_actions')} (expected {NUM_ACTIONS})")
        if "state_dict" not in payload:
            raise ValueError(f"Missing state_dict in checkpoint {path}")

        model_config = payload.get("model_config", {})
        try:
            model = CNNPolicyNetwork.from_config(model_config)
            model.load_state_dict(payload["state_dict"], strict=True)
        except Exception as exc:
            raise ValueError(f"Failed to reconstruct CNNPolicyNetwork from checkpoint: {exc}") from exc

        model.to(device)
        model.eval()
        return model, payload

    def act(self, board: ChessBoard) -> chess.Move:
        """
        Choose best legal move using policy network.
        Raises ValueError if board is terminal or has no legal actions.
        Does not mutate the caller's board or move history.
        """
        if board.is_terminal():
            raise ValueError(f"Cannot act on terminal position: {board.fen()}")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            raise ValueError("No legal actions available in current position")

        # 1. Encode board state using current encoder: HWC (8, 8, 14) float32
        state_hwc = ChessEncoder.encode_board(board.board)

        # 2. Transpose to CHW (14, 8, 8) and add batch dim -> NCHW (1, 14, 8, 8)
        state_chw = np.transpose(state_hwc, (2, 0, 1))
        state_tensor = torch.from_numpy(state_chw).unsqueeze(0).to(self.device)

        # 3. Model forward pass in eval mode without gradients
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0)  # (4672,)

        # 4. Construct legal action mask
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool, device=self.device)
        legal_indices = list(ChessEncoder.legal_action_indices(board.board))
        if not legal_indices:
            raise ValueError("No legal action indices extracted from board")

        for idx in legal_indices:
            legal_mask[idx] = True

        # 5. Mask illegal logits with large negative value before argmax
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        best_action_idx = masked_logits.argmax().item()

        # 6. Decode into chess.Move
        move = ChessEncoder.decode_move(best_action_idx, board.board)
        if move is None or not board.is_legal(move):
            raise RuntimeError(f"SupervisedAgent produced invalid or illegal move: {move} (action_idx={best_action_idx})")

        return move

    def reset(self) -> None:
        """Reset internal agent state between games."""
        pass
