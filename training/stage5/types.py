"""Stage 5 data contracts and training configuration."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
from games.chess.board import ChessBoard
from models.cnn_policy import CNNPolicyNetwork
from models.value_net import ValueNetwork


@dataclass
class GameStart:
    """One reconstructed start position and the learner color for that game."""

    board: ChessBoard
    learner_color: bool
    source_type: str
    source_id: str

@dataclass
class TrajectoryStep:
    """One learner decision, stored on CPU without an autograd graph."""

    state: np.ndarray
    action: int
    legal_mask: np.ndarray
    current_player: int
    ply_index: int
    target: Optional[float] = None

@dataclass
class GameTrajectory:
    """Learner-only decisions and the final status of one collected game."""

    steps: List[TrajectoryStep]
    learner_color: int
    truncated: bool
    termination_reason: str
    winner_white: Optional[int]
    plies_played: int

@dataclass
class ValueBatch:
    """One CPU value sample per usable terminal game."""

    states: torch.Tensor
    targets: torch.Tensor
    games_used: int
    games_skipped: int
    games_truncated: int

@dataclass
class Stage5TrainConfig:
    """Configuration for the in-memory Stage 5 training loop."""

    stage4_checkpoint: Optional[str | Path] = None
    output_dir: str | Path = "checkpoints/stage5"
    resume_checkpoint: Optional[str | Path] = None
    device: str = "cpu"
    seed: int = 42
    iterations: int = 1
    games_per_iteration: int = 2
    max_plies: int = 300
    chunk_size: Optional[int] = None
    policy_lr: float = 1e-4
    value_lr: float = 1e-3
    baseline: str = "value"
    value_mode: str = "joint"
    value_checkpoint: Optional[str | Path] = None
    opponent_mode: str = "fixed"
    opponent_snapshot_interval: int = 1
    opponent_pool_size: int = 4
    start_mode: str = "standard"
    mixed_start_manifest: Optional[str | Path] = None
    policy_init_checkpoint: Optional[str | Path] = None

    def validate(self) -> None:
        if self.policy_init_checkpoint is not None and self.resume_checkpoint is not None:
            raise ValueError(
                "policy_init_checkpoint cannot be used with resume_checkpoint"
            )
        if self.policy_init_checkpoint is not None and not Path(
            self.policy_init_checkpoint
        ).is_file():
            raise FileNotFoundError(
                f"Policy init checkpoint not found: {self.policy_init_checkpoint}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(self.iterations, int) or self.iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if (
            not isinstance(self.games_per_iteration, int)
            or self.games_per_iteration <= 0
        ):
            raise ValueError("games_per_iteration must be a positive integer")
        if not isinstance(self.max_plies, int) or self.max_plies < 0:
            raise ValueError("max_plies must be a non-negative integer")
        if self.chunk_size is not None and (
            not isinstance(self.chunk_size, int) or self.chunk_size <= 0
        ):
            raise ValueError("chunk_size must be positive when provided")
        if not np.isfinite(self.policy_lr) or self.policy_lr <= 0:
            raise ValueError("policy_lr must be finite and positive")
        if not np.isfinite(self.value_lr) or self.value_lr <= 0:
            raise ValueError("value_lr must be finite and positive")
        if self.baseline not in {"zero", "value"}:
            raise ValueError("baseline must be 'zero' or 'value'")
        if self.value_mode not in {"joint", "disabled", "frozen"}:
            raise ValueError("value_mode must be 'joint', 'disabled', or 'frozen'")
        if self.value_mode == "disabled" and self.baseline != "zero":
            raise ValueError("value_mode='disabled' requires baseline='zero'")
        if self.value_mode == "frozen":
            if self.baseline != "value":
                raise ValueError("value_mode='frozen' requires baseline='value'")
            if self.resume_checkpoint is None:
                if self.value_checkpoint is None:
                    raise ValueError(
                        "Fresh value_mode='frozen' requires value_checkpoint"
                    )
                if not Path(self.value_checkpoint).is_file():
                    raise FileNotFoundError(
                        f"Frozen Value checkpoint not found: {self.value_checkpoint}"
                    )
            elif self.value_checkpoint is not None:
                raise ValueError(
                    "Frozen resume must use the embedded Value and reject "
                    "value_checkpoint"
                )
        elif self.value_checkpoint is not None:
            raise ValueError(
                "value_checkpoint is only valid with value_mode='frozen'"
            )
        if self.opponent_mode not in {"fixed", "pool"}:
            raise ValueError("opponent_mode must be 'fixed' or 'pool'")
        if (
            not isinstance(self.opponent_snapshot_interval, int)
            or self.opponent_snapshot_interval <= 0
        ):
            raise ValueError("opponent_snapshot_interval must be positive")
        if not isinstance(self.opponent_pool_size, int) or self.opponent_pool_size < 1:
            raise ValueError("opponent_pool_size must be at least 1")
        if self.start_mode not in {"standard", "mixed"}:
            raise ValueError("start_mode must be 'standard' or 'mixed'")
        if self.start_mode == "standard":
            if self.mixed_start_manifest is not None:
                raise ValueError(
                    "mixed_start_manifest requires start_mode='mixed'"
                )
        else:
            if self.games_per_iteration != 32:
                raise ValueError(
                    "mixed start_mode requires games_per_iteration=32"
                )
            if self.mixed_start_manifest is None:
                raise ValueError(
                    "mixed start_mode requires mixed_start_manifest"
                )
            if not Path(self.mixed_start_manifest).is_file():
                raise FileNotFoundError(
                    f"Mixed-start manifest not found: {self.mixed_start_manifest}"
                )
        if self.resume_checkpoint is None and self.stage4_checkpoint is None:
            raise ValueError("stage4_checkpoint is required for a fresh run")
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Invalid device: {self.device!r}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")

@dataclass
class OpponentSnapshot:
    """Immutable inference snapshot stored in chronological pool order."""

    opponent_id: str
    iteration: int
    model: CNNPolicyNetwork

@dataclass
class Stage5TrainResult:
    """Models and per-iteration metrics; no checkpoint is written yet."""

    policy_model: nn.Module
    value_model: nn.Module
    opponent_model: nn.Module
    history: List[Dict[str, object]]
    opponent_pool: List[OpponentSnapshot]

@dataclass
class Stage5CheckpointState:
    """Strictly reconstructed Stage 5 models, optimizers, and raw metadata."""

    policy_model: CNNPolicyNetwork
    value_model: ValueNetwork
    opponent_model: CNNPolicyNetwork
    policy_optimizer: torch.optim.Optimizer
    value_optimizer: torch.optim.Optimizer
    opponent_mode: str
    opponent_pool: List[OpponentSnapshot]
    opponent_rng_state: Optional[Dict[str, object]]
    start_source: Dict[str, object]
    start_rng_state: Optional[Dict[str, object]]
    payload: Dict[str, object]
