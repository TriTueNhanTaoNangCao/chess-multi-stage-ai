"""Backward-compatible facade for the public Stage 5 REINFORCE APIs.

Only names listed in ``__all__`` are supported public exports.  Private imports
below remain implementation compatibility for the existing Stage 5 test suite.
"""

from training.stage5.types import (
    GameStart,
    GameTrajectory,
    OpponentSnapshot,
    Stage5CheckpointState,
    Stage5TrainConfig,
    Stage5TrainResult,
    TrajectoryStep,
    ValueBatch,
)
from training.stage5.selfplay import (
    MIXED_START_MIX,
    _collect_trajectory_batch_from_starts_with_opponent_pool,
    _collect_trajectory_batch_from_starts_with_rng,
    _collect_trajectory,
    _collect_trajectory_batch_with_opponent_pool,
    _collect_trajectory_batch_with_rng,
    _sample_legal_action,
    _snapshot_policy,
    _validate_learner_color,
    build_mixed_start_batch,
    collect_trajectory,
    collect_trajectory_batch,
    load_mixed_start_manifest,
)
from training.stage5.updates import (
    build_value_batch,
    reinforce_policy_update,
    update_value_network,
)
from training.stage5.checkpoint import (
    _move_optimizer_state,
    _serialized_training_config,
    _validate_resume_config,
    load_stage5_checkpoint,
    load_stage5_policy_checkpoint,
    load_value_only_checkpoint,
    save_stage5_checkpoint,
)
from training.stage5.trainer import train

__all__ = [
    "GameStart",
    "GameTrajectory",
    "OpponentSnapshot",
    "Stage5CheckpointState",
    "Stage5TrainConfig",
    "Stage5TrainResult",
    "TrajectoryStep",
    "ValueBatch",
    "build_value_batch",
    "build_mixed_start_batch",
    "collect_trajectory",
    "collect_trajectory_batch",
    "load_stage5_checkpoint",
    "load_stage5_policy_checkpoint",
    "load_value_only_checkpoint",
    "load_mixed_start_manifest",
    "reinforce_policy_update",
    "save_stage5_checkpoint",
    "train",
    "update_value_network",
]
