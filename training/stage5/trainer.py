"""Stage 5 training-loop orchestration."""
from __future__ import annotations
import copy
import random
from pathlib import Path
from typing import Dict, List
import chess
import numpy as np
import torch
from models.value_net import ValueNetwork
from training.stage4_train_sl import load_checkpoint
from training.stage5.checkpoint import (
    _validate_resume_config,
    load_stage5_checkpoint,
    load_stage5_policy_checkpoint,
    load_value_only_checkpoint,
    save_stage5_checkpoint,
)
from training.stage5.selfplay import (
    _collect_trajectory_batch_from_starts_with_opponent_pool,
    _collect_trajectory_batch_from_starts_with_rng,
    _collect_trajectory_batch_with_opponent_pool,
    _collect_trajectory_batch_with_rng,
    _snapshot_policy,
    build_mixed_start_batch,
    load_mixed_start_manifest,
)
from training.stage5.types import OpponentSnapshot, Stage5TrainConfig, Stage5TrainResult
from training.stage5.updates import build_value_batch, reinforce_policy_update, update_value_network

def train(config: Stage5TrainConfig) -> Stage5TrainResult:
    """Run or resume Stage 5; ``iterations`` is the total target count."""
    if not isinstance(config, Stage5TrainConfig):
        raise TypeError("config must be a Stage5TrainConfig")
    config.validate()
    device = torch.device(config.device)
    last_checkpoint = Path(config.output_dir) / "last.pt"

    if config.resume_checkpoint is None:
        if last_checkpoint.exists():
            raise FileExistsError(
                f"Stage 5 checkpoint already exists; pass resume_checkpoint: "
                f"{last_checkpoint}"
            )
        if config.policy_init_checkpoint is None:
            policy_model, policy_metadata = load_checkpoint(
                config.stage4_checkpoint, device=device
            )
            opponent_model, opponent_metadata = load_checkpoint(
                config.stage4_checkpoint, device=device
            )
            policy_init_source = None
        else:
            policy_model, policy_init_source = load_stage5_policy_checkpoint(
                config.policy_init_checkpoint, device=device
            )
            policy_metadata = policy_init_source
            opponent_model, opponent_metadata = load_checkpoint(
                config.stage4_checkpoint, device=device
            )
            if policy_model.get_config() != opponent_model.get_config():
                raise ValueError(
                    "Policy init and Stage 4 opponent model_config must match"
                )
        if policy_model is opponent_model:
            raise RuntimeError("Learner and opponent must be independent model instances")
        policy_model.requires_grad_(True)
        opponent_model.requires_grad_(False).eval()
        claim_draw = policy_metadata.get("claim_draw", True)
        if not isinstance(claim_draw, bool):
            raise ValueError("Policy init claim_draw metadata must be boolean")
        if opponent_metadata.get("claim_draw") != claim_draw:
            raise ValueError(
                "Policy init and Stage 4 opponent claim_draw metadata must match"
            )

        if config.value_mode == "frozen":
            value_model, value_source = load_value_only_checkpoint(
                config.value_checkpoint,
                device=device,
                claim_draw=claim_draw,
            )
        else:
            # Deterministic initialization without advancing the global CPU RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.seed)
                value_model = ValueNetwork().to(device)
            value_source = None
        policy_optimizer = torch.optim.Adam(
            policy_model.parameters(), lr=config.policy_lr
        )
        value_optimizer = torch.optim.Adam(
            value_model.parameters(), lr=config.value_lr
        )
        move_rng = torch.Generator(device="cpu")
        move_rng.manual_seed(config.seed)
        value_rng = np.random.default_rng(config.seed)
        opponent_rng = np.random.default_rng(config.seed)
        if config.start_mode == "mixed":
            mixed_start_source, start_source = load_mixed_start_manifest(
                config.mixed_start_manifest,
                claim_draw=claim_draw,
            )
            start_rng = np.random.default_rng(config.seed)
        else:
            mixed_start_source = None
            start_source = {"mode": "standard"}
            start_rng = None
        opponent_pool = [OpponentSnapshot("stage4", 0, opponent_model)]
        history: List[Dict[str, object]] = []
        completed_iteration = 0
        stage4_source: Dict[str, object] = {
            "path": str(config.stage4_checkpoint),
            "format_version": opponent_metadata.get("format_version"),
            "stage": opponent_metadata.get("stage"),
            "model_class": opponent_metadata.get("model_class"),
            "epoch": opponent_metadata.get("epoch"),
            "step": opponent_metadata.get("step"),
            "seed": opponent_metadata.get("seed"),
        }
        source_checkpoint = (
            stage4_source
            if policy_init_source is None
            else {
                "kind": "fresh_stage5_policy_init",
                "policy_init": copy.deepcopy(policy_init_source),
                "stage4_opponent": stage4_source,
            }
        )
    else:
        resumed = load_stage5_checkpoint(config.resume_checkpoint, device=device)
        payload = resumed.payload
        completed_iteration = int(payload["completed_iteration"])
        _validate_resume_config(
            config, payload["training_config"], completed_iteration
        )
        policy_model = resumed.policy_model
        value_model = resumed.value_model
        opponent_model = resumed.opponent_model
        policy_optimizer = resumed.policy_optimizer
        value_optimizer = resumed.value_optimizer
        opponent_pool = resumed.opponent_pool
        claim_draw = bool(payload["claim_draw"])
        history = copy.deepcopy(payload["history"])
        source_checkpoint = copy.deepcopy(payload["source_checkpoint"])
        value_source = copy.deepcopy(payload.get("value_source"))

        if config.start_mode == "mixed":
            mixed_start_source, current_start_source = load_mixed_start_manifest(
                config.mixed_start_manifest,
                claim_draw=claim_draw,
            )
            if current_start_source != resumed.start_source:
                raise ValueError(
                    "Resume mixed-start manifest or mix does not match checkpoint"
                )
            start_source = current_start_source
            start_rng = np.random.default_rng()
            start_rng.bit_generator.state = copy.deepcopy(
                resumed.start_rng_state
            )
        else:
            mixed_start_source = None
            start_source = {"mode": "standard"}
            start_rng = None

        move_rng = torch.Generator(device="cpu")
        move_rng.set_state(payload["collector_rng_state"].cpu())
        value_rng = np.random.default_rng()
        value_rng.bit_generator.state = copy.deepcopy(payload["value_rng_state"])
        opponent_rng = np.random.default_rng(config.seed)
        if resumed.opponent_rng_state is not None:
            opponent_rng.bit_generator.state = copy.deepcopy(
                resumed.opponent_rng_state
            )
        try:
            random.setstate(payload["python_rng_state"])
            np.random.set_state(payload["numpy_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"].cpu())
            cuda_states = payload["cuda_rng_state_all"]
            if cuda_states is not None and torch.cuda.is_available():
                if len(cuda_states) != torch.cuda.device_count():
                    raise ValueError(
                        "CUDA RNG device count differs from the Stage 5 checkpoint"
                    )
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Failed to restore Stage 5 RNG state: {exc}") from exc

    for iteration_index in range(completed_iteration, config.iterations):
        first_color = (
            chess.WHITE
            if (iteration_index * config.games_per_iteration) % 2 == 0
            else chess.BLACK
        )
        starts = None
        if config.start_mode == "mixed":
            starts = build_mixed_start_batch(
                mixed_start_source,
                rng=start_rng,
                claim_draw=claim_draw,
            )

        if config.start_mode == "mixed" and config.opponent_mode == "fixed":
            trajectories = _collect_trajectory_batch_from_starts_with_rng(
                policy_model,
                opponent_model,
                starts,
                device=device,
                rng=move_rng,
                max_plies=config.max_plies,
                claim_draw=claim_draw,
            )
            opponent_games = {"stage4": len(trajectories)}
        elif config.start_mode == "mixed":
            trajectories, opponent_games = (
                _collect_trajectory_batch_from_starts_with_opponent_pool(
                    policy_model,
                    opponent_pool,
                    starts,
                    device=device,
                    move_rng=move_rng,
                    opponent_rng=opponent_rng,
                    max_plies=config.max_plies,
                    claim_draw=claim_draw,
                )
            )
        elif config.opponent_mode == "fixed":
            trajectories = _collect_trajectory_batch_with_rng(
                policy_model,
                opponent_model,
                config.games_per_iteration,
                device=device,
                rng=move_rng,
                first_learner_color=first_color,
                max_plies=config.max_plies,
                claim_draw=claim_draw,
                initial_board=None,
            )
            opponent_games = {"stage4": len(trajectories)}
        else:
            trajectories, opponent_games = (
                _collect_trajectory_batch_with_opponent_pool(
                    policy_model,
                    opponent_pool,
                    config.games_per_iteration,
                    device=device,
                    move_rng=move_rng,
                    opponent_rng=opponent_rng,
                    first_learner_color=first_color,
                    max_plies=config.max_plies,
                    claim_draw=claim_draw,
                    initial_board=None,
                )
            )

        policy_metrics = reinforce_policy_update(
            policy_model,
            policy_optimizer,
            trajectories,
            device=device,
            chunk_size=config.chunk_size,
            value_model=value_model if config.baseline == "value" else None,
        )
        if config.value_mode == "joint":
            value_batch = build_value_batch(trajectories, rng=value_rng)
            value_metrics = update_value_network(
                value_model,
                value_optimizer,
                value_batch,
                device=device,
            )
        else:
            # Schema-compatible dormant payload: this Value is not trained.
            value_metrics = {"value_loss": None, "samples_used": 0}

        wins = draws = losses = truncated = 0
        for trajectory in trajectories:
            if trajectory.truncated:
                truncated += 1
            elif trajectory.winner_white == 0:
                draws += 1
            elif trajectory.winner_white == trajectory.learner_color:
                wins += 1
            else:
                losses += 1
        if (
            config.opponent_mode == "pool"
            and (iteration_index + 1) % config.opponent_snapshot_interval == 0
        ):
            opponent_pool.append(
                _snapshot_policy(
                    policy_model,
                    opponent_id=f"learner_iter_{iteration_index + 1:06d}",
                    iteration=iteration_index + 1,
                    device=device,
                )
            )
            while len(opponent_pool) > config.opponent_pool_size:
                opponent_pool.pop(1)

        history_record = {
                "iteration": iteration_index + 1,
                "games": len(trajectories),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "truncated": truncated,
                "decisions": policy_metrics["decisions"],
                "policy_loss": policy_metrics["policy_loss"],
                "policy_games_used": policy_metrics["games_used"],
                "policy_games_skipped": policy_metrics["games_skipped"],
                "value_loss": value_metrics["value_loss"],
                "value_samples_used": value_metrics["samples_used"],
                "opponent_games": opponent_games,
                "opponent_pool_ids": [
                    snapshot.opponent_id for snapshot in opponent_pool
                ],
            }
        if starts is not None:
            history_record["start_games"] = {
                source_type: sum(
                    start.source_type == source_type for start in starts
                )
                for source_type in ("standard", "pgn", "mate_in_one")
            }
        history.append(history_record)
        save_stage5_checkpoint(
            last_checkpoint,
            policy_model=policy_model,
            value_model=value_model,
            opponent_model=opponent_model,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            config=config,
            completed_iteration=iteration_index + 1,
            history=history,
            source_checkpoint=source_checkpoint,
            collector_rng=move_rng,
            value_rng=value_rng,
            claim_draw=claim_draw,
            opponent_pool=opponent_pool,
            opponent_rng=opponent_rng,
            start_source=start_source,
            start_rng=start_rng,
            value_source=value_source,
        )

    return Stage5TrainResult(
        policy_model=policy_model,
        value_model=value_model,
        opponent_model=opponent_model,
        history=history,
        opponent_pool=opponent_pool,
    )
