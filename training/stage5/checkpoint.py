"""Stage 5 checkpoint serialization and strict resume loading."""
from __future__ import annotations
import copy
import hashlib
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from games.chess.encoder import NUM_ACTIONS, NUM_CHANNELS
from models.cnn_policy import CNNPolicyNetwork
from models.value_net import ValueNetwork
from training.stage5.types import OpponentSnapshot, Stage5CheckpointState, Stage5TrainConfig

def _serialized_training_config(config: Stage5TrainConfig) -> Dict[str, object]:
    values = asdict(config)
    for key in (
        "stage4_checkpoint",
        "policy_init_checkpoint",
        "output_dir",
        "resume_checkpoint",
        "mixed_start_manifest",
        "value_checkpoint",
    ):
        if values[key] is not None:
            values[key] = str(values[key])
    return values


def load_stage5_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CNNPolicyNetwork, Dict[str, object]]:
    """Load only Policy weights and provenance from a Stage 5 checkpoint.

    Optimizers, RNG state, opponent pool, history, and embedded Value state are
    intentionally ignored so the caller can start an independent fresh run.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Stage 5 Policy init checkpoint not found: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"Failed to load Stage 5 Policy init checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid Stage 5 Policy init checkpoint payload")
    expected = {
        "format_version": 1,
        "stage": 5,
        "model_class": "CNNPolicyNetwork",
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": NUM_ACTIONS,
        "value_perspective": "side_to_move",
        "value_model_class": "ValueNetwork",
        "opponent_model_class": "CNNPolicyNetwork",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Incompatible Stage 5 Policy init {key}: "
                f"{payload.get(key)!r} (expected {value!r})"
            )
    if not isinstance(payload.get("claim_draw"), bool):
        raise ValueError("Stage 5 Policy init claim_draw must be boolean")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or (
        model_config.get("in_channels") != NUM_CHANNELS
        or model_config.get("num_actions") != NUM_ACTIONS
    ):
        raise ValueError("Stage 5 Policy init model_config is incompatible")
    required_fields = (
        "state_dict",
        "value_model_config",
        "value_state_dict",
        "opponent_model_config",
        "opponent_state_dict",
        "policy_optimizer_state_dict",
        "value_optimizer_state_dict",
    )
    missing = [key for key in required_fields if key not in payload]
    if missing:
        raise ValueError(f"Stage 5 Policy checkpoint missing fields: {missing}")
    completed_iteration = payload.get("completed_iteration")
    if not isinstance(completed_iteration, int) or completed_iteration < 0:
        raise ValueError("Stage 5 Policy init completed_iteration is invalid")
    try:
        with torch.random.fork_rng(devices=[]):
            model = CNNPolicyNetwork.from_config(model_config)
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        raise ValueError(
            f"Failed strict Stage 5 Policy init reconstruction: {exc}"
        ) from exc
    model.to(torch.device(device)).eval()
    training_config = payload.get("training_config")
    provenance: Dict[str, object] = {
        "kind": "stage5_policy_init",
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "format_version": payload["format_version"],
        "stage": payload["stage"],
        "model_class": payload["model_class"],
        "model_config": copy.deepcopy(model_config),
        "encoder_version": payload["encoder_version"],
        "action_version": payload["action_version"],
        "num_actions": payload["num_actions"],
        "value_perspective": payload["value_perspective"],
        "claim_draw": payload["claim_draw"],
        "completed_iteration": completed_iteration,
        "seed": payload.get("seed"),
        "policy_lr": (
            training_config.get("policy_lr")
            if isinstance(training_config, dict) else None
        ),
        "baseline": (
            training_config.get("baseline")
            if isinstance(training_config, dict) else None
        ),
        "value_mode": payload.get(
            "value_mode",
            training_config.get("value_mode")
            if isinstance(training_config, dict) else None,
        ),
    }
    return model, provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_value_only_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    claim_draw: bool,
) -> tuple[ValueNetwork, Dict[str, object]]:
    """Strictly load a standalone Value checkpoint for a frozen baseline."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Value-only checkpoint not found: {checkpoint_path}"
        )
    if not isinstance(claim_draw, bool):
        raise ValueError("claim_draw compatibility policy must be boolean")
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to load Value-only checkpoint {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid Value-only checkpoint payload")
    expected = {
        "format_version": 1,
        "stage": 5,
        "checkpoint_type": "value_only",
        "model_class": "ValueNetwork",
        "encoder_version": "hwc14-white-v1",
        "value_perspective": "side_to_move",
        "claim_draw": claim_draw,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Incompatible Value-only checkpoint {key}: "
                f"{payload.get(key)!r} (expected {value!r})"
            )
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or (
        model_config.get("in_channels") != NUM_CHANNELS
    ):
        raise ValueError(
            "Value-only model_config is incompatible with the encoder"
        )
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Value-only checkpoint is missing state_dict")
    try:
        model = ValueNetwork.from_config(model_config)
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        raise ValueError(
            f"Failed strict Value-only model reconstruction: {exc}"
        ) from exc
    model.to(torch.device(device)).eval().requires_grad_(False)
    provenance: Dict[str, object] = {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "format_version": payload["format_version"],
        "stage": payload["stage"],
        "checkpoint_type": payload["checkpoint_type"],
        "model_class": payload["model_class"],
        "model_config": copy.deepcopy(model_config),
        "encoder_version": payload["encoder_version"],
        "value_perspective": payload["value_perspective"],
        "claim_draw": payload["claim_draw"],
        "epoch": payload.get("epoch"),
        "source_policy": copy.deepcopy(payload.get("source_policy")),
        "source_data": copy.deepcopy(payload.get("source_data")),
    }
    return model, provenance

def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

def save_stage5_checkpoint(
    path: str | Path,
    *,
    policy_model: CNNPolicyNetwork,
    value_model: ValueNetwork,
    opponent_model: CNNPolicyNetwork,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    config: Stage5TrainConfig,
    completed_iteration: int,
    history: Sequence[Dict[str, object]],
    source_checkpoint: Dict[str, object],
    collector_rng: torch.Generator,
    value_rng: np.random.Generator,
    claim_draw: bool,
    opponent_pool: Optional[Sequence[OpponentSnapshot]] = None,
    opponent_rng: Optional[np.random.Generator] = None,
    start_source: Optional[Dict[str, object]] = None,
    start_rng: Optional[np.random.Generator] = None,
    value_source: Optional[Dict[str, object]] = None,
) -> None:
    """Atomically save the single resumable Stage 5 ``last.pt`` payload."""
    if not isinstance(policy_model, CNNPolicyNetwork):
        raise TypeError("policy_model must be CNNPolicyNetwork")
    if not isinstance(value_model, ValueNetwork):
        raise TypeError("value_model must be ValueNetwork")
    if not isinstance(opponent_model, CNNPolicyNetwork):
        raise TypeError("opponent_model must be CNNPolicyNetwork")
    if not isinstance(completed_iteration, int) or completed_iteration < 0:
        raise ValueError("completed_iteration must be non-negative")
    if len(history) != completed_iteration:
        raise ValueError("history length must equal completed_iteration")
    if not isinstance(claim_draw, bool):
        raise ValueError("claim_draw must be boolean")
    pool = list(opponent_pool) if opponent_pool is not None else [
        OpponentSnapshot("stage4", 0, opponent_model)
    ]
    if not pool or pool[0].opponent_id != "stage4":
        raise ValueError("Opponent pool must keep stage4 as its first entry")
    if len(pool) > config.opponent_pool_size:
        raise ValueError("Opponent pool exceeds opponent_pool_size")
    if len({snapshot.opponent_id for snapshot in pool}) != len(pool):
        raise ValueError("Opponent pool IDs must be unique")
    if config.opponent_mode == "pool" and opponent_rng is None:
        raise ValueError("Pool mode requires opponent_rng")
    if config.start_mode == "mixed":
        if not isinstance(start_source, dict) or start_source.get("mode") != "mixed":
            raise ValueError("Mixed-start checkpoint requires source metadata")
        if "positions" in start_source:
            raise ValueError("Mixed-start checkpoint must not embed manifest positions")
        if start_rng is None:
            raise ValueError("Mixed-start checkpoint requires source RNG")
    else:
        if start_source is None:
            start_source = {"mode": "standard"}
        if not isinstance(start_source, dict) or start_source.get("mode") != "standard":
            raise ValueError("Standard checkpoint requires standard source metadata")
        if start_rng is not None:
            raise ValueError("Standard checkpoint must not save source RNG state")
    if config.value_mode == "frozen":
        if not isinstance(value_source, dict):
            raise ValueError("Frozen Value checkpoint requires source provenance")
        expected_value_source = {
            "checkpoint_type": "value_only",
            "model_class": "ValueNetwork",
            "encoder_version": "hwc14-white-v1",
            "value_perspective": "side_to_move",
            "claim_draw": claim_draw,
        }
        for key, expected in expected_value_source.items():
            if value_source.get(key) != expected:
                raise ValueError(f"Invalid frozen Value source {key}")
        if value_source.get("model_config") != value_model.get_config():
            raise ValueError("Frozen Value source config does not match Value model")
        source_sha256 = value_source.get("sha256")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError("Frozen Value source has invalid SHA256")
    elif value_source is not None:
        raise ValueError("Non-frozen Value mode must not save source provenance")

    payload: Dict[str, object] = {
        "format_version": 1,
        "stage": 5,
        "model_class": "CNNPolicyNetwork",
        "model_config": policy_model.get_config(),
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": NUM_ACTIONS,
        "value_perspective": "side_to_move",
        "claim_draw": claim_draw,
        "seed": config.seed,
        "step": completed_iteration,
        "completed_iteration": completed_iteration,
        "training_config": _serialized_training_config(config),
        "history": list(history),
        "state_dict": policy_model.state_dict(),
        "value_model_class": "ValueNetwork",
        "value_model_config": value_model.get_config(),
        "value_state_dict": value_model.state_dict(),
        "opponent_model_class": "CNNPolicyNetwork",
        "opponent_model_config": opponent_model.get_config(),
        "opponent_state_dict": opponent_model.state_dict(),
        "value_mode": config.value_mode,
        "value_source": copy.deepcopy(value_source),
        "opponent_mode": config.opponent_mode,
        "opponent_pool": (
            [
                {
                    "id": snapshot.opponent_id,
                    "iteration": snapshot.iteration,
                    "model_class": "CNNPolicyNetwork",
                    "model_config": snapshot.model.get_config(),
                    "state_dict": snapshot.model.state_dict(),
                }
                for snapshot in pool
            ]
            if config.opponent_mode == "pool"
            else None
        ),
        "opponent_rng_state": (
            copy.deepcopy(opponent_rng.bit_generator.state)
            if opponent_rng is not None
            else None
        ),
        "start_mode": config.start_mode,
        "start_source": copy.deepcopy(start_source),
        "start_rng_state": (
            copy.deepcopy(start_rng.bit_generator.state)
            if start_rng is not None
            else None
        ),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "value_optimizer_state_dict": value_optimizer.state_dict(),
        "source_checkpoint": copy.deepcopy(source_checkpoint),
        "collector_rng_state": collector_rng.get_state().cpu(),
        "value_rng_state": copy.deepcopy(value_rng.bit_generator.state),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state_all": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

def load_stage5_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Stage5CheckpointState:
    """Strictly reconstruct a resumable Stage 5 checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage 5 checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Invalid Stage 5 checkpoint payload")

    expected = {
        "format_version": 1,
        "stage": 5,
        "model_class": "CNNPolicyNetwork",
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": NUM_ACTIONS,
        "value_perspective": "side_to_move",
        "value_model_class": "ValueNetwork",
        "opponent_model_class": "CNNPolicyNetwork",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Incompatible Stage 5 checkpoint {key}: "
                f"{payload.get(key)!r} (expected {value!r})"
            )
    if not isinstance(payload.get("claim_draw"), bool):
        raise ValueError("Stage 5 checkpoint claim_draw must be boolean")

    required_fields = (
        "model_config",
        "state_dict",
        "value_model_config",
        "value_state_dict",
        "opponent_model_config",
        "opponent_state_dict",
        "policy_optimizer_state_dict",
        "value_optimizer_state_dict",
        "training_config",
        "completed_iteration",
        "history",
        "source_checkpoint",
        "collector_rng_state",
        "value_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    )
    missing = [key for key in required_fields if key not in payload]
    if missing:
        raise ValueError(f"Stage 5 checkpoint missing fields: {missing}")
    if payload["opponent_model_config"] != payload["model_config"]:
        raise ValueError("Opponent model config must match policy model config")
    model_config = payload["model_config"]
    value_model_config = payload["value_model_config"]
    if not isinstance(model_config, dict) or (
        model_config.get("in_channels") != NUM_CHANNELS
        or model_config.get("num_actions") != NUM_ACTIONS
    ):
        raise ValueError("Policy model config is incompatible with the encoder/action space")
    if not isinstance(value_model_config, dict) or (
        value_model_config.get("in_channels") != NUM_CHANNELS
    ):
        raise ValueError("Value model config is incompatible with the encoder")
    completed_iteration = payload["completed_iteration"]
    history = payload["history"]
    if not isinstance(completed_iteration, int) or completed_iteration < 0:
        raise ValueError("Invalid completed_iteration in Stage 5 checkpoint")
    if not isinstance(history, list) or len(history) != completed_iteration:
        raise ValueError("Stage 5 history does not match completed_iteration")
    if not isinstance(payload["training_config"], dict):
        raise ValueError("Invalid Stage 5 training_config")
    training_config = payload["training_config"]
    value_mode = payload.get(
        "value_mode", training_config.get("value_mode", "joint")
    )
    if value_mode not in {"joint", "disabled", "frozen"}:
        raise ValueError("Invalid value_mode in Stage 5 checkpoint")
    if (
        "value_mode" in payload
        and training_config.get("value_mode", "joint") != value_mode
    ):
        raise ValueError("Value mode metadata does not match training_config")
    if value_mode == "disabled" and training_config.get("baseline") != "zero":
        raise ValueError("Disabled Value checkpoint requires baseline='zero'")
    if value_mode == "frozen" and training_config.get("baseline") != "value":
        raise ValueError("Frozen Value checkpoint requires baseline='value'")
    # Normalize legacy format-version-1 checkpoints in memory only.
    training_config["value_mode"] = value_mode
    training_config.setdefault("value_checkpoint", None)
    payload["value_mode"] = value_mode
    value_source = payload.get("value_source")
    if value_mode == "frozen":
        if not isinstance(value_source, dict):
            raise ValueError("Frozen Stage 5 checkpoint lacks Value provenance")
        expected_value_source = {
            "checkpoint_type": "value_only",
            "model_class": "ValueNetwork",
            "model_config": value_model_config,
            "encoder_version": "hwc14-white-v1",
            "value_perspective": "side_to_move",
            "claim_draw": payload["claim_draw"],
        }
        for key, expected_value in expected_value_source.items():
            if value_source.get(key) != expected_value:
                raise ValueError(
                    f"Frozen Value provenance mismatch for {key}"
                )
        source_sha256 = value_source.get("sha256")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError("Frozen Value provenance has invalid SHA256")
    elif value_source is not None:
        raise ValueError("Non-frozen checkpoint contains Value provenance")
    payload["value_source"] = copy.deepcopy(value_source)
    opponent_mode = payload.get(
        "opponent_mode", training_config.get("opponent_mode", "fixed")
    )
    if opponent_mode not in {"fixed", "pool"}:
        raise ValueError("Invalid opponent_mode in Stage 5 checkpoint")
    if (
        "opponent_mode" in payload
        and training_config.get("opponent_mode", "fixed") != opponent_mode
    ):
        raise ValueError("Opponent mode metadata does not match training_config")
    start_mode = payload.get(
        "start_mode", training_config.get("start_mode", "standard")
    )
    if start_mode not in {"standard", "mixed"}:
        raise ValueError("Invalid start_mode in Stage 5 checkpoint")
    if (
        "start_mode" in payload
        and training_config.get("start_mode", "standard") != start_mode
    ):
        raise ValueError("Start mode metadata does not match training_config")
    training_config["start_mode"] = start_mode
    training_config.setdefault("mixed_start_manifest", None)
    payload["start_mode"] = start_mode
    start_source = payload.get("start_source")
    start_rng_state = payload.get("start_rng_state")
    if start_mode == "standard":
        if start_source is None:
            start_source = {"mode": "standard"}
        if not isinstance(start_source, dict) or start_source.get("mode") != "standard":
            raise ValueError("Invalid standard start-source metadata")
        if start_rng_state is not None:
            raise ValueError("Standard checkpoint must not contain start RNG state")
    else:
        if not isinstance(start_source, dict) or start_source.get("mode") != "mixed":
            raise ValueError("Mixed checkpoint is missing start-source metadata")
        if not isinstance(start_source.get("manifest_sha256"), str):
            raise ValueError("Mixed checkpoint has invalid manifest SHA256")
        if not isinstance(start_source.get("mix"), dict):
            raise ValueError("Mixed checkpoint has invalid mix metadata")
        if not isinstance(start_rng_state, dict):
            raise ValueError("Mixed checkpoint is missing start RNG state")
    payload["start_source"] = copy.deepcopy(start_source)
    payload["start_rng_state"] = copy.deepcopy(start_rng_state)
    if payload.get("step") != completed_iteration:
        raise ValueError("Stage 5 step does not match completed_iteration")
    if payload["training_config"].get("seed") != payload.get("seed"):
        raise ValueError("Stage 5 seed metadata does not match training_config")
    if not isinstance(payload["source_checkpoint"], dict):
        raise ValueError("Invalid Stage 5 source_checkpoint")
    if not isinstance(payload["collector_rng_state"], torch.Tensor):
        raise ValueError("Invalid collector RNG state")
    if not isinstance(payload["value_rng_state"], dict):
        raise ValueError("Invalid value RNG state")

    try:
        policy_model = CNNPolicyNetwork.from_config(payload["model_config"])
        policy_model.load_state_dict(payload["state_dict"], strict=True)
        value_model = ValueNetwork.from_config(payload["value_model_config"])
        value_model.load_state_dict(payload["value_state_dict"], strict=True)
        opponent_model = CNNPolicyNetwork.from_config(
            payload["opponent_model_config"]
        )
        opponent_model.load_state_dict(payload["opponent_state_dict"], strict=True)
    except Exception as exc:
        raise ValueError(f"Failed strict Stage 5 model reconstruction: {exc}") from exc

    try:
        policy_lr = float(training_config["policy_lr"])
        value_lr = float(training_config["value_lr"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint training_config lacks valid learning rates") from exc

    device_obj = torch.device(device)
    policy_model.to(device_obj).eval()
    if value_mode == "frozen":
        value_model.to(device_obj).eval().requires_grad_(False)
    else:
        value_model.to(device_obj).train()
    opponent_model.to(device_obj).eval().requires_grad_(False)
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=policy_lr)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=value_lr)
    try:
        policy_optimizer.load_state_dict(payload["policy_optimizer_state_dict"])
        value_optimizer.load_state_dict(payload["value_optimizer_state_dict"])
    except Exception as exc:
        raise ValueError(f"Failed Stage 5 optimizer reconstruction: {exc}") from exc
    _move_optimizer_state(policy_optimizer, device_obj)
    _move_optimizer_state(value_optimizer, device_obj)

    pool_payload = payload.get("opponent_pool")
    opponent_pool: List[OpponentSnapshot] = []
    if pool_payload is None:
        if opponent_mode != "fixed":
            raise ValueError("Pool checkpoint is missing opponent_pool")
        opponent_pool = [OpponentSnapshot("stage4", 0, opponent_model)]
    else:
        if not isinstance(pool_payload, list) or not pool_payload:
            raise ValueError("Invalid opponent_pool in Stage 5 checkpoint")
        seen_ids = set()
        for index, entry in enumerate(pool_payload):
            if not isinstance(entry, dict):
                raise ValueError("Invalid opponent pool entry")
            opponent_id = entry.get("id")
            iteration = entry.get("iteration")
            if not isinstance(opponent_id, str) or not opponent_id:
                raise ValueError("Invalid opponent pool ID")
            if opponent_id in seen_ids:
                raise ValueError("Duplicate opponent pool ID")
            if not isinstance(iteration, int) or iteration < 0:
                raise ValueError("Invalid opponent snapshot iteration")
            if entry.get("model_class") != "CNNPolicyNetwork":
                raise ValueError("Invalid opponent snapshot model_class")
            if entry.get("model_config") != payload["model_config"]:
                raise ValueError("Opponent snapshot config must match learner Policy")
            if index == 0:
                if opponent_id != "stage4" or iteration != 0:
                    raise ValueError("First opponent pool entry must be stage4")
                if entry.get("model_config") != payload["opponent_model_config"]:
                    raise ValueError("Stage 4 pool config does not match opponent model")
                try:
                    opponent_model.load_state_dict(entry["state_dict"], strict=True)
                except Exception as exc:
                    raise ValueError(f"Invalid Stage 4 pool state: {exc}") from exc
                snapshot_model = opponent_model
            else:
                if iteration <= opponent_pool[-1].iteration:
                    raise ValueError("Opponent snapshots must be in iteration order")
                try:
                    snapshot_model = CNNPolicyNetwork.from_config(
                        entry["model_config"]
                    )
                    snapshot_model.load_state_dict(entry["state_dict"], strict=True)
                except Exception as exc:
                    raise ValueError(
                        f"Failed strict opponent snapshot reconstruction: {exc}"
                    ) from exc
                snapshot_model.to(device_obj).eval().requires_grad_(False)
            seen_ids.add(opponent_id)
            opponent_pool.append(
                OpponentSnapshot(opponent_id, iteration, snapshot_model)
            )
        if opponent_mode == "fixed" and len(opponent_pool) != 1:
            raise ValueError("Fixed checkpoint must contain only the Stage 4 opponent")
        saved_pool_size = training_config.get("opponent_pool_size", 1)
        if len(opponent_pool) > saved_pool_size:
            raise ValueError("Checkpoint opponent pool exceeds configured size")

    opponent_rng_state = payload.get("opponent_rng_state")
    if opponent_mode == "pool" and not isinstance(opponent_rng_state, dict):
        raise ValueError("Pool checkpoint is missing opponent RNG state")
    return Stage5CheckpointState(
        policy_model=policy_model,
        value_model=value_model,
        opponent_model=opponent_model,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        opponent_mode=opponent_mode,
        opponent_pool=opponent_pool,
        opponent_rng_state=copy.deepcopy(opponent_rng_state),
        start_source=copy.deepcopy(start_source),
        start_rng_state=copy.deepcopy(start_rng_state),
        payload=payload,
    )

def _validate_resume_config(
    config: Stage5TrainConfig,
    saved_config: Dict[str, object],
    completed_iteration: int,
) -> None:
    saved_value_mode = saved_config.get("value_mode", "joint")
    if saved_value_mode != config.value_mode:
        raise ValueError(
            "Resume config mismatch for value_mode: "
            f"checkpoint={saved_value_mode!r}, "
            f"requested={config.value_mode!r}"
        )
    saved_opponent_mode = saved_config.get("opponent_mode", "fixed")
    if saved_opponent_mode != config.opponent_mode:
        raise ValueError(
            "Resume config mismatch for opponent_mode: "
            f"checkpoint={saved_opponent_mode!r}, "
            f"requested={config.opponent_mode!r}"
        )
    saved_start_mode = saved_config.get("start_mode", "standard")
    if saved_start_mode != config.start_mode:
        raise ValueError(
            "Resume config mismatch for start_mode: "
            f"checkpoint={saved_start_mode!r}, "
            f"requested={config.start_mode!r}"
        )
    compatible_fields = (
        "seed",
        "games_per_iteration",
        "max_plies",
        "chunk_size",
        "policy_lr",
        "value_lr",
        "baseline",
    )
    if config.opponent_mode == "pool":
        compatible_fields += (
            "opponent_snapshot_interval",
            "opponent_pool_size",
        )
    for field in compatible_fields:
        if saved_config.get(field) != getattr(config, field):
            raise ValueError(
                f"Resume config mismatch for {field}: "
                f"checkpoint={saved_config.get(field)!r}, "
                f"requested={getattr(config, field)!r}"
            )
    if config.iterations < completed_iteration:
        raise ValueError(
            "iterations is the total target and cannot be less than "
            f"completed_iteration={completed_iteration}"
        )
