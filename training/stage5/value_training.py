"""Value-only training loop and metric helpers for Stage 5."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Callable, Dict

import numpy as np
import torch

from models.value_net import ValueNetwork
from training.stage5.types import ValueBatch
from training.stage5.updates import update_value_network


ENCODER_VERSION = "hwc14-white-v1"


def _validation_predictions(
    model: torch.nn.Module,
    states: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    previous_mode = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            chunks.append(model(states[start : start + batch_size].to(device)).cpu())
    model.train(previous_mode)
    return torch.cat(chunks, dim=0)


def _mse_by_label(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float | None]:
    predictions = predictions.reshape(-1).astype(np.float64)
    targets = targets.reshape(-1).astype(np.float64)
    result: Dict[str, float | None] = {}
    for name, mask in (
        ("all", np.ones(targets.shape, dtype=np.bool_)),
        ("label_-1", targets == -1.0),
        ("label_0", targets == 0.0),
        ("label_+1", targets == 1.0),
        ("draw_0", targets == 0.0),
        ("decisive_pm1", targets != 0.0),
    ):
        result[name] = (
            float(np.mean((predictions[mask] - targets[mask]) ** 2))
            if mask.any()
            else None
        )
    return result


def train_value(
    args: argparse.Namespace,
    config: Dict[str, object],
    config_sha256: str,
    source_sha256: str,
    dataset_path: Path,
    manifest_path: Path,
    *,
    provenance: Dict[str, object],
    sha256_file: Callable[[Path], str],
    atomic_json: Callable[[Path, Dict[str, object]], None],
    source_policies: Dict[str, object] | None = None,
    purpose: str = "stage5_value_v2_training_summary",
) -> None:
    best_path = args.checkpoint_dir / "best.pt"
    summary_path = args.checkpoint_dir / "training_summary.json"
    if best_path.exists() or summary_path.exists():
        if not best_path.is_file() or not summary_path.is_file():
            raise FileExistsError("Value checkpoint output is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("config_sha256") != config_sha256:
            raise ValueError("Existing training summary config mismatch")
        if summary.get("best_checkpoint", {}).get("sha256") != sha256_file(best_path):
            raise ValueError("Existing best checkpoint hash mismatch")
        print("TRAINING already complete", flush=True)
        return
    if args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir()):
        raise FileExistsError("Checkpoint directory contains unexpected files")

    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = sha256_file(dataset_path)
    if data_manifest["dataset"]["sha256"] != dataset_sha256:
        raise ValueError("Dataset hash differs from manifest")
    with np.load(dataset_path, allow_pickle=False) as data:
        train_states = data["train_states"].copy()
        train_targets = data["train_targets"].copy()
        val_states = data["val_states"].copy()
        val_targets = data["val_targets"].copy()
        # Test arrays are intentionally not opened here.
    if not train_states.shape[0] or not val_states.shape[0]:
        raise ValueError("Train and validation splits must be non-empty")

    random.seed(args.value_seed)
    np.random.seed(args.value_seed)
    torch.manual_seed(args.value_seed)
    torch.cuda.manual_seed_all(args.value_seed)
    device = torch.device(getattr(args, "device", "cuda"))
    model = ValueNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.value_lr)
    shuffle_rng = torch.Generator(device="cpu")
    shuffle_rng.manual_seed(args.value_seed)
    train_states_t = torch.from_numpy(train_states)
    train_targets_t = torch.from_numpy(train_targets)
    val_states_t = torch.from_numpy(val_states)
    val_targets_t = torch.from_numpy(val_targets)

    history = []
    best_val = math.inf
    best_epoch = None
    best_state = None
    bad_epochs = 0
    stop_reason = "max_epochs"
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_states_t.shape[0], generator=shuffle_rng)
        loss_sum = 0.0
        seen = 0
        for start in range(0, permutation.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            states_batch = train_states_t[indices]
            targets_batch = train_targets_t[indices]
            metrics = update_value_network(
                model,
                optimizer,
                ValueBatch(
                    states=states_batch,
                    targets=targets_batch,
                    games_used=int(states_batch.shape[0]),
                    games_skipped=0,
                    games_truncated=0,
                ),
                device=device,
            )
            count = int(states_batch.shape[0])
            loss_sum += float(metrics["value_loss"]) * count
            seen += count
        train_mse = loss_sum / seen
        predictions = _validation_predictions(
            model, val_states_t, device, args.batch_size
        )
        val_mse = float(torch.mean((predictions - val_targets_t) ** 2).item())
        if not math.isfinite(train_mse) or not math.isfinite(val_mse):
            raise RuntimeError("Non-finite Value loss")
        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "improved": improved,
            }
        )
        print(
            f"EPOCH {epoch:03d} train_mse={train_mse:.8f} "
            f"val_mse={val_mse:.8f} best={best_val:.8f}@{best_epoch} "
            f"bad={bad_epochs}/{args.patience}",
            flush=True,
        )
        if bad_epochs >= args.patience:
            stop_reason = f"early_stopping_patience_{args.patience}"
            break
    training_seconds = time.perf_counter() - started
    if best_state is None or best_epoch is None:
        raise RuntimeError("No finite Value checkpoint selected")

    model.load_state_dict(best_state, strict=True)
    best_predictions = _validation_predictions(
        model, val_states_t, device, args.batch_size
    ).numpy()
    train_mean = float(train_targets.mean())
    zero_predictions = np.zeros_like(val_targets, dtype=np.float64)
    mean_predictions = np.full_like(val_targets, train_mean, dtype=np.float64)
    validation = {
        "value_best": _mse_by_label(best_predictions, val_targets),
        "always_zero": _mse_by_label(zero_predictions, val_targets),
        "always_train_target_mean": {
            "constant": train_mean,
            **_mse_by_label(mean_predictions, val_targets),
        },
        "prediction_distribution": {
            "mean": float(np.mean(best_predictions)),
            "std": float(np.std(best_predictions)),
            "min": float(np.min(best_predictions)),
            "max": float(np.max(best_predictions)),
        },
    }
    source_policy = (
        dict(source_policies["rl"])
        if source_policies is not None
        else {
            "path": str(args.stage4_checkpoint).replace("\\", "/"),
            "sha256": source_sha256,
        }
    )
    checkpoint = {
        "format_version": 1,
        "stage": 5,
        "checkpoint_type": "value_only",
        "purpose": purpose,
        "model_class": "ValueNetwork",
        "model_config": model.get_config(),
        "state_dict": best_state,
        "encoder_version": ENCODER_VERSION,
        "value_perspective": "side_to_move",
        "claim_draw": True,
        "source_policy": source_policy,
        "source_data": {
            "dataset_path": str(dataset_path).replace("\\", "/"),
            "dataset_sha256": dataset_sha256,
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "config_sha256": config_sha256,
        "config": config,
        "provenance": provenance,
        "seed": args.value_seed,
        "epoch": best_epoch,
        "val_mse": best_val,
        "training_config": config["value_training"],
    }
    if source_policies is not None:
        checkpoint["source_policies"] = source_policies
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_temporary = best_path.with_name(".best.pt.tmp")
    try:
        torch.save(checkpoint, best_temporary)
        loaded = torch.load(best_temporary, map_location="cpu", weights_only=False)
        reconstructed = ValueNetwork.from_config(loaded["model_config"])
        reconstructed.load_state_dict(loaded["state_dict"], strict=True)
        if loaded.get("checkpoint_type") != "value_only":
            raise RuntimeError("Strict Value checkpoint verification failed")
        os.replace(best_temporary, best_path)
    finally:
        if best_temporary.exists():
            best_temporary.unlink()
    best_sha256 = sha256_file(best_path)
    summary = {
        "schema_version": 1,
        "purpose": purpose,
        "checkpoint_type": "value_only",
        "config": config,
        "config_sha256": config_sha256,
        "source_policy": checkpoint["source_policy"],
        "source_data": checkpoint["source_data"],
        "git": provenance,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "stop_reason": stop_reason,
        "history": history,
        "validation": validation,
        "test_status": "held_out_not_evaluated",
        "timing_seconds": training_seconds,
        "best_checkpoint": {
            "path": str(best_path).replace("\\", "/"),
            "sha256": best_sha256,
        },
    }
    if source_policies is not None:
        summary["source_policies"] = source_policies
    atomic_json(summary_path, summary)
    print(
        "TRAINING_RESULT="
        + json.dumps(
            {
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_val_mse": best_val,
                "stop_reason": stop_reason,
                "validation": validation,
                "seconds": training_seconds,
                "best_sha256": best_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
