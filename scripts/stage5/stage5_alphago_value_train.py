"""Train or explicitly test a Value-only model on an existing three-phase dataset.

This post-experiment entry point never collects games or rebuilds a dataset.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage5.stage5_value_io import (
    _atomic_json,
    _canonical_json,
    _sha256_bytes,
    _sha256_file,
)
from training.stage5.value_training import (
    _mse_by_label,
    _validation_predictions,
    train_value,
)
from training.stage5.checkpoint import load_value_only_checkpoint


PURPOSE = "stage5_alphago2016_chess_value_training"
COLLECTOR_PURPOSE = "stage5_value_alphago2016_chess"


def _git_provenance(
    script_path: Path, source_paths: Sequence[Path] = ()
) -> Dict[str, object]:
    paths = [script_path, *source_paths]
    source_sha256 = {
        str(path.resolve()): _sha256_file(path)
        for path in paths
        if path.is_file()
    }
    result: Dict[str, object] = {"source_sha256": source_sha256}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError) as exc:
        result.update(
            {"status": "unavailable", "head": None, "reason": type(exc).__name__}
        )
        return result
    result.update(
        {
            "status": "available",
            "head": head,
            "working_tree_diff_sha256": _sha256_bytes(diff),
            "working_tree_diff_command": "git diff --binary HEAD",
        }
    )
    return result


def load_three_phase_inputs(
    dataset_path: Path, manifest_path: Path
) -> tuple[Dict[str, object], Dict[str, object]]:
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("dataset.npz and manifest.json are both required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != COLLECTOR_PURPOSE:
        raise ValueError("Manifest is not from the three-phase collector")
    dataset_sha256 = _sha256_file(dataset_path)
    if manifest.get("dataset", {}).get("sha256") != dataset_sha256:
        raise ValueError("Dataset SHA256 differs from manifest")
    collector_config_sha256 = manifest.get("config_sha256")
    if not isinstance(collector_config_sha256, str) or len(collector_config_sha256) != 64:
        raise ValueError("Manifest config_sha256 is invalid")
    sources = manifest.get("config", {}).get("sources")
    if not isinstance(sources, dict) or sources.get("claim_draw") is not True:
        raise ValueError("Manifest lacks claim_draw=True source provenance")
    for name, stage in (("sl", 4), ("rl", 5)):
        source = sources.get(name)
        if not isinstance(source, dict) or source.get("stage") != stage:
            raise ValueError(f"Manifest {name.upper()} provenance is invalid")
        digest = source.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Manifest {name.upper()} SHA256 is invalid")
    if sources["rl"].get("baseline") != "zero" or sources["rl"].get("value_mode") != "disabled":
        raise ValueError("Three-phase RL source must use zero/disabled")
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {
            "train_states", "train_targets", "val_states", "val_targets",
            "test_states", "test_targets", "config_sha256",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Dataset arrays are missing: {sorted(missing)}")
        if str(data["config_sha256"].item()) != collector_config_sha256:
            raise ValueError("Dataset config hash differs from manifest")
        # Training validation intentionally inspects only train/val payloads.
        for split in ("train", "val"):
            states = data[f"{split}_states"]
            targets = data[f"{split}_targets"]
            if states.dtype != np.float32 or states.ndim != 4 or states.shape[1:] != (14, 8, 8):
                raise ValueError(f"{split}_states must be float32 NCHW (N,14,8,8)")
            if targets.dtype != np.float32 or targets.shape != (states.shape[0], 1):
                raise ValueError(f"{split}_targets shape/dtype mismatch")
            if not states.shape[0] or not np.isfinite(states).all() or not np.isfinite(targets).all():
                raise ValueError(f"{split} contains empty or non-finite data")
            if not np.isin(targets, (-1.0, 0.0, 1.0)).all():
                raise ValueError(f"{split} contains invalid Value targets")
    source_data = {
        "dataset_path": str(dataset_path).replace("\\", "/"),
        "dataset_sha256": dataset_sha256,
        "manifest_path": str(manifest_path).replace("\\", "/"),
        "manifest_sha256": _sha256_file(manifest_path),
        "collector_config_sha256": collector_config_sha256,
    }
    return sources, source_data


def training_config(args: argparse.Namespace, sources: Dict[str, object], source_data: Dict[str, object]) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "post_experiment_source_completion": True,
        "source_policies": sources,
        "source_data": source_data,
        "value_training": {
            "seed": args.value_seed,
            "optimizer": "Adam",
            "learning_rate": args.value_lr,
            "loss": "MSE",
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "selection": "lowest_validation_mse",
            "device": args.device,
        },
    }


def run_training(args: argparse.Namespace) -> None:
    sources, source_data = load_three_phase_inputs(args.dataset, args.manifest)
    config = training_config(args, sources, source_data)
    config_sha256 = _sha256_bytes(_canonical_json(config).encode("utf-8"))
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    provenance = _git_provenance(
        Path(__file__).resolve(),
        (
            REPO_ROOT / "training/stage5/value_training.py",
            REPO_ROOT / "training/stage5/updates.py",
            REPO_ROOT / "models/value_net.py",
        ),
    )
    train_value(
        args, config, config_sha256, sources["rl"]["sha256"],
        args.dataset, args.manifest,
        provenance=provenance,
        sha256_file=_sha256_file,
        atomic_json=_atomic_json,
        source_policies=sources,
        purpose=PURPOSE,
    )


def evaluate_test(args: argparse.Namespace) -> Dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
    sources, source_data = load_three_phase_inputs(args.dataset, args.manifest)
    model, checkpoint_provenance = load_value_only_checkpoint(
        args.checkpoint, device=args.device, claim_draw=True
    )
    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_sources = checkpoint_payload.get("source_policies")
    if not isinstance(checkpoint_sources, dict):
        raise ValueError("Value checkpoint lacks dual SL/RL provenance")
    for name in ("sl", "rl"):
        if checkpoint_sources.get(name, {}).get("sha256") != sources[name]["sha256"]:
            raise ValueError(f"Value checkpoint {name.upper()} source differs from dataset")
    checkpoint_data = checkpoint_payload.get("source_data")
    if not isinstance(checkpoint_data, dict) or checkpoint_data.get("dataset_sha256") != source_data["dataset_sha256"]:
        raise ValueError("Value checkpoint source dataset differs from evaluation dataset")
    with np.load(args.dataset, allow_pickle=False) as data:
        test_states = data["test_states"].copy()
        test_targets = data["test_targets"].copy()
        train_targets = data["train_targets"].copy()
    if test_states.dtype != np.float32 or test_states.shape[1:] != (14, 8, 8):
        raise ValueError("test_states must be float32 NCHW (N,14,8,8)")
    if test_targets.dtype != np.float32 or test_targets.shape != (test_states.shape[0], 1):
        raise ValueError("test_targets shape/dtype mismatch")
    if not test_states.shape[0] or not np.isfinite(test_states).all() or not np.isfinite(test_targets).all():
        raise ValueError("Test split is empty or non-finite")
    predictions = _validation_predictions(
        model, torch.from_numpy(test_states), torch.device(args.device), args.batch_size
    ).numpy()
    train_mean = float(train_targets.mean())
    payload = {
        "schema_version": 1,
        "purpose": "stage5_value_explicit_test_evaluation",
        "checkpoint": checkpoint_provenance,
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "source_policies": sources,
        "source_data": source_data,
        "test_samples": int(test_targets.shape[0]),
        "metrics": {
            "value": _mse_by_label(predictions, test_targets),
            "always_zero": _mse_by_label(np.zeros_like(test_targets), test_targets),
            "always_train_target_mean": {
                "constant": train_mean,
                **_mse_by_label(np.full_like(test_targets, train_mean), test_targets),
            },
        },
    }
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train from an existing dataset only")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--checkpoint-dir", required=True, type=Path)
    train.add_argument("--device", default="cuda")
    train.add_argument("--value-seed", type=int, default=2072)
    train.add_argument("--value-lr", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--max-epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=10)
    test = subparsers.add_parser("evaluate-test", help="Explicitly open and score test")
    test.add_argument("--dataset", required=True, type=Path)
    test.add_argument("--manifest", required=True, type=Path)
    test.add_argument("--checkpoint", required=True, type=Path)
    test.add_argument("--output", required=True, type=Path)
    test.add_argument("--device", default="cpu")
    test.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        if not math.isfinite(args.value_lr) or args.value_lr <= 0:
            raise ValueError("value_lr must be finite and positive")
        for name in ("value_seed", "batch_size", "max_epochs", "patience"):
            if getattr(args, name) <= 0:
                raise ValueError(f"{name} must be positive")
        run_training(args)
    else:
        if args.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        result = evaluate_test(args)
        print(json.dumps(result["metrics"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
