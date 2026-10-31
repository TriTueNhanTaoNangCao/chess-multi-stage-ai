"""Kaggle entry point for the fixed Stage 5 AlphaGo-style Value pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


EXPECTED_SL_SHA256 = "28e3c34619255796fc58a2006e7869a533812046a3b52366e8d8fe3925d7d464"
EXPECTED_RL_SHA256 = "1b3480e25cbf4b55cafeae85f0915a409993b7b2d08b068e35bb9cd056b244e0"
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/stage5_alphago_value_2071")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_metadata(sl_path: Path, rl_path: Path) -> None:
    sl = torch.load(sl_path, map_location="cpu", weights_only=False)
    rl = torch.load(rl_path, map_location="cpu", weights_only=False)
    if not isinstance(sl, dict) or not isinstance(rl, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    expected_common = {
        "model_class": "CNNPolicyNetwork",
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "claim_draw": True,
    }
    for key, expected in expected_common.items():
        if sl.get(key) != expected or rl.get(key) != expected:
            raise ValueError(f"Checkpoint metadata mismatch for {key}")
    if sl.get("stage") != 4 or rl.get("stage") != 5:
        raise ValueError("Expected Stage 4 SL and Stage 5 RL checkpoints")
    training_config = rl.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("RL checkpoint lacks training_config")
    if training_config.get("baseline") != "zero":
        raise ValueError("RL checkpoint baseline must be 'zero'")
    if rl.get("value_mode", training_config.get("value_mode")) != "disabled":
        raise ValueError("RL checkpoint value_mode must be 'disabled'")


def validate_resume_shards(
    shard_dir: Path, config_sha256: str
) -> list[Path]:
    candidates = sorted(shard_dir.glob("shard_*.npz"))
    if not candidates and (shard_dir / "shards").is_dir():
        shard_dir = shard_dir / "shards"
        candidates = sorted(shard_dir.glob("shard_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No resume shards found under {shard_dir}")
    if len(candidates) > 32:
        raise ValueError("Resume input contains more than 32 shards")
    for index, path in enumerate(candidates):
        if path.name != f"shard_{index:05d}.npz":
            raise ValueError("Resume shard sequence is not contiguous from zero")
        with np.load(path, allow_pickle=False) as data:
            checks = {
                "shard_index": int(data["shard_index"].item()) == index,
                "games_completed": int(data["games_completed"].item()) == (index + 1) * 512,
                "config_sha256": str(data["config_sha256"].item()) == config_sha256,
                "sl_checkpoint_sha256": str(data["sl_checkpoint_sha256"].item()) == EXPECTED_SL_SHA256,
                "rl_checkpoint_sha256": str(data["rl_checkpoint_sha256"].item()) == EXPECTED_RL_SHA256,
                "rng_state": "collection_rng_state_json" in data.files,
            }
            failed = [name for name, valid in checks.items() if not valid]
            if failed:
                raise ValueError(f"Invalid resume shard {path}: {failed}")
    return candidates


def copy_resume_shards(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if list(destination.glob("shard_*.npz")):
        raise FileExistsError("Writable output already contains shards; omit --resume-shards")
    for source in paths:
        target = destination / source.name
        temporary = destination / f".{source.name}.tmp"
        try:
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != sha256_file(source):
                raise RuntimeError(f"Resume shard copy hash mismatch: {source}")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume-shards",
        type=Path,
        help="Read-only Kaggle input directory containing prior shard_*.npz files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    bundle_root = args.bundle_root.resolve()
    source_root = bundle_root / "source"
    sl_path = bundle_root / "checkpoints/stage4/best_lichess.pt"
    rl_path = bundle_root / "checkpoints/stage5/batch_comparison/b16/best.pt"
    collector = source_root / "scripts/stage5/stage5_alphago_value_data.py"
    for path in (source_root, sl_path, rl_path, collector):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; enable a Kaggle GPU accelerator")
    if sha256_file(sl_path) != EXPECTED_SL_SHA256:
        raise ValueError("SL checkpoint SHA256 mismatch")
    if sha256_file(rl_path) != EXPECTED_RL_SHA256:
        raise ValueError("RL checkpoint SHA256 mismatch")
    validate_checkpoint_metadata(sl_path, rl_path)

    sys.path.insert(0, str(source_root))
    from scripts.stage5.stage5_alphago_value_data import experiment_config
    from scripts.stage5.stage5_value_io import _canonical_json, _sha256_bytes

    provenance = {
        "sl": {
            "path": str(sl_path), "sha256": EXPECTED_SL_SHA256, "stage": 4,
            "model_class": "CNNPolicyNetwork", "encoder_version": "hwc14-white-v1",
            "action_version": "az73-rankflip-v1",
        },
        "rl": {
            "path": str(rl_path), "sha256": EXPECTED_RL_SHA256, "stage": 5,
            "model_class": "CNNPolicyNetwork", "encoder_version": "hwc14-white-v1",
            "action_version": "az73-rankflip-v1", "baseline": "zero",
            "value_mode": "disabled",
        },
        "claim_draw": True,
    }
    fixed = argparse.Namespace(
        total_games=16_384, shard_size=512, seed=2071, u_min=1, u_max=100,
        max_plies=300,
    )
    config = experiment_config(fixed, provenance)
    config_sha256 = _sha256_bytes(_canonical_json(config).encode("utf-8"))

    if args.resume_shards is not None:
        paths = validate_resume_shards(args.resume_shards, config_sha256)
        copy_resume_shards(paths, output_dir / "shards")
        print(f"RESUME copied_and_validated={len(paths)}", flush=True)

    command = [
        sys.executable, "-u", str(collector),
        "--sl-checkpoint", str(sl_path), "--rl-checkpoint", str(rl_path),
        "--u-min", "1", "--u-max", "100", "--data-dir", str(output_dir),
        "--total-games", "16384", "--shard-size", "512",
        "--max-plies", "300", "--seed", "2071", "--device", "cuda",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    started = time.perf_counter()
    print("JOB_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    subprocess.run(command, check=True, cwd=source_root, env=environment)
    job_seconds = time.perf_counter() - started

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("collection", {}).setdefault("timing", {})[
        "job_wall_seconds"
    ] = job_seconds
    temporary = manifest_path.with_name(".manifest.json.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"JOB_COMPLETE total_seconds={job_seconds:.3f} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
