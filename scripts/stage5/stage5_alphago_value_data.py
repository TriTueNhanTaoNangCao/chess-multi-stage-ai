"""Generate chess Value data with the AlphaGo 2016 SL/uniform/RL recipe.

This command performs collection and dataset splitting only.  Both checkpoints
and the inclusive one-based U range are mandatory; it never selects a best/last
checkpoint and never starts Value training.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage5.stage5_value_io import (
    _atomic_json,
    _atomic_npz,
    _canonical_json,
    _game_fingerprint,
    _sha256_bytes,
    _sha256_file,
    _state_fingerprint,
)
from training.stage5.alphago_value_data import (
    AlphaGoValueDataConfig,
    CollectionRNG,
    collect_value_game,
    load_value_data_policies,
)


SCHEMA_VERSION = 1
PURPOSE = "stage5_value_alphago2016_chess"


def experiment_config(args: argparse.Namespace, provenance: Dict[str, object]) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "sources": provenance,
        "collection": {
            "total_games": args.total_games,
            "shard_size": args.shard_size,
            "seed": args.seed,
            "u_min": args.u_min,
            "u_max": args.u_max,
            "u_semantics": "one_based_generated_ply; SL plays U-1, uniform move is U",
            "max_plies": args.max_plies,
            "max_plies_semantics": "generated plies from standard initial position",
            "claim_draw": provenance["claim_draw"],
            "policy_sampling": "legal_masked_temperature_1",
            "uniform_sampling": "uniform_over_all_legal_moves",
            "start_source": "standard_chess_only; no Arena validation data",
        },
        "split": {
            "ratios": [0.8, 0.1, 0.1],
            "seed": args.seed,
            "unit": "complete_game_fingerprint_group",
            "exact_tensor_cross_split_filter": True,
        },
    }


def validate_args(args: argparse.Namespace) -> None:
    AlphaGoValueDataConfig(
        u_min=args.u_min,
        u_max=args.u_max,
        max_plies=args.max_plies,
        claim_draw=True,
    ).validate()
    for name in ("total_games", "shard_size"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.total_games % args.shard_size:
        raise ValueError("total_games must be divisible by shard_size")
    if args.stop_after_shards is not None and args.stop_after_shards <= 0:
        raise ValueError("stop_after_shards must be positive")


def existing_shards(shard_dir: Path) -> List[Path]:
    if not shard_dir.exists():
        return []
    shards = sorted(shard_dir.glob("shard_*.npz"))
    for index, path in enumerate(shards):
        if path.name != f"shard_{index:05d}.npz":
            raise ValueError("Shard sequence is not contiguous")
    return shards


def restore_resume(
    shards: Sequence[Path],
    *,
    config_sha256: str,
    sl_sha256: str,
    rl_sha256: str,
    rng: CollectionRNG,
) -> int:
    games_completed = 0
    for index, path in enumerate(shards):
        with np.load(path, allow_pickle=False) as data:
            if int(data["shard_index"].item()) != index:
                raise ValueError(f"Shard index mismatch in {path}")
            if str(data["config_sha256"].item()) != config_sha256:
                raise ValueError(f"Config mismatch in {path}")
            if str(data["sl_checkpoint_sha256"].item()) != sl_sha256:
                raise ValueError(f"SL checkpoint mismatch in {path}")
            if str(data["rl_checkpoint_sha256"].item()) != rl_sha256:
                raise ValueError(f"RL checkpoint mismatch in {path}")
            games_completed = int(data["games_completed"].item())
    if shards:
        with np.load(shards[-1], allow_pickle=False) as data:
            rng.restore_json(str(data["collection_rng_state_json"].item()))
    return games_completed


def game_arrays(games) -> Dict[str, np.ndarray]:
    sample_games, states, targets, state_hashes = [], [], [], []
    for index, game in enumerate(games):
        if game.status == "sampled":
            if game.state is None or game.target is None:
                raise RuntimeError("Sampled game lacks state or target")
            sample_games.append(index)
            states.append(game.state)
            targets.append(game.target)
            state_hashes.append(_state_fingerprint(game.state))
        elif game.state is not None or game.target is not None:
            raise RuntimeError("Excluded game leaked a Value sample")
    return {
        "game_ids": np.asarray([g.game_id for g in games], dtype=np.str_),
        "game_fingerprints": np.asarray(
            [_game_fingerprint(g.root_fen, g.moves_uci) for g in games], dtype="<U64"
        ),
        "u": np.asarray([g.u for g in games], dtype=np.int16),
        "root_fens": np.asarray([g.root_fen for g in games], dtype=np.str_),
        "moves_uci": np.asarray([" ".join(g.moves_uci) for g in games], dtype=np.str_),
        "statuses": np.asarray([g.status for g in games], dtype=np.str_),
        "termination_reasons": np.asarray(
            [g.termination_reason for g in games], dtype=np.str_
        ),
        "truncated": np.asarray([g.truncated for g in games], dtype=np.bool_),
        "winner_white": np.asarray(
            [2 if g.winner_white is None else g.winner_white for g in games], dtype=np.int8
        ),
        "saved_current_player": np.asarray(
            [0 if g.saved_current_player is None else g.saved_current_player for g in games],
            dtype=np.int8,
        ),
        "random_moves_uci": np.asarray(
            ["" if g.random_move_uci is None else g.random_move_uci for g in games],
            dtype=np.str_,
        ),
        "sl_plies": np.asarray([g.sl_plies for g in games], dtype=np.int16),
        "rl_plies": np.asarray([g.rl_plies for g in games], dtype=np.int16),
        "sample_game_indices": np.asarray(sample_games, dtype=np.int32),
        "sample_states": (
            np.stack(states).astype(np.float32, copy=False)
            if states else np.empty((0, 14, 8, 8), dtype=np.float32)
        ),
        "sample_targets": np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        "sample_state_fingerprints": np.asarray(state_hashes, dtype="<U64"),
    }


def collect_shards(
    args: argparse.Namespace,
    *,
    config: Dict[str, object],
    config_sha256: str,
) -> bool:
    shard_dir = args.data_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = existing_shards(shard_dir)
    expected = args.total_games // args.shard_size
    if len(shards) > expected:
        raise ValueError("More shards exist than requested")
    policies = load_value_data_policies(
        args.sl_checkpoint, args.rl_checkpoint, device=args.device
    )
    if policies.provenance != config["sources"]:
        raise ValueError("Loaded checkpoint provenance differs from configuration")
    rng = CollectionRNG.from_seed(args.seed)
    completed = restore_resume(
        shards,
        config_sha256=config_sha256,
        sl_sha256=policies.provenance["sl"]["sha256"],
        rl_sha256=policies.provenance["rl"]["sha256"],
        rng=rng,
    )
    if completed != len(shards) * args.shard_size:
        raise ValueError("Completed-game count does not match shard count")
    collector_config = AlphaGoValueDataConfig(
        args.u_min, args.u_max, args.max_plies, policies.claim_draw
    )
    for shard_index in range(len(shards), expected):
        shard_started = time.perf_counter()
        games = [
            collect_value_game(
                policies.sl_policy,
                policies.rl_policy,
                config=collector_config,
                rng=rng,
                game_id=f"game_{completed + offset:09d}",
                device=args.device,
            )
            for offset in range(args.shard_size)
        ]
        completed += args.shard_size
        arrays = game_arrays(games)
        shard_collection_seconds = time.perf_counter() - shard_started
        arrays.update(
            {
                "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
                "config_sha256": np.asarray(config_sha256),
                "sl_checkpoint_sha256": np.asarray(policies.provenance["sl"]["sha256"]),
                "rl_checkpoint_sha256": np.asarray(policies.provenance["rl"]["sha256"]),
                "shard_index": np.asarray(shard_index, dtype=np.int32),
                "games_completed": np.asarray(completed, dtype=np.int32),
                "shard_collection_seconds": np.asarray(
                    shard_collection_seconds, dtype=np.float64
                ),
                "collection_rng_state_json": np.asarray(rng.state_json()),
            }
        )
        destination = shard_dir / f"shard_{shard_index:05d}.npz"
        _atomic_npz(destination, arrays)
        with np.load(destination, allow_pickle=False) as check:
            if any(check[key].dtype == object for key in check.files):
                raise RuntimeError("Shard contains object dtype")
        counts = Counter(g.status for g in games)
        shard_elapsed = time.perf_counter() - shard_started
        print(
            f"COLLECT shard={shard_index + 1}/{expected} games={completed}/{args.total_games} "
            f"sampled={counts['sampled']} excluded={args.shard_size-counts['sampled']} "
            f"collection_seconds={shard_collection_seconds:.3f} "
            f"elapsed_seconds={shard_elapsed:.3f}",
            flush=True,
        )
        if args.stop_after_shards and shard_index - len(shards) + 1 >= args.stop_after_shards:
            return False
    return True


def load_rows(shards: Sequence[Path]) -> Tuple[List[dict], List[dict]]:
    games, samples = [], []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            base = len(games)
            for index in range(len(data["game_ids"])):
                games.append(
                    {
                        "id": str(data["game_ids"][index]),
                        "fingerprint": str(data["game_fingerprints"][index]),
                        "status": str(data["statuses"][index]),
                        "reason": str(data["termination_reasons"][index]),
                        "u": int(data["u"][index]),
                    }
                )
            for sample_index, local_index in enumerate(data["sample_game_indices"]):
                game_index = base + int(local_index)
                samples.append(
                    {
                        "game_index": game_index,
                        "game_id": games[game_index]["id"],
                        "game_fingerprint": games[game_index]["fingerprint"],
                        "state": data["sample_states"][sample_index].copy(),
                        "target": float(data["sample_targets"][sample_index, 0]),
                        "state_fingerprint": str(data["sample_state_fingerprints"][sample_index]),
                    }
                )
    return games, samples


def sample_arrays(rows: Sequence[dict]) -> Dict[str, np.ndarray]:
    return {
        "states": np.stack([r["state"] for r in rows]).astype(np.float32)
        if rows else np.empty((0, 14, 8, 8), dtype=np.float32),
        "targets": np.asarray([r["target"] for r in rows], dtype=np.float32).reshape(-1, 1),
        "game_ids": np.asarray([r["game_id"] for r in rows], dtype=np.str_),
        "game_fingerprints": np.asarray([r["game_fingerprint"] for r in rows], dtype="<U64"),
        "state_fingerprints": np.asarray([r["state_fingerprint"] for r in rows], dtype="<U64"),
    }


def filter_cross_split_state_duplicates(rows: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Keep test authoritative, then val, and remove exact tensor overlap upstream."""
    val_hashes = {r["state_fingerprint"] for r in rows["val"]}
    test_hashes = {r["state_fingerprint"] for r in rows["test"]}
    return {
        "train": [
            r for r in rows["train"]
            if r["state_fingerprint"] not in val_hashes | test_hashes
        ],
        "val": [
            r for r in rows["val"] if r["state_fingerprint"] not in test_hashes
        ],
        "test": list(rows["test"]),
    }


def build_dataset(args: argparse.Namespace, config: Dict[str, object], config_sha256: str) -> None:
    dataset_path = args.data_dir / "dataset.npz"
    manifest_path = args.data_dir / "manifest.json"
    if dataset_path.exists() or manifest_path.exists():
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Dataset/manifest output is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_sha256:
            raise ValueError("Existing dataset manifest config mismatch")
        if manifest.get("dataset", {}).get("sha256") != _sha256_file(dataset_path):
            raise ValueError("Existing dataset hash mismatch")
        print("DATASET already complete", flush=True)
        return
    shards = existing_shards(args.data_dir / "shards")
    if len(shards) != args.total_games // args.shard_size:
        raise RuntimeError("Cannot split before all collection shards exist")
    games, samples = load_rows(shards)
    if len(games) != args.total_games:
        raise RuntimeError("Collected game count differs from configuration")
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, game in enumerate(games):
        groups[game["fingerprint"]].append(index)
    fingerprints = np.asarray(sorted(groups), dtype="<U64")
    split_rng = np.random.default_rng(args.seed)
    split_rng.shuffle(fingerprints)
    train_end, val_end = int(.8 * len(fingerprints)), int(.9 * len(fingerprints))
    assignment = {
        str(fp): "train" if i < train_end else "val" if i < val_end else "test"
        for i, fp in enumerate(fingerprints)
    }
    rows = {
        split: [r for r in samples if assignment[r["game_fingerprint"]] == split]
        for split in ("train", "val", "test")
    }
    filtered = filter_cross_split_state_duplicates(rows)
    arrays: Dict[str, np.ndarray] = {"config_sha256": np.asarray(config_sha256)}
    for split, split_rows in filtered.items():
        for key, value in sample_arrays(split_rows).items():
            arrays[f"{split}_{key}"] = value
    _atomic_npz(dataset_path, arrays)
    excluded = Counter(g["status"] for g in games if g["status"] != "sampled")
    reasons = Counter(g["reason"] for g in games if g["status"] != "sampled")
    shard_seconds = []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            shard_seconds.append(
                float(data["shard_collection_seconds"].item())
                if "shard_collection_seconds" in data.files else None
            )
    label_counts = {
        split: {
            "-1": int(np.sum(arrays[f"{split}_targets"] == -1)),
            "0": int(np.sum(arrays[f"{split}_targets"] == 0)),
            "+1": int(np.sum(arrays[f"{split}_targets"] == 1)),
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "config": config,
        "config_sha256": config_sha256,
        "collection": {
            "games": len(games),
            "samples": len(samples),
            "excluded_by_status": dict(excluded),
            "excluded_by_reason": dict(reasons),
            "per_game_u_and_game_id_stored_in_shards": True,
            "timing": {
                "shard_collection_seconds": shard_seconds,
                "total_collection_seconds": (
                    float(sum(shard_seconds))
                    if all(value is not None for value in shard_seconds) else None
                ),
            },
        },
        "split": {
            "games": Counter(assignment[g["fingerprint"]] for g in games),
            "samples_before_filter": {k: len(v) for k, v in rows.items()},
            "samples_after_filter": {k: len(v) for k, v in filtered.items()},
            "removed_exact_tensor_overlap": {
                "train": len(rows["train"]) - len(filtered["train"]),
                "val": len(rows["val"]) - len(filtered["val"]),
            },
            "test_status": "held_out_not_evaluated",
            "label_counts": label_counts,
        },
        "dataset": {"path": str(dataset_path), "sha256": _sha256_file(dataset_path)},
    }
    manifest["split"]["games"] = dict(manifest["split"]["games"])
    _atomic_json(manifest_path, manifest)
    print(
        "SUMMARY "
        f"games={len(games)} samples={len(samples)} "
        f"labels={json.dumps(label_counts, sort_keys=True)} "
        f"excluded={json.dumps(dict(excluded), sort_keys=True)} "
        f"reasons={json.dumps(dict(reasons), sort_keys=True)} "
        f"total_collection_seconds={manifest['collection']['timing']['total_collection_seconds']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sl-checkpoint", required=True, type=Path)
    parser.add_argument("--rl-checkpoint", required=True, type=Path)
    parser.add_argument("--u-min", required=True, type=int)
    parser.add_argument("--u-max", required=True, type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage5/value_alphago2016"))
    parser.add_argument("--total-games", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2062)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-after-shards", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    policies = load_value_data_policies(args.sl_checkpoint, args.rl_checkpoint, device="cpu")
    config = experiment_config(args, policies.provenance)
    config_sha256 = _sha256_bytes(_canonical_json(config).encode("utf-8"))
    if collect_shards(args, config=config, config_sha256=config_sha256):
        build_dataset(args, config, config_sha256)


if __name__ == "__main__":
    main()
