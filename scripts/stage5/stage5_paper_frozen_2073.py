"""Fresh-init B16 Policy with the AlphaGo-style frozen Value, then validate."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import training.stage5.trainer as stage5_trainer
from scripts.stage5.stage5_validation_helpers import evaluate_branch, sha256, summarize_training
from training.stage4_train_sl import load_checkpoint as load_stage4_checkpoint
from training.stage5.checkpoint import (
    load_stage5_checkpoint,
    load_value_only_checkpoint,
)
from training.stage5.types import Stage5TrainConfig


STAGE4 = Path("checkpoints/stage4/best_lichess.pt")
POLICY_INIT = Path("checkpoints/stage5/batch_comparison/b16/best.pt")
VALUE = Path("checkpoints/stage5/stage5_alphago_value_2071_training_kaggle/best.pt")
OPENINGS_SOURCE = Path("results/stage5/batch_comparison/comparison.json")
SAVED_B16_VALIDATION = Path("results/stage5/batch_comparison/comparison.json")
OUTPUT_DIR = Path("checkpoints/stage5/paper_frozen_2073")
RESUME_CHECKPOINT = OUTPUT_DIR / "last.pt"
RESULT_DIR = Path("results/stage5/paper_frozen_2073")
RESULT_PATH = RESULT_DIR / "comparison.json"
EXIT_CODE_PATH = RESULT_DIR / "resume_exit_code.txt"
EXPECTED_POLICY_SHA256 = "1b3480e25cbf4b55cafeae85f0915a409993b7b2d08b068e35bb9cd056b244e0"
EXPECTED_VALUE_SHA256 = "93dd82049962b4e8f9c94927879a766ba10fc914f9474dda6a794b3f88ca07ca"


def load_openings() -> tuple[list[dict], str]:
    payload = json.loads(OPENINGS_SOURCE.read_text(encoding="utf-8"))
    openings = payload.get("openings")
    if not isinstance(openings, list) or len(openings) != 50:
        raise ValueError("Validation source must contain exactly 50 openings")
    for index, opening in enumerate(openings, 1):
        moves = opening.get("moves_uci")
        if not isinstance(moves, list) or len(moves) != 8:
            raise ValueError(f"Opening {index} must contain exactly 8 plies")
    return openings, sha256(OPENINGS_SOURCE)


def ensure_preconditions() -> tuple[list[dict], dict[str, str]]:
    for path in (STAGE4, POLICY_INIT, VALUE, OPENINGS_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not RESUME_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CHECKPOINT}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    hashes = {
        "stage4": sha256(STAGE4),
        "policy_init": sha256(POLICY_INIT),
        "value": sha256(VALUE),
    }
    if hashes["policy_init"] != EXPECTED_POLICY_SHA256:
        raise ValueError("Policy init SHA256 mismatch")
    if hashes["value"] != EXPECTED_VALUE_SHA256:
        raise ValueError("Value SHA256 mismatch")
    openings, openings_hash = load_openings()
    hashes["openings_source"] = openings_hash
    return openings, hashes


def training_config() -> Stage5TrainConfig:
    return Stage5TrainConfig(
        resume_checkpoint=RESUME_CHECKPOINT,
        output_dir=OUTPUT_DIR,
        device="cuda",
        seed=2073,
        iterations=20,
        games_per_iteration=32,
        max_plies=300,
        chunk_size=8,
        policy_lr=1e-5,
        value_lr=1e-3,
        baseline="value",
        value_mode="frozen",
        opponent_mode="pool",
        opponent_snapshot_interval=5,
        opponent_pool_size=4,
        start_mode="standard",
    )


def strict_resume_preflight() -> dict:
    loaded = load_stage5_checkpoint(RESUME_CHECKPOINT, device="cpu")
    payload = loaded.payload
    config = payload["training_config"]
    expected_config = {
        "seed": 2073,
        "iterations": 20,
        "games_per_iteration": 32,
        "chunk_size": 8,
        "policy_lr": 1e-5,
        "max_plies": 300,
        "baseline": "value",
        "value_mode": "frozen",
        "opponent_mode": "pool",
        "opponent_pool_size": 4,
        "opponent_snapshot_interval": 5,
        "start_mode": "standard",
    }
    actual_config = {key: config.get(key) for key in expected_config}
    if actual_config != expected_config:
        raise ValueError(f"Resume config mismatch: {actual_config}")
    if payload["completed_iteration"] != 17 or len(payload["history"]) != 17:
        raise ValueError("Resume checkpoint must contain exactly 17 iterations")
    if payload["history"][-1].get("iteration") != 17:
        raise ValueError("Resume history does not end at iteration 17")
    source = payload.get("source_checkpoint", {})
    if source.get("policy_init", {}).get("sha256") != EXPECTED_POLICY_SHA256:
        raise ValueError("Resume Policy provenance mismatch")
    if payload.get("value_source", {}).get("sha256") != EXPECTED_VALUE_SHA256:
        raise ValueError("Resume Value provenance mismatch")
    if any(parameter.requires_grad for parameter in loaded.value_model.parameters()):
        raise ValueError("Embedded frozen Value unexpectedly requires gradients")
    if loaded.value_optimizer.state_dict()["state"]:
        raise ValueError("Embedded frozen Value optimizer unexpectedly has state")
    pool_ids = [snapshot.opponent_id for snapshot in loaded.opponent_pool]
    expected_pool = [
        "stage4",
        "learner_iter_000005",
        "learner_iter_000010",
        "learner_iter_000015",
    ]
    if pool_ids != expected_pool:
        raise ValueError(f"Resume opponent pool mismatch: {pool_ids}")
    del loaded
    gc.collect()
    return {
        "strict_load": True,
        "checkpoint_sha256": sha256(RESUME_CHECKPOINT),
        "completed_iteration": 17,
        "history_length": 17,
        "opponent_pool_ids": pool_ids,
        "embedded_frozen_value": True,
    }


def score_from_record(record: dict) -> float:
    if record.get("result") == "1/2-1/2":
        return 0.5
    stage5_white = record.get("stage5_color") == "white"
    won = (stage5_white and record.get("result") == "1-0") or (
        not stage5_white and record.get("result") == "0-1"
    )
    return 1.0 if won else 0.0


def pair_scores_from_records(records: list[dict]) -> list[float]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["opening_index"]), []).append(record)
    if sorted(grouped) != list(range(1, 51)):
        raise ValueError("Validation records do not cover openings 1..50")
    scores = []
    for index in range(1, 51):
        pair = grouped[index]
        if len(pair) != 2 or {row.get("stage5_color") for row in pair} != {
            "white", "black"
        }:
            raise ValueError(f"Opening {index} does not contain a color pair")
        scores.append(sum(score_from_record(row) for row in pair))
    return scores


def reusable_b16_validation(openings: list[dict]) -> dict | None:
    if not SAVED_B16_VALIDATION.is_file():
        return None
    payload = json.loads(SAVED_B16_VALIDATION.read_text(encoding="utf-8"))
    expected_validation = {
        "openings": 50,
        "games_per_milestone": 100,
        "colors_per_opening": 2,
        "policy": "argmax",
        "device": "cpu",
        "claim_draw": True,
        "max_plies_after_opening": 300,
    }
    if payload.get("validation") != expected_validation:
        return None
    if payload.get("openings") != openings:
        return None
    branch = payload.get("branches", {}).get("b16", {})
    best = branch.get("best", {})
    if best.get("sha256") != EXPECTED_POLICY_SHA256:
        return None
    selected = None
    for milestone in branch.get("milestones", []):
        if (
            milestone.get("completed_iteration") == best.get("completed_iteration")
            and milestone.get("last_sha256") == EXPECTED_POLICY_SHA256
        ):
            selected = milestone.get("validation")
            break
    if not isinstance(selected, dict) or len(selected.get("records", [])) != 100:
        return None
    result = copy.deepcopy(selected)
    result["pair_scores"] = pair_scores_from_records(result["records"])
    result["reused"] = True
    result["reused_from"] = str(SAVED_B16_VALIDATION).replace("\\", "/")
    result["checkpoint_sha256"] = EXPECTED_POLICY_SHA256
    return result


def strict_integrity(last_path: Path, hashes_before: dict[str, str]) -> dict:
    loaded = load_stage5_checkpoint(last_path, device="cpu")
    payload = loaded.payload
    config = payload["training_config"]
    expected = {
        "completed_iteration": 20,
        "baseline": "value",
        "value_mode": "frozen",
        "opponent_mode": "pool",
        "start_mode": "standard",
        "opponent_snapshot_interval": 5,
        "opponent_pool_size": 4,
    }
    actual = {
        "completed_iteration": payload["completed_iteration"],
        **{key: config.get(key) for key in expected if key != "completed_iteration"},
    }
    if actual != expected:
        raise ValueError(f"Strict last.pt metadata mismatch: {actual}")
    source = payload.get("source_checkpoint", {})
    if source.get("policy_init", {}).get("sha256") != EXPECTED_POLICY_SHA256:
        raise ValueError("Policy init provenance mismatch")
    if payload.get("value_source", {}).get("sha256") != EXPECTED_VALUE_SHA256:
        raise ValueError("Frozen Value provenance mismatch")
    if loaded.value_optimizer.state_dict()["state"]:
        raise ValueError("Frozen Value optimizer unexpectedly contains state")
    pool_ids = [snapshot.opponent_id for snapshot in loaded.opponent_pool]
    if pool_ids != ["stage4", "learner_iter_000010", "learner_iter_000015", "learner_iter_000020"]:
        raise ValueError(f"Unexpected final opponent pool: {pool_ids}")
    source_value, _ = load_value_only_checkpoint(VALUE, device="cpu", claim_draw=True)
    for key, tensor in source_value.state_dict().items():
        if not torch.equal(tensor, loaded.value_model.state_dict()[key]):
            raise ValueError(f"Frozen Value changed: {key}")
    source_stage4, _ = load_stage4_checkpoint(STAGE4, device="cpu")
    for key, tensor in source_stage4.state_dict().items():
        if not torch.equal(tensor, payload["opponent_state_dict"][key]):
            raise ValueError(f"Stage 4 opponent changed: {key}")
    hashes_after = {
        "stage4": sha256(STAGE4),
        "policy_init": sha256(POLICY_INIT),
        "value": sha256(VALUE),
    }
    for key, before in hashes_before.items():
        if key in hashes_after and hashes_after[key] != before:
            raise RuntimeError(f"Source checkpoint changed: {key}")
    return {
        "strict_load": True,
        "checkpoint_sha256": sha256(last_path),
        "completed_iteration": payload["completed_iteration"],
        "history_length": len(payload["history"]),
        "value_frozen_and_matches_source": True,
        "stage4_opponent_matches_source": True,
        "source_checkpoints_unchanged": True,
        "opponent_pool_ids": pool_ids,
        "policy_init_provenance": copy.deepcopy(source["policy_init"]),
        "value_source": copy.deepcopy(payload["value_source"]),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    openings, hashes_before = ensure_preconditions()
    resume_preflight = strict_resume_preflight()
    print(
        f"RESUME strict_load=True completed_iteration=17 "
        f"checkpoint_sha256={resume_preflight['checkpoint_sha256']}",
        flush=True,
    )
    active = "candidate"
    original_save = stage5_trainer.save_stage5_checkpoint

    def save_with_progress(*args, **kwargs):
        original_save(*args, **kwargs)
        row = kwargs["history"][-1]
        print(
            f"TRAIN {active} iter={row['iteration']:02d}/20 "
            f"W/D/L/T={row['wins']}/{row['draws']}/{row['losses']}/{row['truncated']} "
            f"loss={row['policy_loss']} pool={row['opponent_pool_ids']}",
            flush=True,
        )

    started = time.perf_counter()
    stage5_trainer.save_stage5_checkpoint = save_with_progress
    try:
        result = stage5_trainer.train(training_config())
    finally:
        stage5_trainer.save_stage5_checkpoint = original_save
    training = summarize_training(result.history)
    training["elapsed_seconds"] = time.perf_counter() - started
    del result
    torch.cuda.empty_cache()
    gc.collect()

    last_path = OUTPUT_DIR / "last.pt"
    strict = strict_integrity(last_path, hashes_before)
    candidate = evaluate_branch("candidate", last_path, openings)
    source = reusable_b16_validation(openings)
    if source is None:
        source = evaluate_branch("source_b16", POLICY_INIT, openings)
        source["reused"] = False
        source["checkpoint_sha256"] = EXPECTED_POLICY_SHA256
    deltas = [
        candidate_score - source_score
        for source_score, candidate_score in zip(
            source["pair_scores"], candidate["pair_scores"]
        )
    ]
    paired = {
        "candidate_minus_b16_score": candidate["score"] - source["score"],
        "candidate_better_pairs": sum(delta > 0 for delta in deltas),
        "equal_pairs": sum(delta == 0 for delta in deltas),
        "candidate_worse_pairs": sum(delta < 0 for delta in deltas),
        "per_opening_delta": deltas,
    }
    payload = {
        "schema_version": 1,
        "purpose": "stage5_fresh_b16_frozen_alphago_value_validation",
        "selection_status": "validation_candidate_not_best",
        "config": {
            "stage4_opponent": str(STAGE4).replace("\\", "/"),
            "policy_init": str(POLICY_INIT).replace("\\", "/"),
            "value_checkpoint": str(VALUE).replace("\\", "/"),
            "output_checkpoint": str(last_path).replace("\\", "/"),
            "seed": 2073,
            "iterations": 20,
            "games_per_iteration": 32,
            "games_total": 640,
            "chunk_size": 8,
            "policy_lr": 1e-5,
            "value_mode": "frozen",
            "baseline": "value",
            "start_mode": "standard",
            "max_plies": 300,
            "claim_draw": True,
            "sampling_temperature": 1,
            "opponent_mode": "pool",
            "opponent_pool_size": 4,
            "opponent_snapshot_interval": 5,
            "validation_openings_source": str(OPENINGS_SOURCE).replace("\\", "/"),
            "validation_openings": 50,
            "games_per_validation": 100,
            "validation_device": "cpu",
            "validation_move_selection": "argmax",
            "max_plies_after_opening": 300,
        },
        "sha256_before": hashes_before,
        "sha256_after": {
            "stage4": sha256(STAGE4),
            "policy_init": sha256(POLICY_INIT),
            "value": sha256(VALUE),
            "candidate_last": strict["checkpoint_sha256"],
        },
        "strict_load": strict,
        "resume_preflight": resume_preflight,
        "training": training,
        "validation": {"candidate": candidate, "b16_source": source},
        "paired_comparison": paired,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(RESULT_PATH, payload)
    print(
        f"COMPLETE candidate_W/D/L={candidate['wins']}/{candidate['draws']}/{candidate['losses']} "
        f"candidate_score={candidate['score']:.6f} b16_score={source['score']:.6f} "
        f"delta={paired['candidate_minus_b16_score']:.6f} "
        f"pairs={paired['candidate_better_pairs']}/{paired['equal_pairs']}/"
        f"{paired['candidate_worse_pairs']} b16_reused={source['reused']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        EXIT_CODE_PATH.write_text("1\n", encoding="utf-8")
        raise
    else:
        EXIT_CODE_PATH.write_text("0\n", encoding="utf-8")
