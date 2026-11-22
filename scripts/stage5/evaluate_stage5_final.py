"""Resumable fixed 800-game validation of the two retained Stage 5 Policies."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

import chess

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.random_agent import RandomAgent
from agents.stage1_heuristics.think3_agent import Think3Agent
from agents.stage2_tree_search.alphabeta_agent import AlphaBetaAgent
from agents.stage3_mcts_pure.mcts_pure_agent import PureMCTSAgent
from agents.stage5_rl.rl_agent import RLPolicyAgent
from arena.evaluator import play_game
from games.chess.board import ChessBoard


SCHEMA_VERSION = 1
BASE_SEED = 207_400_000
MAX_PLIES_AFTER_OPENING = 300
OPENINGS_PATH = Path("results/stage5/batch_comparison/comparison.json")
OUTPUT_DIR = Path("results/stage5/final_eval")
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
EXIT_CODE_PATH = OUTPUT_DIR / "exit_code.txt"
POLICIES = {
    "b16": {
        "path": Path("checkpoints/stage5/batch_comparison/b16/best.pt"),
        "sha256": "1b3480e25cbf4b55cafeae85f0915a409993b7b2d08b068e35bb9cd056b244e0",
    },
    "paper": {
        "path": Path("checkpoints/stage5/paper_frozen_2073/last.pt"),
        "sha256": "4d70697035064bfc4be2e505959bb070d4a3afd0c697390b78908605016e1832",
    },
}
OPPONENTS = ("random", "think3", "alphabeta_d3_q6", "mcts_s200_r30")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_openings() -> tuple[list[dict], dict]:
    if not OPENINGS_PATH.is_file():
        raise FileNotFoundError(OPENINGS_PATH)
    payload = json.loads(OPENINGS_PATH.read_text(encoding="utf-8"))
    openings = payload.get("openings")
    if not isinstance(openings, list) or len(openings) != 50:
        raise ValueError("Expected exactly 50 saved validation openings")
    for index, opening in enumerate(openings, 1):
        moves = opening.get("moves_uci")
        if not isinstance(moves, list) or len(moves) != 8:
            raise ValueError(f"Opening {index} must contain exactly 8 plies")
        board = reconstruct_opening(opening)
        if board.is_terminal():
            raise ValueError(f"Opening {index} is terminal")
    return openings, {
        "path": str(OPENINGS_PATH).replace("\\", "/"),
        "file_sha256": sha256_file(OPENINGS_PATH),
        "ordered_openings_sha256": canonical_sha256(openings),
        "count": 50,
        "plies_per_opening": 8,
    }


def reconstruct_opening(opening: dict) -> ChessBoard:
    board = ChessBoard(claim_draw=True)
    for uci in opening["moves_uci"]:
        move = chess.Move.from_uci(uci)
        if not board.board.is_legal(move):
            raise ValueError(f"Illegal saved opening move: {uci}")
        board.step(move)
    if board.fen() != opening["fen_after_8_plies"]:
        raise ValueError("Saved opening FEN differs from reconstructed history")
    return board


def opponent_seed(opponent_index: int, opening_index: int, policy_white: bool) -> int:
    color_index = 0 if policy_white else 1
    return BASE_SEED + opponent_index * 1_000 + opening_index * 2 + color_index


def game_id(policy: str, opponent: str, opening_index: int, policy_white: bool) -> str:
    color = "white" if policy_white else "black"
    return f"{policy}__vs__{opponent}__opening_{opening_index:03d}__policy_{color}"


def build_schedule(opening_limit: int = 50) -> list[dict]:
    if not isinstance(opening_limit, int) or not 1 <= opening_limit <= 50:
        raise ValueError("opening_limit must be an integer in 1..50")
    schedule = []
    for policy in POLICIES:
        for opponent_index, opponent in enumerate(OPPONENTS):
            for opening_index in range(1, opening_limit + 1):
                for policy_white in (True, False):
                    schedule.append(
                        {
                            "game_id": game_id(
                                policy, opponent, opening_index, policy_white
                            ),
                            "policy": policy,
                            "opponent": opponent,
                            "opening_index": opening_index,
                            "policy_color": "white" if policy_white else "black",
                            "opponent_seed": opponent_seed(
                                opponent_index, opening_index, policy_white
                            ),
                        }
                    )
    ids = [row["game_id"] for row in schedule]
    expected = 2 * len(OPPONENTS) * opening_limit * 2
    if len(schedule) != expected or len(set(ids)) != expected:
        raise RuntimeError(
            f"Final evaluation schedule must have {expected} unique game IDs"
        )
    return schedule


def validate_sources() -> tuple[list[dict], dict, dict]:
    openings, opening_source = load_openings()
    policy_sources = {}
    for name, expected in POLICIES.items():
        path = expected["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(f"{name} checkpoint SHA256 mismatch: {actual}")
        policy_sources[name] = {
            "path": str(path).replace("\\", "/"), "sha256": actual
        }
    config = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "stage5_final_fixed_opening_validation",
        "policies": policy_sources,
        "opponents": {
            "random": {"type": "RandomAgent"},
            "think3": {"type": "Think3Agent", "lookahead": "one_ply"},
            "alphabeta_d3_q6": {
                "type": "AlphaBetaAgent", "depth": 3,
                "use_quiescence": True, "q_depth": 6,
            },
            "mcts_s200_r30": {
                "type": "PureMCTSAgent", "simulations": 200,
                "c": 1.414, "max_rollout_depth": 30,
            },
        },
        "openings": opening_source,
        "games_per_matchup": 100,
        "total_games": 800,
        "colors_per_opening": 2,
        "claim_draw": True,
        "max_plies_after_opening": MAX_PLIES_AFTER_OPENING,
        "policy_device": "cpu",
        "policy_move_selection": "legal_masked_argmax",
        "seed_formula": (
            "207400000 + opponent_index*1000 + one_based_opening_index*2 "
            "+ color_index; color_index=0 policy_white, 1 policy_black; "
            "policy name intentionally excluded"
        ),
        "paired_games_are_not_claimed_independent": True,
        "elo_not_computed": True,
    }
    config["config_sha256"] = canonical_sha256(config)
    return openings, policy_sources, config


def make_opponent(name: str, seed: int):
    if name == "random":
        return RandomAgent(name="Random", seed=seed)
    if name == "think3":
        return Think3Agent(name="Think3", seed=seed)
    if name == "alphabeta_d3_q6":
        return AlphaBetaAgent(
            name="AlphaBeta_d3_q6", depth=3, use_quiescence=True, q_depth=6
        )
    if name == "mcts_s200_r30":
        return PureMCTSAgent(
            simulations=200, c=1.414, max_rollout_depth=30,
            seed=seed, name="PureMCTS_s200_r30",
        )
    raise ValueError(f"Unknown opponent: {name}")


def empty_matchup(policy: str, opponent: str, config: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "policy": policy,
        "opponent": opponent,
        "policy_checkpoint": config["policies"][policy],
        "opening_source": config["openings"],
        "evaluation_config": config,
        "records": [],
    }


def load_matchup(path: Path, policy: str, opponent: str, config: dict) -> dict:
    if not path.exists():
        return empty_matchup(policy, opponent, config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("config_sha256") == config["config_sha256"]
        and payload.get("policy") == policy
        and payload.get("opponent") == opponent
        and payload.get("policy_checkpoint") == config["policies"][policy]
        and payload.get("opening_source") == config["openings"]
    )
    if not expected:
        raise ValueError(f"Resume metadata mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Resume records are invalid: {path}")
    ids = [row.get("game_id") for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate resume game ID: {path}")
    allowed = {
        game_id(policy, opponent, opening_index, policy_white)
        for opening_index in range(1, 51)
        for policy_white in (True, False)
    }
    if not set(ids).issubset(allowed):
        raise ValueError(f"Unexpected resume game ID: {path}")
    return payload


def play_scheduled_game(
    policy_agent: RLPolicyAgent, opponent_name: str, opening: dict,
    opening_index: int, policy_white: bool, seed: int, identifier: str,
) -> dict:
    opponent = make_opponent(opponent_name, seed)
    board = reconstruct_opening(opening)
    white, black = (
        (policy_agent, opponent) if policy_white else (opponent, policy_agent)
    )
    started = time.perf_counter()
    result = play_game(
        white, black, max_plies=MAX_PLIES_AFTER_OPENING, board=board
    )
    elapsed = time.perf_counter() - started
    record = result.to_dict()
    if result.result == "1/2-1/2":
        policy_score = 0.5
    else:
        policy_won = (
            policy_white and result.result == "1-0"
        ) or (not policy_white and result.result == "0-1")
        policy_score = 1.0 if policy_won else 0.0
    policy_seconds = result.white_seconds if policy_white else result.black_seconds
    policy_calls = result.white_calls if policy_white else result.black_calls
    opponent_seconds = result.black_seconds if policy_white else result.white_seconds
    opponent_calls = result.black_calls if policy_white else result.white_calls
    record.update(
        {
            "game_id": identifier,
            "opening_index": opening_index,
            "policy_color": "white" if policy_white else "black",
            "opponent_seed": seed,
            "policy_score": policy_score,
            "policy_seconds": policy_seconds,
            "policy_calls": policy_calls,
            "opponent_seconds": opponent_seconds,
            "opponent_calls": opponent_calls,
            "elapsed_seconds": elapsed,
            "ply_cap_adjudication": result.reason == "max_plies_adjudication",
            "truncated": result.reason == "max_plies_adjudication",
            "rules_draw": (
                result.result == "1/2-1/2"
                and result.reason != "max_plies_adjudication"
            ),
        }
    )
    return record


def summarize_records(records: list[dict]) -> dict:
    wins = sum(row["policy_score"] == 1.0 for row in records)
    draws = sum(row["policy_score"] == 0.5 for row in records)
    losses = sum(row["policy_score"] == 0.0 for row in records)
    policy_seconds = sum(float(row["policy_seconds"]) for row in records)
    policy_calls = sum(int(row["policy_calls"]) for row in records)
    opponent_seconds = sum(float(row["opponent_seconds"]) for row in records)
    opponent_calls = sum(int(row["opponent_calls"]) for row in records)
    by_color = {}
    for color in ("white", "black"):
        subset = [row for row in records if row["policy_color"] == color]
        by_color[color] = {
            "games": len(subset),
            "wins": sum(row["policy_score"] == 1.0 for row in subset),
            "draws": sum(row["policy_score"] == 0.5 for row in subset),
            "losses": sum(row["policy_score"] == 0.0 for row in subset),
            "score": (
                sum(float(row["policy_score"]) for row in subset) / len(subset)
                if subset else None
            ),
        }
    return {
        "games": len(records), "wins": wins, "draws": draws, "losses": losses,
        "score": (wins + 0.5 * draws) / len(records) if records else None,
        "by_policy_color": by_color,
        "policy_ms_per_move": 1000 * policy_seconds / policy_calls if policy_calls else 0.0,
        "opponent_ms_per_move": 1000 * opponent_seconds / opponent_calls if opponent_calls else 0.0,
        "termination_reasons": dict(Counter(row["reason"] for row in records)),
        "ply_cap_adjudications": sum(row["ply_cap_adjudication"] for row in records),
        "truncated": sum(row["truncated"] for row in records),
        "rules_draws": sum(row["rules_draw"] for row in records),
        "errors_or_forfeits": sum(
            bool(row.get("error")) or "forfeit" in row.get("reason", "")
            for row in records
        ),
        "unique_game_ids": len({row["game_id"] for row in records}),
        "paired_games_are_not_claimed_independent": True,
    }


def reduced_output_dir(opening_limit: int) -> Path:
    return OUTPUT_DIR / f"reduced_{opening_limit}"


def build_reduced_config(base_config: dict, openings: list[dict], opening_limit: int) -> dict:
    target_games = 2 * len(OPPONENTS) * opening_limit * 2
    config = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "stage5_final_reduced_fixed_opening_validation",
        "reduction_reason": "reduced evaluation budget because of wall-clock cost",
        "parent_800_config_sha256": base_config["config_sha256"],
        "parent_records_preserved": True,
        "opening_selection": "first_n_in_saved_order; not selected by result",
        "opening_limit": opening_limit,
        "selected_openings_sha256": canonical_sha256(openings[:opening_limit]),
        "policies": base_config["policies"],
        "opponents": base_config["opponents"],
        "opening_source": base_config["openings"],
        "games_per_matchup": opening_limit * 2,
        "total_games": target_games,
        "colors_per_opening": 2,
        "claim_draw": base_config["claim_draw"],
        "max_plies_after_opening": base_config["max_plies_after_opening"],
        "policy_device": base_config["policy_device"],
        "policy_move_selection": base_config["policy_move_selection"],
        "seed_formula": base_config["seed_formula"],
        "opponent_strength_unchanged": True,
        "records_outside_target_retained_as_supplemental": True,
        "paired_games_are_not_claimed_independent": True,
        "elo_not_computed": True,
    }
    config["config_sha256"] = canonical_sha256(config)
    return config


def load_original_records(base_config: dict) -> tuple[dict[str, dict], dict[str, str]]:
    records: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for policy in POLICIES:
        for opponent in OPPONENTS:
            path = OUTPUT_DIR / f"{policy}_vs_{opponent}.json"
            if not path.exists():
                continue
            payload = load_matchup(path, policy, opponent, base_config)
            for record in payload["records"]:
                identifier = record["game_id"]
                if identifier in records:
                    raise ValueError(f"Duplicate original game ID: {identifier}")
                records[identifier] = record
                sources[identifier] = str(path).replace("\\", "/")
    return records, sources


def empty_reduced_matchup(policy: str, opponent: str, config: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "policy": policy,
        "opponent": opponent,
        "policy_checkpoint": config["policies"][policy],
        "reduced_config": config,
        "records": [],
        "record_sources": {},
    }


def load_reduced_matchup(path: Path, policy: str, opponent: str, config: dict) -> dict:
    if not path.exists():
        return empty_reduced_matchup(policy, opponent, config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("config_sha256") == config["config_sha256"]
        and payload.get("policy") == policy
        and payload.get("opponent") == opponent
        and payload.get("policy_checkpoint") == config["policies"][policy]
        and payload.get("reduced_config") == config
    ):
        raise ValueError(f"Reduced resume metadata mismatch: {path}")
    records = payload.get("records")
    sources = payload.get("record_sources")
    if not isinstance(records, list) or not isinstance(sources, dict):
        raise ValueError(f"Reduced records metadata is invalid: {path}")
    ids = [record.get("game_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate reduced game ID: {path}")
    return payload


def run_reduced(opening_limit: int, stop_file: Path | None) -> bool:
    openings, _, base_config = validate_sources()
    target_schedule = build_schedule(opening_limit)
    target_ids = {row["game_id"] for row in target_schedule}
    config = build_reduced_config(base_config, openings, opening_limit)
    output_dir = reduced_output_dir(opening_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != config:
            raise ValueError("Reduced config differs from existing config.json")
    else:
        atomic_json(config_path, config)
    original_records, original_sources = load_original_records(base_config)
    original_target_ids = set(original_records).intersection(target_ids)
    original_extra_ids = set(original_records).difference(target_ids)
    newly_played = 0
    run_started = time.perf_counter()
    matchup_summaries = {}
    all_ids = set()
    policy_agents: dict[str, RLPolicyAgent] = {}
    for policy_name in POLICIES:
        for opponent_name in OPPONENTS:
            path = output_dir / f"{policy_name}_vs_{opponent_name}.json"
            payload = load_reduced_matchup(path, policy_name, opponent_name, config)
            records_by_id = {row["game_id"]: row for row in payload["records"]}
            expected_rows = [
                row for row in target_schedule
                if row["policy"] == policy_name and row["opponent"] == opponent_name
            ]
            for scheduled in expected_rows:
                identifier = scheduled["game_id"]
                if identifier in records_by_id:
                    if identifier in original_records and records_by_id[identifier] != original_records[identifier]:
                        raise ValueError(f"Reduced/original record mismatch: {identifier}")
                    continue
                if identifier in original_records:
                    record = original_records[identifier]
                    payload["records"].append(record)
                    records_by_id[identifier] = record
                    payload["record_sources"][identifier] = {
                        "kind": "reused_parent_800_record",
                        "path": original_sources[identifier],
                    }
                    continue
                if stop_file is not None and stop_file.exists():
                    payload["summary"] = summarize_records(payload["records"])
                    atomic_json(path, payload)
                    print(
                        f"STOP_REQUESTED file={stop_file} new_games={newly_played}",
                        flush=True,
                    )
                    return False
                if policy_name not in policy_agents:
                    policy_agents[policy_name] = RLPolicyAgent(
                        config["policies"][policy_name]["path"], device="cpu",
                        seed=None, name=f"Stage5_{policy_name}",
                    )
                opening_index = int(scheduled["opening_index"])
                policy_white = scheduled["policy_color"] == "white"
                record = play_scheduled_game(
                    policy_agents[policy_name], opponent_name,
                    openings[opening_index - 1], opening_index, policy_white,
                    int(scheduled["opponent_seed"]), identifier,
                )
                payload["records"].append(record)
                records_by_id[identifier] = record
                payload["record_sources"][identifier] = {
                    "kind": "played_for_reduced_schedule"
                }
                payload["summary"] = summarize_records(payload["records"])
                atomic_json(path, payload)
                newly_played += 1
                if newly_played % 10 == 0:
                    print(
                        f"REDUCED_PROGRESS new_games={newly_played} "
                        f"target_saved={len(original_target_ids)+newly_played}/"
                        f"{config['total_games']} elapsed_seconds="
                        f"{time.perf_counter()-run_started:.3f}",
                        flush=True,
                    )
            expected_count = opening_limit * 2
            if len(payload["records"]) != expected_count:
                raise RuntimeError(f"Reduced matchup must have {expected_count} records: {path}")
            ids = [row["game_id"] for row in payload["records"]]
            if set(ids) != {row["game_id"] for row in expected_rows}:
                raise RuntimeError(f"Reduced matchup game IDs mismatch: {path}")
            payload["summary"] = summarize_records(payload["records"])
            atomic_json(path, payload)
            matchup_summaries[f"{policy_name}_vs_{opponent_name}"] = payload["summary"]
            all_ids.update(ids)
    if all_ids != target_ids:
        raise RuntimeError("Reduced final game ID set differs from target schedule")
    if any(summary["games"] != opening_limit * 2 for summary in matchup_summaries.values()):
        raise RuntimeError("Main comparison mixes unequal matchup game counts")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "completed": True,
        "games": len(target_ids),
        "unique_game_ids": len(all_ids),
        "games_per_matchup": opening_limit * 2,
        "reused_parent_records": len(original_target_ids),
        "newly_played_records": len(target_ids) - len(original_target_ids),
        "supplemental_parent_records_outside_target": len(original_extra_ids),
        "matchups": matchup_summaries,
        "run_elapsed_seconds": time.perf_counter() - run_started,
        "reduction_reason": config["reduction_reason"],
        "elo_not_computed": True,
        "paired_games_are_not_claimed_independent": True,
    }
    atomic_json(output_dir / "summary.json", summary)
    print(
        f"REDUCED_COMPLETE games={len(target_ids)} reused="
        f"{len(original_target_ids)} newly_played={len(target_ids)-len(original_target_ids)} "
        f"supplemental={len(original_extra_ids)}",
        flush=True,
    )
    return True


def run_evaluation() -> None:
    openings, _, config = validate_sources()
    schedule = build_schedule()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    newly_played = 0
    run_started = time.perf_counter()
    matchup_summaries = {}
    all_ids = set()
    for policy_name, policy_source in config["policies"].items():
        policy_agent = RLPolicyAgent(
            policy_source["path"], device="cpu", seed=None,
            name=f"Stage5_{policy_name}",
        )
        for opponent_name in OPPONENTS:
            path = OUTPUT_DIR / f"{policy_name}_vs_{opponent_name}.json"
            payload = load_matchup(path, policy_name, opponent_name, config)
            existing = {row["game_id"] for row in payload["records"]}
            expected_rows = [
                row for row in schedule
                if row["policy"] == policy_name and row["opponent"] == opponent_name
            ]
            for scheduled in expected_rows:
                identifier = scheduled["game_id"]
                if identifier in existing:
                    continue
                opening_index = int(scheduled["opening_index"])
                policy_white = scheduled["policy_color"] == "white"
                record = play_scheduled_game(
                    policy_agent, opponent_name, openings[opening_index - 1],
                    opening_index, policy_white, int(scheduled["opponent_seed"]),
                    identifier,
                )
                payload["records"].append(record)
                existing.add(identifier)
                payload["summary"] = summarize_records(payload["records"])
                atomic_json(path, payload)
                newly_played += 1
                if newly_played % 10 == 0:
                    print(
                        f"PROGRESS new_games={newly_played} total_saved="
                        f"{sum(len(json.loads(p.read_text(encoding='utf-8')).get('records', [])) for p in OUTPUT_DIR.glob('*_vs_*.json'))}/800 "
                        f"elapsed_seconds={time.perf_counter()-run_started:.3f}",
                        flush=True,
                    )
            if len(payload["records"]) != 100:
                raise RuntimeError(f"Incomplete matchup after run: {path}")
            ids = [row["game_id"] for row in payload["records"]]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate final game ID: {path}")
            all_ids.update(ids)
            payload["summary"] = summarize_records(payload["records"])
            atomic_json(path, payload)
            matchup_summaries[f"{policy_name}_vs_{opponent_name}"] = payload["summary"]
    if len(all_ids) != 800:
        raise RuntimeError(f"Expected 800 unique saved games, got {len(all_ids)}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "completed": True,
        "games": 800,
        "unique_game_ids": 800,
        "matchups": matchup_summaries,
        "run_elapsed_seconds": time.perf_counter() - run_started,
        "elo_not_computed": True,
        "paired_games_are_not_claimed_independent": True,
    }
    atomic_json(SUMMARY_PATH, summary)
    print(
        f"COMPLETE games=800 unique_game_ids=800 "
        f"elapsed_seconds={summary['run_elapsed_seconds']:.3f}", flush=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate hashes, openings, paths, and the 800-game schedule only",
    )
    parser.add_argument(
        "--opening-limit", type=int, default=50,
        help="Use the first N saved openings in their existing order (1..50)",
    )
    parser.add_argument(
        "--stop-file", type=Path,
        help="If this file exists between games, stop before starting the next game",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    openings, policies, config = validate_sources()
    schedule = build_schedule(args.opening_limit)
    effective_config = (
        config if args.opening_limit == 50
        else build_reduced_config(config, openings, args.opening_limit)
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "ok", "games": len(schedule),
                    "unique_game_ids": len({row['game_id'] for row in schedule}),
                    "openings": len(openings), "policies": policies,
                    "opening_limit": args.opening_limit,
                    "config_sha256": effective_config["config_sha256"],
                    "output_dir": str(
                        OUTPUT_DIR if args.opening_limit == 50
                        else reduced_output_dir(args.opening_limit)
                    ).replace("\\", "/"),
                },
                sort_keys=True,
            )
        )
        return
    runtime_dir = (
        OUTPUT_DIR if args.opening_limit == 50
        else reduced_output_dir(args.opening_limit)
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "worker_pid.txt").write_text(
        f"{os.getpid()}\n", encoding="utf-8"
    )
    if args.opening_limit == 50:
        run_evaluation()
    else:
        run_reduced(args.opening_limit, args.stop_file)


if __name__ == "__main__":
    parsed_args = parse_args()
    exit_code_path = (
        EXIT_CODE_PATH if parsed_args.opening_limit == 50
        else reduced_output_dir(parsed_args.opening_limit) / "exit_code.txt"
    )
    try:
        main(parsed_args)
    except Exception:
        exit_code_path.parent.mkdir(parents=True, exist_ok=True)
        exit_code_path.write_text("1\n", encoding="utf-8")
        raise
    else:
        if not parsed_args.preflight_only:
            exit_code_path.write_text("0\n", encoding="utf-8")
