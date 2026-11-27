"""Acceptance checks for the resumable experiment runner."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path

import chess.pgn
import pytest

from arena.session import GameSession
from scripts.run_all_models import (ROOT, bind_manifest, build_agent, export_report, load_config,
                                   opening_board, provenance, run, schedule, statistics, timeout_record,
                                   validate_records)
from services.history import HistoryStore


CONFIG = "thực nghiệm/chess/experiment_cfg.json"


def test_official_schedule_has_55_pairs_100_white_100_black_and_same_starts():
    config = load_config(CONFIG)
    jobs = schedule(config)
    assert len(jobs) == 11000
    assert len({j["pair"] for j in jobs}) == 55
    for index in range(0, len(jobs), 2):
        a, b = jobs[index:index+2]
        assert a["white"] == b["black"] and a["black"] == b["white"]
        assert a["opening"] == b["opening"] and a["seeds"] == b["seeds"]
        assert opening_board(a["opening"], True).turn
    for pair in {j["pair"] for j in jobs}:
        subset = [j for j in jobs if j["pair"] == pair]
        assert len(subset) == 200
        assert sum(j["white"]["id"] == subset[0]["white"]["id"] for j in subset) == 100
    assert jobs == schedule(config)


def test_missing_checkpoints_and_manifest_resume_guards(tmp_path):
    config = load_config(CONFIG, ["random", "think1"], smoke=True)
    frozen, missing = provenance(config)
    assert not missing
    bind_manifest(tmp_path, frozen)
    bind_manifest(tmp_path, frozen)
    changed = deepcopy(frozen)
    changed["fingerprint"] = "changed"
    with pytest.raises(ValueError, match="Resume refused"):
        bind_manifest(tmp_path, changed)
    # Neural agents are always passed explicitly; no default-path fallback.
    bad = {"id": "az", "key": "alphazero", "kwargs": {"checkpoint": str(tmp_path / "missing.pt")}}
    with pytest.raises(FileNotFoundError):
        build_agent(bad, 42, "cpu")


def test_timeout_keeps_replay_and_counts_as_forfeit():
    game = GameSession("a", "b")
    game.accept(chess.Move.from_uci("e2e4"))
    record = timeout_record(game.record(), 2)
    assert record["result"] == "1-0" and record["timeout"]
    assert record["black_calls"] == 1 and record["black_seconds"] == 2
    assert record["final_fen"] == game.state.fen()
    assert record["telemetry"][-1]["color"] == "black"
    row = next(r for r in statistics([record], ["a", "b"]) if r["agent"] == "a")
    assert row["win_pct"] == 100 and row["white_win_pct"] == 100 and row["black_win_pct"] is None


def test_partial_run_resume_verify_export_and_no_duplicates(tmp_path):
    config = load_config(CONFIG, ["random", "think1"], smoke=True)
    assert not run(config, tmp_path, limit=1)
    assert json.loads((tmp_path / "progress.json").read_text())["completed_games"] == 1
    assert run(config, tmp_path)
    store = HistoryStore(tmp_path / "history.sqlite3")
    records = store.list_games()
    assert len(records) == 2
    frozen = json.loads((tmp_path / "manifest.json").read_text())
    assert validate_records(records, config, frozen["fingerprint"], require_complete=True) == 2
    summary = export_report(tmp_path, frozen, store)
    assert summary["complete"] and summary["clean"]
    for row in json.loads((tmp_path / "tournament_results.json").read_text(encoding="utf-8"))["leaderboard"]:
        assert row["white_games"] == row["black_games"] == 1
    assert run(config, tmp_path)  # repeated invocation skips both existing games
    assert len(store.list_games()) == 2
    pgn = StringIO((tmp_path / "games.pgn").read_text(encoding="utf-8"))
    games = [chess.pgn.read_game(pgn), chess.pgn.read_game(pgn)]
    assert all(g and not g.errors for g in games)
    damaged = deepcopy(records)
    damaged[0]["final_fen"] = "wrong"
    with pytest.raises(ValueError, match="Replay mismatch"):
        validate_records(damaged, config, frozen["fingerprint"])
    with pytest.raises(ValueError, match="Incomplete"):
        validate_records(records[:1], config, frozen["fingerprint"], require_complete=True)


def test_value_checkpoint_uses_value_schema_without_action_fields(tmp_path):
    import torch
    from models.cnn_policy import CNNPolicyNetwork
    from models.value_net import ValueNetwork
    policy = CNNPolicyNetwork(hidden_dim=4, fc_dim=8)
    value = ValueNetwork(hidden_dim=4, fc1_dim=8)
    common = dict(encoder_version="hwc14-white-v1", claim_draw=True)
    policy_path, value_path = tmp_path / "policy.pt", tmp_path / "value.pt"
    torch.save(dict(**common, model_class="CNNPolicyNetwork", model_config=policy.get_config(),
                    state_dict=policy.state_dict(), action_version="az73-rankflip-v1", num_actions=4672), policy_path)
    torch.save(dict(**common, model_class="ValueNetwork", model_config=value.get_config(),
                    state_dict=value.state_dict(), value_perspective="side_to_move"), value_path)
    spec = dict(id="ag", key="alphago", kwargs=dict(checkpoint_policy=str(policy_path),
                checkpoint_value=str(value_path), simulations=1))
    agent = build_agent(spec, 42, "cpu")
    assert agent.policy_net is not None and agent.value_net is not None


def test_real_worker_timeout_can_be_terminated():
    from scripts.run_all_models import isolated_job
    config = load_config(CONFIG, ["random", "minimax"], smoke=True)
    config["move_timeout_seconds"] = 1
    config["agents"][1]["kwargs"]["depth"] = 6
    record = isolated_job(config, schedule(config)[0])
    assert record["timeout"] and record["result"] == "1-0"
    assert record["telemetry"][-1]["color"] == "black"
    assert record["moves"]  # Random's completed move survives the timeout.


def test_statistics_percentages_by_color():
    base = GameSession("a", "b").record()
    records = [{**base, "result": "1-0"},
               {**base, "white": "b", "black": "a", "result": "1/2-1/2"}]
    a = next(row for row in statistics(records, ["a", "b"]) if row["agent"] == "a")
    assert (a["wins"], a["draws"], a["losses"]) == (1, 1, 0)
    assert a["win_pct"] == 50 and a["score_pct"] == 75
    assert a["white_win_pct"] == 100 and a["black_win_pct"] == 0
