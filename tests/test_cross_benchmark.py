"""Cross-only scheduling and reporting must exclude both internal groups."""
from copy import deepcopy
import importlib.util
import json

import pytest

from arena.session import GameSession
from scripts import run_all_models as core

spec = importlib.util.spec_from_file_location("cross_runner", next(core.ROOT.glob("*/chess/run_cross.py")))
cross = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross)
CONFIG = next(core.ROOT.glob("*/chess/experiment_cfg.json"))


def test_cross_schedule_is_28_pairs_and_280_games():
    config = cross.budget.config_for_budget(CONFIG, 10)
    jobs = cross.cross_jobs(config)
    assert len(jobs) == 280 and len({j["pair"] for j in jobs}) == 28
    for i in range(0, len(jobs), 2):
        a, b = jobs[i:i+2]
        assert (a["white"]["key"] in cross.NEURAL) != (a["black"]["key"] in cross.NEURAL)
        assert a["white"] == b["black"] and a["opening"] == b["opening"]
    assert all(sum(j["pair"] == pair for j in jobs) == 10 for pair in {j["pair"] for j in jobs})


def test_classic_and_cross_scope_keeps_49_pairs_without_neural_internal():
    config = cross.budget.config_for_budget(CONFIG, 10)
    config["pair_scope"] = cross.CLASSIC_CROSS
    jobs = cross.cross_jobs(config)
    assert len(jobs) == 490 and len({j["pair"] for j in jobs}) == 49
    assert all(not (j["white"]["key"] in cross.NEURAL and j["black"]["key"] in cross.NEURAL) for j in jobs)
    internal = [j for j in jobs if j["white"]["key"] not in cross.NEURAL and j["black"]["key"] not in cross.NEURAL]
    assert len(internal) == 210 and len(jobs) - len(internal) == 280
    for pair in {j["pair"] for j in jobs}:
        selected = [j for j in jobs if j["pair"] == pair]
        assert len(selected) == 10
        assert sum(j["white"]["id"] == selected[0]["white"]["id"] for j in selected) == 5


def test_overlapping_reuse_is_deduplicated_but_conflicting_games_rejected(monkeypatch):
    config = core.load_config(CONFIG, ["random", "think1"], smoke=True)
    config["pair_scope"] = cross.CLASSIC_CROSS
    jobs = cross.cross_jobs(config)
    records = [dict(id=j["id"], run_id="new", origin={"folder": "source"}, result="1-0") for j in jobs]
    monkeypatch.setattr(cross.budget, "choose_reuse", lambda path, target: deepcopy(records))
    assert len(cross.reuse_records(["a", "b"], {}, jobs)) == 2
    def conflict(path, target):
        result = deepcopy(records)
        result[0]["result"] = "0-1" if path.name == "b" else "1-0"
        return result
    monkeypatch.setattr(cross.budget, "choose_reuse", conflict)
    with pytest.raises(ValueError, match="Conflicting"):
        cross.reuse_records(["a", "b"], {}, jobs)


def test_cross_report_counts_only_cross_games_and_rejects_internal(tmp_path):
    config = core.load_config(CONFIG, ["random", "think1"], smoke=True)
    config["agents"][1] = dict(id="alphazero", key="alphazero", kwargs={"simulations": 1})
    config["pair_scope"] = "classic_vs_neural_only"
    config["mode"] = "reduced_cross_only"
    frozen = dict(config=config, fingerprint="test", checkpoint_sha256={})
    store = core.HistoryStore(tmp_path / "history.sqlite3")
    records = []
    for job in cross.cross_jobs(config):
        game = GameSession(job["white"]["id"], job["black"]["id"],
                           board=core.opening_board(job["opening"], True), max_plies=config["max_plies"])
        while not game.finished:
            game.accept(game.state.get_legal_actions()[0])
        record = game.record()
        record.update(id=job["id"], run_id="test", seeds=job["seeds"], pair=job["pair"],
                      started_at=core.utc_now(), ended_at=core.utc_now())
        records.append(record)
    with pytest.raises(ValueError, match="Unbalanced"):
        cross.validate(records[:1], frozen)
    cross.budget.save_pair(store, records)
    summary = cross.report(tmp_path, frozen, store)
    assert summary["complete"] and summary["expected_games"] == 2
    assert summary["expected_pairs"] == 1 and summary["color_balanced"]
    payload = json.loads((tmp_path / "tournament_results.json").read_text(encoding="utf-8"))
    assert payload["summary"] == summary
    assert {r["pair"] for r in payload["matchups"]} == {"random__alphazero"}
    assert "2/2 games" in (tmp_path / "leaderboard.md").read_text()
    wrong = deepcopy(records)
    wrong[0]["id"] = "random__think1__o000__l0"
    with pytest.raises(ValueError, match="Non-cross"):
        cross.validate(wrong, frozen)


def test_combined_report_accepts_old_classic_pair(tmp_path):
    config = core.load_config(CONFIG, ["random", "think1"], smoke=True)
    config["pair_scope"] = cross.CLASSIC_CROSS
    config["mode"] = "reduced_classic_and_cross"
    frozen = dict(config=config, fingerprint="test", checkpoint_sha256={})
    store = core.HistoryStore(tmp_path / "history.sqlite3")
    records = []
    for job in cross.cross_jobs(config):
        game = GameSession(job["white"]["id"], job["black"]["id"],
                           board=core.opening_board(job["opening"], True), max_plies=config["max_plies"])
        while not game.finished:
            game.accept(game.state.get_legal_actions()[0])
        record = game.record()
        record.update(id=job["id"], run_id="test", seeds=job["seeds"], pair=job["pair"],
                      started_at=core.utc_now(), ended_at=core.utc_now(), origin={"folder": "old"})
        records.append(record)
    cross.budget.save_pair(store, records)
    summary = cross.report(tmp_path, frozen, store)
    assert summary["complete"] and summary["classic_internal_games"] == 2
    assert summary["cross_games"] == 0 and summary["reused_games"] == 2
    cross.validate(records, frozen, complete=True)
