"""Balanced reuse and real deadline cancellation for the reduced runner."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time

import pytest

from arena.session import GameSession
from scripts import run_all_models as core
from services.history import HistoryStore


PATH = next(core.ROOT.glob("*/chess/run_budget.py"))
spec = importlib.util.spec_from_file_location("budget_runner", PATH)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)
CONFIG = next(core.ROOT.glob("*/chess/experiment_cfg.json"))


def source_fixture(tmp_path):
    config = core.load_config(CONFIG, ["random", "think1"], smoke=True)
    frozen, _ = core.provenance(config)
    core.atomic_json(tmp_path / "manifest.json", frozen)
    store = HistoryStore(tmp_path / "history.sqlite3")
    records = []
    for job in core.schedule(config):
        game = GameSession(job["white"]["id"], job["black"]["id"],
                           board=core.opening_board(job["opening"], True), max_plies=config["max_plies"])
        while not game.finished:
            game.accept(game.state.get_legal_actions()[0])
        record = game.record()
        record.update(id=job["id"], run_id=frozen["fingerprint"], seeds=job["seeds"],
                      started_at=core.utc_now(), ended_at=core.utc_now(), pair=job["pair"])
        records.append(record)
    target = deepcopy(frozen)
    target["fingerprint"] = "new-run"
    target["config"]["mode"] = "reduced_time_budget"
    return store, records, target


def test_reuse_requires_both_colors_and_retains_original_identity(tmp_path):
    store, records, target = source_fixture(tmp_path)
    store.save(records[0])
    assert budget.choose_reuse(tmp_path, target) == []
    store.save(records[1])
    copied = budget.choose_reuse(tmp_path, target)
    assert len(copied) == 2
    assert copied[0]["origin"]["run_id"] == records[0]["run_id"]
    assert copied[0]["run_id"] == "new-run"
    assert store.list_games()[0]["run_id"] != "new-run"
    budget.balanced(copied, target["config"], "new-run")
    with pytest.raises(ValueError, match="Unbalanced"):
        budget.balanced(copied[:1], target["config"], "new-run")
    changed = deepcopy(target)
    changed["environment"]["python"] = "different"
    with pytest.raises(ValueError, match="environment"):
        budget.choose_reuse(tmp_path, changed)


def test_pair_transaction_rolls_back(tmp_path):
    store, records, _ = source_fixture(tmp_path)
    with store.connect() as db:
        db.execute(f"CREATE TRIGGER fail_second BEFORE INSERT ON games WHEN NEW.id = '{records[1]['id']}' BEGIN SELECT RAISE(ABORT,'test'); END")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        budget.save_pair(store, records)
    assert store.list_games() == []


def test_hard_deadline_kills_real_worker_without_counting_loss():
    config = core.load_config(CONFIG, ["random", "think1"], smoke=True)
    with pytest.raises(budget.BudgetExpired):
        budget.run_game(config, core.schedule(config)[0], time.monotonic() + 0.1)


def test_reduced_schedule_and_source_hash_unchanged():
    config = budget.config_for_budget(CONFIG, 10)
    assert len(core.schedule(config)) == 550
    assert len(config["openings"]) == 5
    assert config["max_plies"] == 300
    assert next(a for a in config["agents"] if a["id"] == "mcts_pure")["kwargs"]["simulations"] == 100
