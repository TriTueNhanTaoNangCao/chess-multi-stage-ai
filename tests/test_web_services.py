from io import StringIO
from copy import deepcopy
import threading
import time

import chess
import chess.pgn
import pytest

from agents import RandomAgent, Think1Agent
from arena.evaluator import play_game
from arena.session import GameSession
from services.agent_config import AgentConfig
from services.game_controller import GameController
from services.history import HistoryStore, replay, standings, to_pgn


@pytest.fixture
def controller(tmp_path):
    return GameController(HistoryStore(tmp_path / "history.sqlite3"))


def drive(controller, until, timeout=10):
    end = time.monotonic() + timeout
    while not until():
        assert time.monotonic() < end, "Worker did not finish"
        controller.tick()
        time.sleep(0.002)


def test_human_validation_undo_and_old_click(controller):
    controller.start(AgentConfig("human"), AgentConfig("human"))
    token = controller.token
    with pytest.raises(ValueError):
        controller.human_move("e2e5", token)
    assert controller.session.calls == [0, 0]
    assert controller.human_move("e2e4", token)
    assert not controller.human_move("e7e5", token)
    controller.undo()
    assert controller.session.state.fen() == chess.Board().fen()
    assert not controller.human_move("e2e4", token)


def test_bot_illegal_move_and_timing():
    game = GameSession("bad", "good")
    game.accept(chess.Move.from_uci("e2e5"), 0.3)
    assert game.result == "0-1" and game.reason == "forfeit"
    assert game.times == [0.3, 0.0]
    assert game.moves == []
    assert game.telemetry[0]["error"]


def test_mate_promotion_pgn_and_replay(controller):
    controller.start(AgentConfig("think1"), AgentConfig("random"),
                     fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    drive(controller, lambda: controller.session.finished)
    record = controller.record()
    assert record["result"] == "1-0"
    assert replay(record).fen() == record["final_fen"]
    pgn = chess.pgn.read_game(StringIO(to_pgn(record)))
    assert pgn.end().board().fen() == record["final_fen"]
    assert pgn.headers["Result"] == "1-0"
    controller.start(AgentConfig("human"), AgentConfig("human"), fen="7k/P7/8/8/8/8/8/7K w - - 0 1")
    controller.human_move("a7a8n", controller.token)
    assert controller.session.state.piece_at(chess.A8).piece_type == chess.KNIGHT


def test_saved_abort_is_not_draw_and_history_survives_reopen(controller):
    controller.start(AgentConfig("human"), AgentConfig("human"))
    controller.human_move("e2e4", controller.token)
    first_id = controller.id
    controller.stop()
    controller.start(AgentConfig("human"), AgentConfig("human"))
    controller.resign(chess.WHITE)
    records = HistoryStore(controller.store.path).list_games()
    assert len(records) == 2
    aborted = next(r for r in records if r["id"] == first_id)
    assert aborted["result"] == "*" and aborted["moves"] == ["e2e4"]
    assert sum(r["games"] for r in standings(records)) == 2


def test_pause_step_and_stale_worker_after_restart(controller):
    released = threading.Event()
    entered = threading.Event()

    class Slow(RandomAgent):
        def act(self, board):
            entered.set()
            assert released.wait(5)
            return chess.Move.from_uci("e2e4")

    controller.start(AgentConfig("random"), AgentConfig("random", seed=7))
    controller.agents[0] = Slow()
    controller.tick()
    assert entered.wait(2)
    controller.running = False
    released.set()
    time.sleep(0.05)
    controller.tick()
    assert not controller.session.moves
    controller.start(AgentConfig("human"), AgentConfig("human"))
    controller.tick()
    assert not controller.session.moves
    assert not controller.worker.pending
    controller.start(AgentConfig("random"), AgentConfig("random", seed=7))
    controller.running = False
    controller.step_requested = True
    drive(controller, lambda: len(controller.session.moves) == 1)
    for _ in range(10):
        controller.tick()
    assert len(controller.session.moves) == 1


def test_tournament_balanced_colors_saved_immediately(controller):
    controller.delay = 0
    controller.start_tournament([AgentConfig("random"), AgentConfig("think1")], games=2, max_plies=2)
    drive(controller, lambda: controller.session.finished and not controller.schedule)
    records = controller.store.list_games()
    assert len(records) == 2
    assert records[0]["white"] == records[1]["black"]
    assert {r["result"] for r in records} == {"1/2-1/2"}
    assert len({r["run_id"] for r in records}) == 1
    assert all(len(r["telemetry"]) == 2 for r in records)


def test_import_idempotent_and_rejects_invalid_batch(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    record = play_game(RandomAgent(), Think1Agent(), max_plies=2).to_dict()
    assert store.import_tournament({"records": [record]}) == 1
    assert store.import_tournament({"records": [record]}) == 0
    with pytest.raises(ValueError):
        store.import_tournament({"records": [record, {**record, "moves": ["e2e5"]}]})
    assert len(store.list_games()) == 1


def test_config_factory_and_variants():
    assert AgentConfig("mcts_pure", simulations=7).build().simulations == 7
    assert AgentConfig("alphabeta", depth=1).build().depth == 1
    assert AgentConfig("alphabeta", depth=1).metadata()["id"] != AgentConfig("alphabeta", depth=2).metadata()["id"]
    with pytest.raises(ValueError):
        AgentConfig("supervised").build()


def test_history_preserves_repetition_policy():
    game = GameSession("a", "b")
    for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]:
        game.accept(chess.Move.from_uci(move), human=True)
    assert game.result == "1/2-1/2"
    assert replay(game.record()).outcome().result() == "1/2-1/2"


@pytest.mark.parametrize("patch", [
    {"configs": {"white": []}},
    {"configs": {"white": {"id": []}}},
    {"started_at": {}},
    {"white_seconds": float("nan")},
    {"black_calls": 1.5},
    {"telemetry": [{}]},
])
def test_import_rejects_bad_metadata_without_partial_writes(tmp_path, patch):
    store = HistoryStore(tmp_path / "history.sqlite3")
    record = GameSession("A", "B").record()
    invalid = {**deepcopy(record), **patch}
    with pytest.raises(ValueError):
        store.import_tournament({"records": [record, invalid]})
    assert store.list_games() == []


def test_import_counts_unique_insertions_and_keeps_existing_records(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    record = {**GameSession("A", "B").record(), "id": "same-game"}
    assert store.import_tournament({"records": [record, record]}) == 1
    assert len(store.list_games()) == 1
    assert store.import_tournament({"records": [{**record, "white": "changed"}]}) == 0
    assert store.list_games()[0]["white"] == "A"


def test_import_rolls_back_database_failure(tmp_path):
    import sqlite3
    store = HistoryStore(tmp_path / "history.sqlite3")
    with store.connect() as db:
        db.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON games
            WHEN NEW.id = 'second' BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
    record = GameSession("A", "B").record()
    with pytest.raises(sqlite3.IntegrityError):
        store.import_tournament({"records": [{**record, "id": "first"}, {**record, "id": "second"}]})
    assert store.list_games() == []


def test_invalid_new_game_keeps_current_match(controller):
    controller.start(AgentConfig("human"), AgentConfig("human"))
    controller.human_move("e2e4", controller.token)
    before, token = deepcopy(controller.record()), controller.token
    with pytest.raises(ValueError):
        controller.start(AgentConfig("human"), AgentConfig("random"), max_plies=0)
    assert controller.record() == before
    assert controller.token == token
    assert not controller.session.finished
