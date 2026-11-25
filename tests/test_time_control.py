"""Optional move budgets, completed search results and hard process deadlines."""
import time

import chess
import pytest

from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.search_limits import SearchExpired, SearchLimits, validate_seconds
from agents.stage2_tree_search import AlphaBetaAgent, MinimaxAgent
from agents.stage3_mcts_pure import PureMCTSAgent
from arena.session import measure_move
from games.chess.board import ChessBoard
from services.agent_config import AgentConfig
from services.search_worker import TimedAgent


class PublishedThenHung(BaseAgent):
    def act(self, board):
        raise AssertionError("Unrestricted entrypoint must not be used")

    def act_timed(self, board, limits):
        limits.publish(chess.Move.from_uci("e2e4"), completed_depth=1)
        time.sleep(5)


class OnePassHung(BaseAgent):
    def act(self, board):
        time.sleep(5)
        return chess.Move.from_uci("e2e4")


class OffOnly(RandomAgent):
    def act_timed(self, board, limits):
        raise AssertionError("No tracking allowed when the limit is off")


@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), float("inf")])
def test_invalid_budget(seconds):
    with pytest.raises(ValueError):
        validate_seconds(seconds)


def test_off_path_does_not_track_and_keeps_seeded_result():
    board = ChessBoard()
    expected = RandomAgent(seed=19).act(board)
    agent = OffOnly(seed=19)
    move, _, error, stats = measure_move(agent, board)
    assert move == expected and not error
    assert "time_limit_seconds" not in stats
    assert not hasattr(agent, "last_search_stats")
    assert not isinstance(AgentConfig("random").build(), TimedAgent)


@pytest.mark.parametrize("cls", [MinimaxAgent, AlphaBetaAgent])
def test_iterative_deepening_retains_completed_depth_and_input(cls):
    board = ChessBoard()
    before = board.fen(), list(board.board.move_stack)
    agent = cls(depth=10)
    limits = SearchLimits(board, 0.3)
    with pytest.raises(SearchExpired):
        agent.act_timed(board, limits)
    assert limits.stats["completed_depth"] >= 1
    assert not limits.stats["fallback_used"] and board.is_legal(limits.best_move)
    assert (board.fen(), list(board.board.move_stack)) == before


@pytest.mark.parametrize("cls", [MinimaxAgent, AlphaBetaAgent])
def test_completed_timed_search_matches_fixed_depth_choice(cls):
    board = ChessBoard()
    expected = cls(depth=2).act(board)
    limits = SearchLimits(board, 10)
    assert cls(depth=2).act_timed(board, limits) == expected
    assert limits.stats["completed_depth"] == 2


def test_mcts_timed_simulations_are_completed_and_board_immutable():
    board = ChessBoard()
    before = board.fen()
    limits = SearchLimits(board, 10)
    agent = PureMCTSAgent(simulations=7, max_rollout_depth=2, seed=42)
    move = agent.act_timed(board, limits)
    assert board.is_legal(move) and board.fen() == before
    assert limits.stats["simulations_completed"] == 7
    assert agent.root.visits == 7


@pytest.mark.parametrize("agent,expected_fallback", [(PublishedThenHung(), False), (OnePassHung(), True)])
def test_hard_stop_uses_last_published_move_or_legal_fallback(agent, expected_fallback):
    proxy = TimedAgent(("agent", agent), 0.25)
    try:
        proxy.prepare()
        board = ChessBoard()
        started = time.monotonic()
        move, elapsed, error, stats = measure_move(proxy, board)
        assert time.monotonic() - started < 1.5
        assert elapsed < 1.5 and not error and board.is_legal(move)
        assert stats["time_exhausted"] and stats["hard_stop"]
        assert stats["fallback_used"] == expected_fallback
        if not expected_fallback:
            assert move.uci() == "e2e4" and stats["completed_depth"] == 1
        assert proxy.process is None
    finally:
        proxy.close()


def test_fast_worker_returns_early_and_reuses_loaded_agent():
    proxy = TimedAgent(("registry", ("random", {"seed": 42})), 2)
    try:
        proxy.prepare()
        process_id = proxy.process.pid
        board = ChessBoard()
        for _ in range(2):
            move, elapsed, error, stats = measure_move(proxy, board)
            assert not error and board.is_legal(move) and elapsed < 1
            assert not stats["time_exhausted"] and not stats["fallback_used"]
            assert proxy.process.pid == process_id
            board.step(move)
    finally:
        proxy.close()


def test_time_limit_changes_config_identity_only_when_enabled():
    ordinary = AgentConfig("random")
    assert ordinary.metadata()["id"] == AgentConfig("random", move_time_limit_seconds=None).metadata()["id"]
    assert ordinary.metadata()["id"] != AgentConfig("random", move_time_limit_seconds=10).metadata()["id"]


@pytest.mark.parametrize("engine", ["alphago", "alphazero"])
def test_neural_mcts_timed_uses_completed_visits_without_mutation(engine):
    from agents.stage6_alphago.mcts import AlphaGoMCTS
    from agents.stage7_alphazero.mcts import AlphaZeroMCTS
    def policy(board):
        moves = board.get_legal_actions()
        return {m: 1 / len(moves) for m in moves}
    if engine == "alphago":
        search = AlphaGoMCTS(policy_fn=policy, value_fn=lambda b: 0.0, rollout_weight=0,
                             simulations=20, seed=42)
    else:
        search = AlphaZeroMCTS(predict_fn=lambda b: (policy(b), 0.0), simulations=20, seed=42)
    board = ChessBoard()
    before = board.fen(), list(board.board.move_stack)
    limits = SearchLimits(board, 10)
    move = search.search_timed(board, limits)
    assert board.is_legal(move) and limits.stats["simulations_completed"] == 20
    assert limits.stats["root_visits"] == 20
    assert (board.fen(), list(board.board.move_stack)) == before


def test_cooperative_deadline_keeps_worker_and_model_alive():
    proxy = TimedAgent(("registry", ("minimax", {"depth": 10})), 0.3)
    try:
        proxy.prepare()
        pid = proxy.process.pid
        board = ChessBoard()
        move, _, error, stats = measure_move(proxy, board)
        assert not error and board.is_legal(move) and stats["time_exhausted"]
        assert not stats["hard_stop"] and proxy.process.pid == pid
        assert stats["completed_depth"] >= 1
    finally:
        proxy.close()



def test_stopping_timed_gui_worker_discards_stale_result(tmp_path):
    from services.game_controller import GameController
    from services.history import HistoryStore
    controller = GameController(HistoryStore(tmp_path / "history.sqlite3"))
    controller.start(AgentConfig("random"), AgentConfig("human"))
    proxy = TimedAgent(("agent", PublishedThenHung()), 3)
    proxy.prepare()
    controller.agents[0] = proxy
    controller.tick()
    assert controller.worker.pending
    time.sleep(0.05)
    controller.stop()
    controller.start(AgentConfig("human"), AgentConfig("human"))
    end = time.monotonic() + 5
    while controller.worker.pending:
        assert time.monotonic() < end
        controller.tick()
        time.sleep(0.01)
    assert not controller.session.moves
    assert proxy.process is None
    controller.stop()


def test_timed_benchmark_resume_and_telemetry_validation(tmp_path):
    from scripts import run_all_models as core
    import json
    config = core.load_config(next(core.ROOT.glob("*/chess/experiment_cfg.json")), ["random", "think1"], smoke=True)
    config["move_time_limit_seconds"] = 0.2
    assert not core.run(config, tmp_path, limit=1)
    assert core.run(config, tmp_path)
    frozen = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    records = core.HistoryStore(tmp_path / "history.sqlite3").list_games()
    assert len(records) == 2 and not any(r["error"] for r in records)
    assert all(e["time_limit_seconds"] == 0.2 for r in records for e in r["telemetry"])
    core.validate_records(records, config, frozen["fingerprint"], require_complete=True)
    records[0]["telemetry"][0]["time_limit_seconds"] = 99
    with pytest.raises(ValueError, match="time limit telemetry"):
        core.validate_records(records, config, frozen["fingerprint"])
