"""Regression checks for lifecycle, errors and metadata in timed play."""
import threading
import time

import chess
import pytest

from agents.base import BaseAgent
from arena.session import measure_move
from arena.tournament import build_cli_agents
from games.chess.board import ChessBoard
from services.agent_config import AgentConfig
from services.game_controller import GameController
from services.history import HistoryStore
from services.search_worker import TimedAgent


class FirstSuccessThenError(BaseAgent):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def act(self, board):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("intentional second-move failure")
        return board.get_legal_actions()[0]


class UnresponsiveTimedSearch(BaseAgent):
    def act(self, board):
        return board.get_legal_actions()[0]

    def act_timed(self, board, limits):
        limits.publish(chess.Move.from_uci("e2e4"), completed_depth=1)
        time.sleep(5)
        return limits.best_move


class CloseProbe:
    name = "probe"
    closed = False

    def close(self):
        self.closed = True


def test_human_checkmate_closes_timed_opponent(tmp_path):
    controller = GameController(HistoryStore(tmp_path / "history.sqlite3"))
    controller.start(AgentConfig("human"), AgentConfig("random", move_time_limit_seconds=1),
                     fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    proxy = controller.agents[1]
    try:
        assert proxy.process.is_alive()
        controller.human_move("f7f8", controller.token)
        assert controller.session.finished
        assert proxy.process is None
    finally:
        controller.stop()


def test_failed_turn_does_not_inherit_previous_search_statistics():
    proxy = TimedAgent(("agent", FirstSuccessThenError()), 1)
    try:
        proxy.prepare()
        board = ChessBoard()
        move, _, error, stats = measure_move(proxy, board)
        assert not error and stats["fallback_used"] is False
        board.step(move)
        _, _, error, stats = measure_move(proxy, board)
        assert "intentional second-move failure" in error
        assert stats == {}
    finally:
        proxy.close()


def test_reset_cancels_inflight_search_and_starts_clean_worker():
    proxy = TimedAgent(("agent", UnresponsiveTimedSearch()), 1)
    result = []
    thread = None
    try:
        proxy.prepare()
        old_pid = proxy.process.pid
        thread = threading.Thread(target=lambda: result.append(measure_move(proxy, ChessBoard())))
        thread.start()
        time.sleep(0.1)
        proxy.reset()
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert proxy.process.is_alive() and proxy.process.pid != old_pid
        assert result and result[0][2]
        assert proxy.last_search_stats == {}
    finally:
        proxy.close()
        if thread:
            thread.join(timeout=3)


def test_timed_arena_metadata_retains_depth_and_seed_policy(monkeypatch):
    import arena.tournament as arena
    probe = CloseProbe()
    monkeypatch.setattr(arena, "build_agent", lambda *a, **kw: probe)
    agents, metadata = build_cli_agents(["alphabeta:4"], 2, 42, move_time_limit_seconds=10)
    assert metadata[0]["depth"] == 4
    assert metadata[0]["seed"] is None
    probe.close()


def test_arena_construction_error_closes_already_built_workers(monkeypatch):
    import arena.tournament as arena
    probe = CloseProbe()
    monkeypatch.setattr(arena, "build_agent", lambda *a, **kw: probe)
    with pytest.raises(ValueError, match="Unknown agent"):
        build_cli_agents(["random", "invalid"], 2, 42, move_time_limit_seconds=10)
    assert probe.closed



def test_main_watch_setup_failure_closes_first_worker(monkeypatch):
    from types import SimpleNamespace
    import main
    probe = CloseProbe()
    def factory(key, *args, **kwargs):
        if key == "invalid":
            raise ValueError("bad opponent")
        return probe
    monkeypatch.setattr(main, "build_agent", factory)
    args = SimpleNamespace(fen=None, ascii=True, watch=True, agent="random", opponent="invalid",
                           depth=2, seed=42, checkpoint=None, move_time_limit=10)
    with pytest.raises(ValueError, match="bad opponent"):
        main.run_cli(args)
    assert probe.closed


def test_duplicate_arena_names_close_constructed_workers(monkeypatch):
    import arena.tournament as arena
    probe = CloseProbe()
    monkeypatch.setattr(arena, "build_cli_agents", lambda *a: ([probe, probe], []))
    with pytest.raises(SystemExit):
        arena.main(["--agents", "random", "random"])
    assert probe.closed


class BoundaryConnection:
    def __init__(self, message):
        self.message = message
        self.poll_count = 0

    def send(self, message):
        pass

    def poll(self, timeout=0):
        self.poll_count += 1
        return self.poll_count == 2

    def recv(self):
        return self.message

    def close(self):
        pass


@pytest.mark.parametrize("kind", ["error", "illegal"])
def test_boundary_ipc_errors_are_not_silently_converted_to_fallback(monkeypatch, kind):
    payload = ("error", "intentional worker failure") if kind == "error" else (
        "best", ("e2e5", {}, time.monotonic()))
    proxy = TimedAgent(("agent", FirstSuccessThenError()), 0.1)
    proxy.connection = BoundaryConnection(payload)
    monkeypatch.setattr(proxy, "prepare", lambda: None)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            proxy.act(ChessBoard())
    finally:
        proxy.close()



def test_buffered_completion_is_not_mistaken_for_hard_timeout(monkeypatch):
    stats = {"completed_depth": 1, "simulations_completed": 0, "fallback_used": False,
             "time_exhausted": False, "hard_stop": False}
    connection = BoundaryConnection(("done", ("e2e4", stats, time.monotonic())))
    proxy = TimedAgent(("agent", FirstSuccessThenError()), 0.1)
    proxy.connection = connection
    monkeypatch.setattr(proxy, "prepare", lambda: None)
    try:
        assert proxy.act(ChessBoard()).uci() == "e2e4"
        assert proxy.connection is connection
        assert not proxy.last_search_stats["hard_stop"]
        assert not proxy.last_search_stats["time_exhausted"]
    finally:
        proxy.close()


def test_pygame_restart_and_undo_keep_budget_and_close_workers(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from games.chess.gui import ChessGUI
    gui = ChessGUI("random", depth=1, move_time_limit_seconds=1)
    proxies = [gui.agent]
    try:
        gui.restart()
        assert proxies[-1].process is None
        assert isinstance(gui.agent, TimedAgent) and gui.agent.seconds == 1
        proxies.append(gui.agent)
        gui.board.step("e4")
        gui.board.step("e5")
        gui.undo()
        assert proxies[-1].process is None
        assert isinstance(gui.agent, TimedAgent) and gui.agent.seconds == 1
        assert gui.board.fen() == ChessBoard().fen()
        proxies.append(gui.agent)
        gui.close()
        assert proxies[-1].process is None
    finally:
        gui.close()
        for proxy in proxies:
            proxy.close()



class ReloadClockAgent:
    name = "clock-agent"

    def __init__(self, clock):
        self.clock = clock
        self.closed = False
        self.prepared = False

    def prepare(self):
        if not self.prepared:
            self.clock[0] += 2.0
            self.prepared = True

    def act(self, board):
        self.prepare()
        self.clock[0] += 0.1
        return chess.Move.from_uci("f7f8")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("watch", [False, True])
def test_cli_clock_excludes_model_recovery(monkeypatch, capsys, watch):
    from types import SimpleNamespace
    import main
    clock = [0.0]
    actor = ReloadClockAgent(clock)
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(main, "build_agent", lambda *a, **kw: actor)
    args = SimpleNamespace(fen="7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", ascii=True,
                           watch=watch, agent="random", opponent="random", depth=2,
                           seed=42, checkpoint=None, move_time_limit=1, side="black", delay=0)
    assert main.run_cli(args) == 0
    assert "(0.100s)" in capsys.readouterr().out
    assert actor.closed


def test_pygame_clock_excludes_model_recovery(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import games.chess.gui as module
    clock = [0.0]
    actor = ReloadClockAgent(clock)
    gui = module.ChessGUI("random", human_color=chess.BLACK, move_delay=0,
                         board=ChessBoard("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"))
    gui.agent = actor
    gui.white_agent = actor
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    try:
        gui.start_worker()
        _, move, seconds, error = gui.results.get(timeout=3)
        assert not error and move.uci() == "f7f8"
        assert seconds == pytest.approx(0.1)
    finally:
        gui.close()
