"""Persistent, opt-in search process. Best-so-far messages stay in memory only."""
import atexit
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from threading import Lock, RLock, Thread
from time import monotonic

import chess

from agents.base import BaseAgent
from agents.search_limits import SearchExpired, SearchLimits, validate_seconds


def _watch_parent():
    parent = mp.parent_process()
    if parent is not None:
        wait([parent.sentinel])
        # Also clean up a grandchild if a benchmark deadline kills its parent.
        os._exit(0)


def _serve(connection, descriptor, torch_threads):
    Thread(target=_watch_parent, daemon=True).start()
    try:
        if torch_threads is not None:
            import torch
            torch.set_num_threads(torch_threads)
        kind, payload = descriptor
        if kind == "config":
            agent = payload._build_unlimited()
        elif kind == "factory":
            from agents import build_agent
            agent = build_agent(**payload)
        elif kind == "registry":
            from agents import AGENT_REGISTRY
            key, kwargs = payload
            agent = AGENT_REGISTRY[key](**kwargs)
        else:
            agent = payload
        agent.reset()
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            if command == "reset":
                agent.reset()
                connection.send(("ready", None))
                continue
            board, seconds, deadline = payload
            def report(move, stats, observed_at):
                connection.send(("best", (move.uci(), stats, observed_at)))
            # Leave a small IPC margin so cooperative searches can keep the loaded
            # model alive; the supervisor still enforces the user's hard ceiling.
            search_deadline = deadline - min(0.02, seconds * 0.1)
            limits = SearchLimits(board, seconds, deadline=search_deadline, report=report)
            simulations = getattr(agent, "simulations", None)
            if simulations is not None:
                limits.stats["simulation_budget"] = simulations
            exhausted = False
            try:
                move = agent.act_timed(board, limits)
                if move != limits.best_move or limits.stats["fallback_used"]:
                    limits.publish(move)
            except SearchExpired:
                exhausted = True
            stats = dict(limits.stats, time_limit_seconds=seconds, time_exhausted=exhausted,
                         hard_stop=False)
            for field in ("nodes_visited", "cutoffs"):
                value = getattr(agent, field, None)
                if isinstance(value, (int, float)):
                    stats[field] = value
            connection.send(("done", (limits.best_move.uci(), stats, monotonic())))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class TimedAgent(BaseAgent):
    """Only timed mode uses this facade. Startup is separate from move timing."""
    def __init__(self, descriptor, seconds, *, name="Timed AI", startup_timeout=300, torch_threads=None):
        super().__init__(name)
        validate_seconds(seconds)
        if seconds is None:
            raise ValueError("TimedAgent requires a positive move time limit")
        self.seconds = seconds
        self.descriptor = descriptor
        self.startup_timeout = startup_timeout
        self.torch_threads = torch_threads
        self.process = None
        self.connection = None
        self.last_search_stats = {}
        self._closed = False
        self._state_lock = RLock()
        self._operation_lock = Lock()
        self._startup_lock = Lock()
        atexit.register(self.close)

    def prepare(self):
        # Serialize startup, but let close() cancel it without waiting for loading.
        with self._startup_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Search worker is closed")
                if self.process is not None and self.process.is_alive():
                    return
                self._discard()
                context = mp.get_context("spawn")
                connection, child = context.Pipe()
                process = context.Process(target=_serve, args=(child, self.descriptor, self.torch_threads), daemon=True)
                self.connection, self.process = connection, process
                try:
                    process.start()
                except BaseException:
                    self._discard(expected=process)
                    raise
                finally:
                    child.close()
            try:
                if not connection.poll(self.startup_timeout):
                    raise RuntimeError("Timed AI startup timeout")
                kind, payload = connection.recv()
                if kind != "ready":
                    raise RuntimeError(f"Timed AI startup failed: {payload}")
                with self._state_lock:
                    if self._closed or self.process is not process:
                        raise RuntimeError("Search worker startup was cancelled")
            except BaseException:
                self._discard(expected=process)
                raise

    def reset(self):
        # A reset during act() must cancel that request, never share its reader.
        acquired = self._operation_lock.acquire(blocking=False)
        if not acquired:
            self._discard()
            acquired = self._operation_lock.acquire(timeout=5)
            if not acquired:
                raise RuntimeError("Could not cancel the active search for reset")
        try:
            self.last_search_stats = {}
            self.prepare()
            with self._state_lock:
                connection, process = self.connection, self.process
            if connection is None:
                raise RuntimeError("Search worker is closed")
            try:
                connection.send(("reset", None))
                if not connection.poll(self.startup_timeout):
                    raise RuntimeError("Timed AI reset timeout")
                kind, payload = connection.recv()
                if kind != "ready":
                    raise RuntimeError(f"Timed AI reset failed: {payload}")
            except BaseException:
                self._discard(expected=process)
                raise
        finally:
            self._operation_lock.release()

    def act(self, board):
        with self._operation_lock:
            return self._act(board)

    def _act(self, board):
        self.last_search_stats = {}
        self.prepare()
        legal = frozenset(board.get_legal_actions())
        if not legal or board.is_terminal():
            raise ValueError("No actions: terminal position")
        best = board.get_legal_actions()[0]
        stats = dict(completed_depth=0, simulations_completed=0, fallback_used=True)
        deadline = monotonic() + self.seconds
        with self._state_lock:
            connection = self.connection
        if connection is None:
            raise RuntimeError("Search worker is closed")
        try:
            connection.send(("move", (board.clone(), self.seconds, deadline)))
            while True:
                remaining = deadline - monotonic()
                available = remaining > 0 and connection.poll(remaining)
                # Process a final already-buffered message using exactly the same
                # validation as the normal path (including errors and done).
                if not available and not connection.poll(0):
                    break
                kind, payload = connection.recv()
                if kind == "error":
                    self._discard()
                    raise RuntimeError(payload)
                if kind not in ("best", "done"):
                    self._discard()
                    raise RuntimeError(f"Unexpected search message: {kind}")
                uci, candidate_stats, observed_at = payload
                move = chess.Move.from_uci(uci)
                if move not in legal:
                    self._discard()
                    raise ValueError(f"Illegal search candidate: {uci}")
                if observed_at <= deadline:
                    best, stats = move, candidate_stats
                if kind == "done":
                    if observed_at > deadline:
                        stats = dict(stats, time_exhausted=True)
                    self.last_search_stats = dict(stats, time_limit_seconds=self.seconds,
                                                  hard_stop=False)
                    return best
            # No completion message: terminate hung inference and use the last
            # pre-deadline recommendation; model recovery happens before next turn.
            self.last_search_stats = dict(stats, time_limit_seconds=self.seconds,
                                          time_exhausted=True, hard_stop=True)
            self._discard()
            return best
        except (EOFError, OSError):
            self._discard()
            raise RuntimeError("Search worker exited during a move") from None

    def _discard(self, *, expected=None):
        # Atomically detach resources: concurrent stop/deadline/reset can then
        # clean a process only once, and cannot kill a replacement worker.
        with self._state_lock:
            if expected is not None and self.process is not expected:
                return
            process, connection = self.process, self.connection
            self.process = self.connection = None
        if process is not None:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=0.2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.2)
            process.close()
        if connection is not None:
            connection.close()

    def close(self):
        with self._state_lock:
            self._closed = True
        self._discard()
        atexit.unregister(self.close)
