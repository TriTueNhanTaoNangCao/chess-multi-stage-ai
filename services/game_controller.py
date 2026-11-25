"""Non-blocking orchestration; workers never access UI or mutate live boards."""
from itertools import combinations
from queue import Empty, Queue
from threading import Thread
from time import monotonic

import chess
from arena.session import GameSession, measure_move
from games.chess.board import ChessBoard
from services.history import new_record_id, utc_now


class MoveWorker:
    def __init__(self):
        self.pending = False
        self.queue = Queue()

    def submit(self, token, agent, board):
        if self.pending:
            return False
        snapshot = board.clone()
        self.pending = True

        def work():
            self.queue.put((token, measure_move(agent, snapshot)))

        Thread(target=work, daemon=True, name="chess-move").start()
        return True

    def poll(self):
        try:
            result = self.queue.get_nowait()
        except Empty:
            return None
        self.pending = False
        return result


class GameController:
    def __init__(self, store):
        self.store = store
        self.worker = MoveWorker()
        self.session = None
        self.agents = []
        self.generation = 0
        self.running = False
        self.step_requested = False
        self.delay = 0.4
        self.next_at = 0.0
        self.schedule = []
        self.schedule_total = 0
        self.schedule_done = 0
        self.run_id = ""
        self.last_error = ""
        self.dirty = False

    def start(self, white, black, *, fen="", max_plies=300, mode="watch", run_id=None):
        # Validate everything before replacing a running match.
        board = ChessBoard(fen or None)
        configs = {"white": white.metadata(), "black": black.metadata()}
        agents = []
        try:
            agents.append(white.build())
            agents.append(black.build())
        except BaseException:
            for agent in agents:
                close = getattr(agent, "close", None)
                if close:
                    close()
            raise
        session = GameSession(white.label, black.label, board=board, max_plies=max_plies)
        session.reset_agents(agents)
        if self.session and not self.session.finished:
            self.stop()
        else:
            self._close_agents()
        self.generation += 1
        self.agents = agents
        self.config_objects = (white, black)
        self.configs = configs
        self.session = session
        self.id = new_record_id()
        self.run_id = run_id or new_record_id()
        self.started_at, self.ended_at = utc_now(), ""
        self.mode = mode
        self.last_error = ""
        if mode != "tournament":
            self.schedule.clear()
        self.running, self.step_requested = True, False
        self.next_at = monotonic()
        self.save()
        if self.session.finished:
            self._close_agents()

    def record(self):
        if self.session is None:
            raise ValueError("Chưa có ván đấu")
        return {**self.session.record(), "id": self.id, "run_id": self.run_id,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "mode": self.mode, "configs": self.configs,
                "max_plies": self.session.max_plies}

    def save(self):
        if self.session is None:
            return
        self.dirty = True
        if self.session.finished and not self.ended_at:
            self.ended_at = utc_now()
        self.store.save(self.record())
        self.dirty = False

    def human_move(self, uci, token):
        if not self.session or token != self.token or self.session.finished:
            return False
        if self.agents[self.session.index] is not None:
            return False
        self.session.accept(chess.Move.from_uci(uci), human=True)
        self.save()
        if self.session.finished:
            self._close_agents()
        return True

    @property
    def token(self):
        return f"{self.generation}:{len(self.session.moves) if self.session else 0}"

    def tick(self):
        if not self.session:
            return
        if self.dirty:
            self.save()
        # When paused, hold any finished worker result until resume/step.
        active = self.running or self.step_requested
        if active or self.session.finished:
            reply = self.worker.poll()
            if reply:
                token, measured = reply
                if token == self.token and not self.session.finished:
                    self.session.accept(*measured)
                    self.step_requested = False
                    self.next_at = monotonic() + self.delay
                    self.save()
                    if self.session.finished:
                        self._close_agents()
        if self.session.finished:
            if self.schedule and self.running and not self.worker.pending and monotonic() >= self.next_at:
                white, black = self.schedule[0]
                try:
                    self.start(white, black, max_plies=self.session.max_plies,
                               mode="tournament", run_id=self.run_id)
                except Exception as exc:
                    self.last_error = str(exc)
                    self.running = False
                    return
                self.schedule.pop(0)
                self.schedule_done += 1
            return
        if not (self.running or self.step_requested) or monotonic() < self.next_at:
            return
        agent = self.agents[self.session.index]
        if agent is not None:
            self.worker.submit(self.token, agent, self.session.state)

    def stop(self):
        self.schedule.clear()
        self.running = self.step_requested = False
        self.generation += 1
        self._close_agents()
        if self.session:
            self.session.abort()
            self.save()

    def resign(self, color):
        if self.session is None:
            return
        self.generation += 1
        self._close_agents()
        self.session.resign(color)
        self.save()

    def undo(self):
        if self.session is None:
            return
        if self.worker.pending:
            raise ValueError("Chờ AI tính xong trước khi lùi nước")
        count = 1
        if self.mode == "play" and self.agents[self.session.index] is None and len(self.session.moves) >= 2:
            count = 2
        self.session.undo(count)
        self.generation += 1
        self.session.reset_agents(self.agents)
        self.save()

    def start_tournament(self, configs, *, games=2, max_plies=300):
        if len(configs) < 2 or games < 2 or games % 2:
            raise ValueError("Cần ít nhất 2 bot và số ván/cặp chẵn, từ 2 trở lên")
        ids = [c.metadata()["id"] for c in configs]
        if len(ids) != len(set(ids)):
            raise ValueError("Các bot trong giải phải có cấu hình khác nhau")
        # Validate all participants before starting the first game.
        for config in configs:
            checked = config.build()
            close = getattr(checked, "close", None)
            if close:
                close()
        schedule = []
        for a, b in combinations(configs, 2):
            schedule.extend((a, b) if i % 2 == 0 else (b, a) for i in range(games))
        white, black = schedule.pop(0)
        self.start(white, black, max_plies=max_plies, mode="tournament")
        self.schedule = schedule
        self.schedule_total = len(schedule) + 1
        self.schedule_done = 0

    def _close_agents(self):
        for agent in self.agents:
            close = getattr(agent, "close", None)
            if close:
                close()
