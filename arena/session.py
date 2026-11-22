"""Incremental game rules shared by the web UI and synchronous arena."""
from time import perf_counter

from games.chess.board import ChessBoard


def measure_move(agent, board):
    """Time inference only, on a private snapshot; report optional search statistics."""
    snapshot = board.clone()
    started = perf_counter()
    move, error, stats = None, "", {}
    try:
        move = agent.act(snapshot)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    seconds = perf_counter() - started
    for attr in ("nodes_visited", "cutoffs", "last_score"):
        value = getattr(agent, attr, None)
        if isinstance(value, (int, float)):
            stats[attr] = value
    if hasattr(agent, "simulations"):
        stats["simulation_budget"] = agent.simulations
    root = getattr(agent, "root", None)
    if root is None:
        root = getattr(getattr(agent, "mcts", None), "root", None)
    if root is not None:
        stats["root_visits"] = root.visits
        stats["root_value"] = root.mean_value
    return move, seconds, error, stats


class GameSession:
    def __init__(self, white, black, *, board=None, max_plies=300):
        if not isinstance(max_plies, int) or max_plies < 1:
            raise ValueError("max_plies must be a positive integer")
        self.state = board.clone() if board is not None else ChessBoard()
        self.initial_fen = self.state.fen()
        self.root_fen = self.state.board.root().fen(en_passant="fen")
        self.initial_moves = [m.uci() for m in self.state.board.move_stack]
        self.names = [white, black]
        self.max_plies = max_plies
        self.moves, self.telemetry = [], []
        self.times, self.calls = [0.0, 0.0], [0, 0]
        self.result, self.reason, self.error = "*", "", ""
        self.refresh_outcome()

    @property
    def finished(self):
        return bool(self.reason)

    @property
    def index(self):
        return 0 if self.state.turn else 1

    def refresh_outcome(self):
        if self.finished:
            return
        outcome = self.state.outcome()
        if outcome:
            self.result, self.reason = outcome.result(), outcome.termination.name.lower()
        elif len(self.moves) >= self.max_plies:
            self.result, self.reason = "1/2-1/2", "max_plies_adjudication"

    def reset_agents(self, agents):
        failures = {}
        for i, agent in enumerate(agents):
            if agent is None:
                continue
            try:
                agent.reset()
            except Exception as exc:
                failures[i] = f"{self.names[i]}: {type(exc).__name__}: {exc}"
        if failures:
            self.result = "1/2-1/2" if len(failures) == 2 else ("0-1" if 0 in failures else "1-0")
            self.reason = "double_forfeit" if len(failures) == 2 else "reset_forfeit"
            self.error = "; ".join(failures.values())

    def accept(self, move, seconds=0.0, error="", stats=None, *, human=False):
        if self.finished:
            raise ValueError("Game has ended")
        if not error and not self.state.is_legal(move):
            if human:
                raise ValueError("Nước đi không hợp lệ")
            error = f"Illegal action: {move!r}"
        index = self.index
        self.times[index] += seconds
        self.calls[index] += 1
        entry = {"ply": len(self.moves) + 1, "color": "white" if index == 0 else "black",
                 "seconds": seconds, **(stats or {})}
        if error:
            self.result, self.reason, self.error = ("0-1" if index == 0 else "1-0"), "forfeit", error
            self.telemetry.append({**entry, "error": error})
            return
        entry.update(uci=move.uci(), san=self.state.san(move))
        self.state.step(move)
        self.moves.append(move.uci())
        self.telemetry.append(entry)
        self.refresh_outcome()

    def undo(self, plies=1):
        if self.finished:
            raise ValueError("Ván đã kết thúc; hãy tạo ván mới")
        for _ in range(min(plies, len(self.moves))):
            entry = self.telemetry.pop()
            index = 0 if entry["color"] == "white" else 1
            self.times[index] = max(0.0, self.times[index] - entry["seconds"])
            self.calls[index] -= 1
            self.moves.pop()
            self.state.pop()

    def abort(self):
        if not self.finished:
            self.result, self.reason = "*", "aborted"

    def resign(self, color):
        if not self.finished:
            self.result, self.reason = ("0-1" if color else "1-0"), "resignation"

    def record(self):
        return dict(white=self.names[0], black=self.names[1], result=self.result,
                    reason=self.reason, error=self.error, moves=list(self.moves),
                    initial_fen=self.initial_fen, final_fen=self.state.fen(),
                    white_seconds=self.times[0], black_seconds=self.times[1],
                    white_calls=self.calls[0], black_calls=self.calls[1],
                    root_fen=self.root_fen, initial_moves=list(self.initial_moves),
                    claim_draw=self.state.claim_draw, telemetry=list(self.telemetry))
