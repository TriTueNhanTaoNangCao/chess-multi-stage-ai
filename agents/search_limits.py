"""Optional per-move search state. Never constructed by unrestricted searches."""
import math
from time import monotonic


class SearchExpired(Exception):
    """Normal end of a time-limited search, not a bot failure."""


def validate_seconds(seconds):
    if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Move time limit must be a finite number > 0, or None")
    return seconds


class SearchLimits:
    def __init__(self, board, seconds, *, deadline=None, report=None):
        validate_seconds(seconds)
        moves = board.get_legal_actions()
        if not moves or board.is_terminal():
            raise ValueError("No actions: terminal position")
        self.deadline = monotonic() + seconds if deadline is None else deadline
        self.best_move = moves[0]
        self.stats = {"completed_depth": 0, "simulations_completed": 0, "fallback_used": True}
        self._legal = frozenset(moves)
        self._report = report
        self._last_report = 0.0

    def check(self):
        if monotonic() >= self.deadline:
            raise SearchExpired()

    def publish(self, move, **stats):
        self.check()
        if move not in self._legal:
            raise ValueError(f"Illegal search candidate: {move}")
        changed = self.best_move != move or self.stats["fallback_used"]
        self.best_move = move
        self.stats.update(stats, fallback_used=False)
        now = monotonic()
        # Every changed recommendation is sent; unchanged counters at most 20 Hz.
        if self._report and (changed or now - self._last_report >= 0.05):
            self._report(move, dict(self.stats), now)
            self._last_report = now


def rollout_position(board, max_depth, rng, limits):
    """Timed rollout on a private board; interrupted playouts are not backed up."""
    position = board.board.copy(stack=True)
    for _ in range(max_depth):
        limits.check()
        if position.outcome(claim_draw=board.claim_draw) is not None:
            break
        moves = list(position.legal_moves)
        if not moves:
            break
        position.push(rng.choice(moves))
    limits.check()
    return position


