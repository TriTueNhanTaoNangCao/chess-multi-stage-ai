"""Depth-limited minimax in equivalent negamax form."""
from agents.base import BaseAgent
from agents.stage1_heuristics.pst import evaluate_relative

MATE = 30000
INF = 1000000

def terminal_score(position, claim_draw, ply):
    outcome = position.outcome(claim_draw=claim_draw)
    if outcome is None:
        return None
    return 0 if outcome.winner is None else -MATE + ply

class MinimaxAgent(BaseAgent):
    def __init__(self, name=None, depth=2, player_id=None):
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        super().__init__(name or f"Minimax (Depth {depth})", player_id)
        self.depth = depth
        self.nodes_visited = 0
        self.last_score = 0
        self.claim_draw = True

    def _search(self, position, depth, ply=0):
        self.nodes_visited += 1
        terminal = terminal_score(position, self.claim_draw, ply)
        if terminal is not None:
            return terminal
        if depth == 0:
            return evaluate_relative(position, check_terminal=False)
        best = -INF
        for move in list(position.legal_moves):
            position.push(move)
            try:
                best = max(best, -self._search(position, depth - 1, ply + 1))
            finally:
                position.pop()
        return best

    def act(self, board):
        moves = board.get_legal_actions()
        if not moves:
            raise ValueError("No actions: terminal position")
        self.nodes_visited = 0
        self.claim_draw = board.claim_draw
        position = board.board.copy(stack=True)
        best, chosen = -INF, moves[0]
        for move in moves:
            position.push(move)
            try:
                score = -self._search(position, self.depth - 1, 1)
            finally:
                position.pop()
            if score > best:
                best, chosen = score, move
        self.last_score = best
        return chosen
