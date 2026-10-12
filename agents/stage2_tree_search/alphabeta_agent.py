"""Alpha-beta with capture ordering and bounded quiescence."""
from agents.stage1_heuristics.pst import PIECE_VALUES, evaluate_relative
from agents.stage1_heuristics.think2_agent import capture_score
from .minimax_agent import MinimaxAgent, terminal_score, INF

class AlphaBetaAgent(MinimaxAgent):
    def __init__(self, name=None, depth=3, use_quiescence=True, player_id=None, q_depth=6):
        super().__init__(name or f"AlphaBeta (Depth {depth})", depth, player_id)
        if not isinstance(q_depth, int) or q_depth < 0:
            raise ValueError("q_depth must be a nonnegative integer")
        self.use_quiescence = use_quiescence
        self.q_depth = q_depth
        self.cutoffs = 0

    def _order_moves(self, position, moves):
        def score(move):
            return ((10000 + capture_score(position, move) if position.is_capture(move) else 0)
                    + (PIECE_VALUES[move.promotion] if move.promotion else 0))
        return sorted(moves, key=score, reverse=True)

    def _quiescence_search(self, position, alpha, beta, q_depth=None, ply=0):
        self.nodes_visited += 1
        terminal = terminal_score(position, self.claim_draw, ply)
        if terminal is not None:
            return terminal
        remaining = self.q_depth if q_depth is None else q_depth
        in_check = position.is_check()
        # Never stand pat in check: all legal evasions must be examined.
        if not in_check:
            stand_pat = evaluate_relative(position, check_terminal=False)
            if remaining <= 0:
                return stand_pat
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
        moves = [m for m in position.legal_moves if in_check or position.is_capture(m) or m.promotion]
        for move in self._order_moves(position, moves):
            position.push(move)
            try:
                if remaining <= 0:
                    # At the safety cap still make one real evasion; this is a
                    # bounded approximation, not a claim of exhaustive tactics.
                    child = terminal_score(position, self.claim_draw, ply + 1)
                    score = -(child if child is not None else evaluate_relative(position, check_terminal=False))
                else:
                    score = -self._quiescence_search(position, -beta, -alpha, remaining - 1, ply + 1)
            finally:
                position.pop()
            if score >= beta:
                self.cutoffs += 1
                return score
            alpha = max(alpha, score)
        return alpha

    def _search(self, position, depth, alpha=-INF, beta=INF, ply=0):
        self.nodes_visited += 1
        terminal = terminal_score(position, self.claim_draw, ply)
        if terminal is not None:
            return terminal
        if depth == 0:
            return (self._quiescence_search(position, alpha, beta, ply=ply)
                    if self.use_quiescence else evaluate_relative(position, check_terminal=False))
        for move in self._order_moves(position, list(position.legal_moves)):
            position.push(move)
            try:
                score = -self._search(position, depth - 1, -beta, -alpha, ply + 1)
            finally:
                position.pop()
            if score >= beta:
                self.cutoffs += 1
                return score
            alpha = max(alpha, score)
        return alpha

    def act(self, board):
        moves = board.get_legal_actions()
        if not moves:
            raise ValueError("No actions: terminal position")
        self.nodes_visited = self.cutoffs = 0
        self.claim_draw = board.claim_draw
        position = board.board.copy(stack=True)
        alpha, chosen = -INF, moves[0]
        for move in self._order_moves(position, moves):
            position.push(move)
            try:
                score = -self._search(position, self.depth - 1, -INF, -alpha, 1)
            finally:
                position.pop()
            if score > alpha:
                alpha, chosen = score, move
        self.last_score = alpha
        return chosen
