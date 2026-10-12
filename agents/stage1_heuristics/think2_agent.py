"""Mate defense, then MVV-LVA (an ordering heuristic, not exchange analysis)."""
import chess
from .think1_agent import Think1Agent
from .pst import PIECE_VALUES

def capture_score(position, move):
    if not position.is_capture(move):
        return 0
    victim = chess.PAWN if position.is_en_passant(move) else position.piece_type_at(move.to_square)
    attacker = position.piece_type_at(move.from_square)
    # King cannot be captured. Cap its attacker penalty below 10 * pawn
    # so every legal capture ranks above a quiet move.
    return 10 * PIECE_VALUES[victim] - (900 if attacker == chess.KING else PIECE_VALUES[attacker])

class Think2Agent(Think1Agent):
    def __init__(self, name="Think2", player_id=None, seed=None):
        super().__init__(name, player_id, seed)

    def act(self, board):
        moves = board.get_legal_actions()
        if not moves:
            raise ValueError("No actions: terminal position")
        position = board.board.copy()
        mate = self.immediate_mate(position, moves)
        if mate:
            return mate
        safe = []
        for move in moves:
            position.push(move)
            try:
                if position.is_game_over(claim_draw=board.claim_draw) or self.immediate_mate(position, list(position.legal_moves)) is None:
                    safe.append(move)
            finally:
                position.pop()
        candidates = safe or moves
        scores = {m: capture_score(position, m) + (PIECE_VALUES[m.promotion] if m.promotion else 0) for m in candidates}
        best = max(scores.values())
        return self.rng.choice([m for m in candidates if scores[m] == best])
