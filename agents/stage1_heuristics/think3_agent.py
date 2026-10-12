"""One-ply positional evaluation: material, PST, development and king safety."""
from .think1_agent import Think1Agent
from .pst import evaluate_relative

class Think3Agent(Think1Agent):
    def __init__(self, name="Think3", player_id=None, seed=None):
        super().__init__(name, player_id, seed)

    def act(self, board):
        moves = board.get_legal_actions()
        if not moves:
            raise ValueError("No actions: terminal position")
        position = board.board.copy()
        scores = {}
        for move in moves:
            position.push(move)
            try:
                scores[move] = -evaluate_relative(position, claim_draw=board.claim_draw)
            finally:
                position.pop()
        best = max(scores.values())
        return self.rng.choice([m for m in moves if scores[m] == best])
