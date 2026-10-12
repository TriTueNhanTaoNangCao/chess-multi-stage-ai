"""One-ply immediate mate scanner."""
from agents.random_agent import RandomAgent

class Think1Agent(RandomAgent):
    def __init__(self, name="Think1", player_id=None, seed=None):
        super().__init__(name, player_id, seed)

    @staticmethod
    def immediate_mate(position, moves):
        for move in moves:
            position.push(move)
            try:
                if position.is_checkmate():
                    return move
            finally:
                position.pop()
        return None

    def act(self, board):
        moves = board.get_legal_actions()
        if not moves:
            raise ValueError("No actions: terminal position")
        return self.immediate_mate(board.board.copy(), moves) or self.rng.choice(moves)
