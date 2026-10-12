"""Seeded baseline without changing global random state."""
import random
from agents.base import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, name="Random", player_id=None, seed=None):
        super().__init__(name, player_id)
        self.rng = random.Random(seed)

    def act(self, board):
        moves = board.get_legal_actions()
        if not moves:
            raise ValueError("No actions: terminal position")
        return self.rng.choice(moves)
