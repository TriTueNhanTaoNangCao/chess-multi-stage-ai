"""Thi: implement Selection UCB1 -> Expansion -> Rollout -> Backprop."""
from agents.base import BaseAgent

class PureMCTSAgent(BaseAgent):
    def __init__(self, simulations=500):
        super().__init__(f"PureMCTS ({simulations})")
        self.simulations = simulations

    def act(self, board):
        # Clone with history; negate values at every alternating-player edge.
        raise NotImplementedError("Stage 3 belongs to Thi; implement MCTS before registry export")
