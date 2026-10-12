"""Thi: node state for UCB1 MCTS; values are from the player-to-move perspective."""
from dataclasses import dataclass, field
import chess

@dataclass
class MCTSNode:
    board: object
    parent: object = None
    move: chess.Move | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)

    @property
    def mean_value(self):
        return self.value_sum/self.visits if self.visits else 0.0
