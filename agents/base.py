"""
Abstract Base Agent definition for all chess agents.
"""

from abc import ABC, abstractmethod
from typing import Optional
import chess
from games.chess.board import ChessBoard


class BaseAgent(ABC):
    """
    Abstract base class for all Chess AI and Human agents.
    Every agent must implement the `act` method.
    """

    def __init__(self, name: str = "BaseAgent", player_id: Optional[int] = None):
        self.name = name
        self.player_id = player_id  # 1 for White, -1 for Black, or None

    @abstractmethod
    def act(self, board: ChessBoard) -> chess.Move:
        """
        Decide and return the chosen chess.Move for the current position.
        The move must be legal according to board.get_legal_actions().
        Never mutate the caller's position/history. Raise ValueError at terminal
        positions. Use board.turn, not player_id, to decide which side to play.
        """
        pass

    def reset(self) -> None:
        """Reset internal search state, caches, or history between games."""
        pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
