"""Game-independent contract."""
from abc import ABC, abstractmethod

class BaseBoard(ABC):
    @abstractmethod
    def get_legal_actions(self):
        """Available actions; empty when terminal."""
    @abstractmethod
    def step(self, move):
        """Apply a legal action or raise ValueError."""
    @abstractmethod
    def is_terminal(self):
        """Whether play has ended."""
    @abstractmethod
    def check_winner(self):
        """White +1, Black -1, draw 0, ongoing None."""
    @abstractmethod
    def clone(self):
        """Independent copy including history."""
