"""CLI SAN/UCI input. GUI obtains moves through mouse events."""
from agents.base import BaseAgent

class HumanAgent(BaseAgent):
    def __init__(self, name="Human", player_id=None, input_fn=input):
        super().__init__(name, player_id)
        self.input_fn = input_fn

    def act(self, board):
        if board.is_terminal():
            raise ValueError("No actions: terminal position")
        while True:
            text = self.input_fn("SAN/UCI (q: quit): ").strip()
            if text.lower() in {"q", "quit", "exit"}:
                raise EOFError
            try:
                move = board.parse_san(text)
                if not board.is_legal(move):
                    raise ValueError("Not a legal action")
                return move
            except ValueError:
                print("Illegal move. Example: e4, Nf3, e2e4.")
