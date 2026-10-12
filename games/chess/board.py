"""Standard chess with validated moves and an explicit draw-claim policy."""
import chess
from games.base import BaseBoard

class ChessBoard(BaseBoard):
    def __init__(self, fen=None, *, claim_draw=True):
        self.board = chess.Board(fen) if fen is not None else chess.Board()
        if not self.board.is_valid():
            raise ValueError("Invalid standard chess position")
        self.claim_draw = claim_draw
        self._initial_fen = self.board.fen(en_passant="fen")

    def reset(self):
        self.board.set_fen(self._initial_fen)

    def get_legal_actions(self):
        return [] if self.is_terminal() else list(self.board.legal_moves)

    @property
    def legal_moves(self):
        return self.get_legal_actions()

    def step(self, move):
        if self.is_terminal():
            raise ValueError("Game has ended")
        if isinstance(move, str):
            move = self.board.parse_san(move)
        if not isinstance(move, chess.Move) or not self.board.is_legal(move):
            raise ValueError(f"Illegal move: {move}")
        self.board.push(move)

    push = step

    def pop(self):
        return self.board.pop()

    def outcome(self):
        return self.board.outcome(claim_draw=self.claim_draw)

    def is_terminal(self):
        return self.outcome() is not None

    is_game_over = is_terminal

    def check_winner(self):
        result = self.outcome()
        if result is None:
            return None
        return 0 if result.winner is None else (1 if result.winner else -1)

    get_result = check_winner

    def get_result_relative(self):
        """Value for side to move: checkmated side gets -1; negate at parent."""
        result = self.check_winner()
        return None if result is None else result * self.current_player

    def clone(self):
        result = ChessBoard(claim_draw=self.claim_draw)
        result.board = self.board.copy(stack=True)
        result._initial_fen = self._initial_fen
        return result

    copy = clone

    @property
    def turn(self):
        return self.board.turn

    @property
    def current_player(self):
        return 1 if self.turn else -1

    @property
    def turn_name(self):
        return "White" if self.turn else "Black"

    def fen(self):
        return self.board.fen()

    def is_legal(self, move):
        return isinstance(move, chess.Move) and not self.is_terminal() and self.board.is_legal(move)

    def is_check(self):
        return self.board.is_check()

    def is_checkmate(self):
        return self.board.is_checkmate()

    def is_stalemate(self):
        return self.board.is_stalemate()

    def is_draw(self):
        return self.check_winner() == 0

    def san(self, move):
        return self.board.san(move)

    def parse_san(self, text):
        return self.board.parse_san(text)

    def parse_uci(self, text):
        return self.board.parse_uci(text)

    def piece_at(self, square):
        return self.board.piece_at(square)

    def __str__(self):
        return str(self.board)
