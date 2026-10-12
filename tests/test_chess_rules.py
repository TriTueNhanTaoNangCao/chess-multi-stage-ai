import unittest
import chess
from games.chess.board import ChessBoard

class ChessRulesTests(unittest.TestCase):
    def test_initial_and_illegal_move_are_atomic(self):
        board = ChessBoard()
        self.assertEqual(len(board.get_legal_actions()), 20)
        before = board.fen()
        for move in [chess.Move.from_uci("e2e5"), chess.Move.null(), "e9e4", 7]:
            with self.assertRaises((ValueError, TypeError)):
                board.step(move)
            self.assertEqual(board.fen(), before)

    def test_castling_and_undo(self):
        board = ChessBoard("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        board.step("O-O")
        self.assertEqual(board.piece_at(chess.G1), chess.Piece(chess.KING, True))
        self.assertEqual(board.piece_at(chess.F1), chess.Piece(chess.ROOK, True))
        board.pop()
        self.assertTrue(board.board.has_kingside_castling_rights(True))

    def test_cannot_castle_through_check(self):
        board = ChessBoard("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1")
        self.assertNotIn(chess.Move.from_uci("e1g1"), board.legal_moves)

    def test_en_passant(self):
        board = ChessBoard()
        for move in ["e4","a6","e5","d5"]:
            board.step(move)
        board.step("exd6")
        self.assertIsNone(board.piece_at(chess.D5))
        self.assertEqual(board.piece_at(chess.D6), chess.Piece(chess.PAWN, True))

    def test_pinned_en_passant_illegal(self):
        board = ChessBoard("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        self.assertNotIn(chess.Move.from_uci("e5d6"), board.legal_moves)

    def test_all_promotions(self):
        board = ChessBoard("7k/P7/8/8/8/8/8/7K w - - 0 1")
        promotions = [m for m in board.legal_moves if m.from_square == chess.A7]
        self.assertEqual({m.promotion for m in promotions}, {2,3,4,5})
        for move in promotions:
            clone = board.clone()
            clone.step(move)
            self.assertEqual(clone.piece_at(chess.A8).piece_type, move.promotion)

    def test_checkmate(self):
        board = ChessBoard()
        for move in ["f3", "e5", "g4", "Qh4#"]:
            board.step(move)
        self.assertTrue(board.is_terminal())
        self.assertEqual(board.check_winner(), -1)
        with self.assertRaises(ValueError):
            board.step("a3")

    def test_stalemate_and_insufficient(self):
        for fen in ["7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "7k/8/8/8/8/8/8/K7 w - - 0 1"]:
            board = ChessBoard(fen)
            self.assertEqual(board.check_winner(), 0)
            self.assertEqual(board.get_legal_actions(), [])

    def test_fifty_and_seventyfive_moves(self):
        fen = "7k/8/8/8/8/8/8/KR6 w - - 100 51"
        self.assertTrue(ChessBoard(fen).is_terminal())
        self.assertFalse(ChessBoard(fen, claim_draw=False).is_terminal())
        self.assertTrue(ChessBoard(fen.replace("100 51", "150 76"), claim_draw=False).is_terminal())

    def test_repetition_clone_retains_history(self):
        board = ChessBoard(claim_draw=False)
        for move in ["Nf3","Nf6","Ng1","Ng8"]*2:
            board.step(move)
        clone = board.clone()
        self.assertTrue(clone.board.can_claim_threefold_repetition())
        self.assertFalse(clone.is_terminal())
        clone.claim_draw = True
        self.assertTrue(clone.is_terminal())
        clone.pop()
        self.assertEqual(len(board.board.move_stack), 8)
        for move in ["Nf3","Nf6","Ng1","Ng8"]*2:
            board.step(move)
        self.assertTrue(board.is_terminal())  # automatic fivefold

    def test_invalid_fen(self):
        with self.assertRaises(ValueError):
            ChessBoard("8/8/8/8/8/8/8/8 w - - 0 1")
