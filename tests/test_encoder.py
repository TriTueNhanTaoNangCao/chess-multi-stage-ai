import random
import unittest
import chess
import numpy as np
from games.chess.encoder import ChessEncoder as E, NUM_ACTIONS

class EncoderTests(unittest.TestCase):
    def test_shape_and_complete_state_roundtrip(self):
        board = chess.Board()
        for move in ["Nf3","Nf6","Ng1","Ng8"]*2:
            board.push_san(move)
        self.assertEqual(E.encode_board(board).shape,(8,8,14))
        restored = E.decode_state(E.encode_state(board))
        self.assertEqual(restored.fen(),board.fen())
        self.assertEqual(restored.move_stack,board.move_stack)
        self.assertTrue(restored.can_claim_threefold_repetition())

    def test_metadata_mismatch_rejected(self):
        state = E.encode_state(chess.Board())
        state["planes"][0,0,0] = 1
        with self.assertRaises(ValueError):
            E.decode_state(state)

    def test_actions_roundtrip_across_random_positions(self):
        rng = random.Random(12)
        board = chess.Board()
        for _ in range(100):
            moves = list(board.legal_moves)
            indices = E.legal_action_indices(board)
            self.assertEqual(len(indices),len(set(indices)))
            mask = E.get_legal_action_mask(board)
            self.assertEqual(mask.shape,(NUM_ACTIONS,))
            self.assertEqual(int(mask.sum()),len(moves))
            for move in moves:
                self.assertEqual(E.decode_move(E.encode_move(move,board.turn),board),move)
            if board.is_game_over():
                board.reset()
            else:
                board.push(rng.choice(moves))

    def test_special_actions(self):
        fens = ["r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "7k/P7/8/8/8/8/8/7K w - - 0 1",
                "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1"]
        for fen in fens:
            board = chess.Board(fen)
            for raw in [board,board.mirror()]:
                for move in raw.legal_moves:
                    self.assertEqual(E.decode_move(E.encode_move(move,raw.turn),raw),move)
        self.assertIsNone(E.decode_move(4672,chess.Board()))
