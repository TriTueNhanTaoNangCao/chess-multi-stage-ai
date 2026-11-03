"""
Unit tests cho AlphaGoNode (Stage 6).
Kiểm tra:
1. Khởi tạo node và bảo toàn move stack của ChessBoard.
2. expand(policy_probs) tạo các node con với đúng prior.
3. selection_score tính đúng theo góc nhìn -Q và exploration ưu tiên prior.
4. best_child chọn chính xác node con có điểm cao nhất.
5. update() tích lũy đúng visits và value_sum.
"""

import math
import random
import unittest
import chess

from games.chess.board import ChessBoard
from agents.stage6_alphago.node import AlphaGoNode


class TestAlphaGoNode(unittest.TestCase):
    def setUp(self):
        self.board = ChessBoard()

    def test_node_initialization(self):
        """Khởi tạo node với giá trị mặc định."""
        node = AlphaGoNode(board=self.board, prior=0.25)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(node.mean_value, 0.0)
        self.assertEqual(node.prior, 0.25)
        self.assertFalse(node.is_expanded)
        self.assertEqual(len(node.children), 0)

    def test_expand_with_priors(self):
        """expand() tạo đầy đủ node con với đúng prior và không làm thay đổi board gốc."""
        node = AlphaGoNode(board=self.board)
        e2e4 = chess.Move.from_uci("e2e4")
        d2d4 = chess.Move.from_uci("d2d4")

        policy_probs = {e2e4: 0.6, d2d4: 0.4}
        node.expand(policy_probs)

        self.assertTrue(node.is_expanded)
        self.assertEqual(len(node.children), 2)
        self.assertIn(e2e4, node.children)
        self.assertIn(d2d4, node.children)

        # Kiểm tra node con
        child_e4 = node.children[e2e4]
        self.assertEqual(child_e4.prior, 0.6)
        self.assertEqual(child_e4.move, e2e4)
        self.assertEqual(child_e4.parent, node)
        # Nước e2e4 đã đi ở con, lượt hiện tại ở con là Đen
        self.assertEqual(child_e4.board.turn, chess.BLACK)

        # Bàn cờ ở node cha vẫn là Trắng đi
        self.assertEqual(node.board.turn, chess.WHITE)

    def test_selection_score_prior_preference_when_unvisited(self):
        """Khi các node con chưa được thăm (N=0), node có prior cao hơn phải có điểm cao hơn."""
        node = AlphaGoNode(board=self.board)
        e2e4 = chess.Move.from_uci("e2e4")
        d2d4 = chess.Move.from_uci("d2d4")
        node.expand({e2e4: 0.7, d2d4: 0.3})

        child_e4 = node.children[e2e4]
        child_d4 = node.children[d2d4]

        score_e4 = child_e4.selection_score(parent_visits=0, c_puct=1.5)
        score_d4 = child_d4.selection_score(parent_visits=0, c_puct=1.5)

        self.assertGreater(score_e4, score_d4)
        self.assertEqual(node.best_child(c_puct=1.5), child_e4)

    def test_selection_score_value_sign_inversion(self):
        """
        Quy ước dấu Value:
        mean_value của con lưu theo góc nhìn của bên đi ở con (Đen).
        Nếu mean_value của con là -1.0 (Đen đang thua đậm),
        thì từ góc nhìn của cha (Trắng vừa đi nước dẫn đến đây), điểm khai thác phải là -(-1.0) = +1.0.
        """
        node = AlphaGoNode(board=self.board)
        e2e4 = chess.Move.from_uci("e2e4")
        node.expand({e2e4: 0.5})
        child = node.children[e2e4]

        # Giả sử mô phỏng xong, Đen tại child thua (-1.0)
        child.update(-1.0)
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.mean_value, -1.0)

        # Điểm nhìn từ cha
        score = child.selection_score(parent_visits=1, c_puct=0.0)  # c_puct=0 để chỉ xét exploitation
        self.assertAlmostEqual(score, 1.0)

    def test_best_child_exploration_vs_exploitation(self):
        """Node có prior thấp hơn nhưng điểm Q rất cao có thể vượt qua node có prior cao."""
        node = AlphaGoNode(board=self.board)
        e2e4 = chess.Move.from_uci("e2e4")
        d2d4 = chess.Move.from_uci("d2d4")
        node.expand({e2e4: 0.8, d2d4: 0.2})

        child_e4 = node.children[e2e4]
        child_d4 = node.children[d2d4]

        # e4 bị thăm nhiều nhưng kết quả xấu cho cha (con thắng: mean_value = +1.0 -> nhìn từ cha là -1.0)
        for _ in range(10):
            child_e4.update(1.0)
            node.visits += 1

        # d4 thăm 2 lần nhưng kết quả tốt cho cha (con thua: mean_value = -1.0 -> nhìn từ cha là +1.0)
        for _ in range(2):
            child_d4.update(-1.0)
            node.visits += 1

        # Dù e4 có prior 0.8, d4 với Q=+1.0 sẽ được chọn
        chosen = node.best_child(c_puct=1.0)
        self.assertEqual(chosen, child_d4)


if __name__ == "__main__":
    unittest.main()

