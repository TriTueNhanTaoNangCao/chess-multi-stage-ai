"""
Unit tests cho động cơ tìm kiếm AlphaGoMCTS (Stage 6).
Kiểm tra:
1. search() trả về nước đi hợp lệ và bảo toàn bàn cờ gốc.
2. Xử lý đường tắt khi chỉ có 1 nước đi duy nhất.
3. Ném lỗi khi tìm kiếm trên bàn cờ đã kết thúc (terminal).
4. Tìm thấy nước chiếu hết ngay (mate-in-one).
5. Các chế độ hybrid leaf evaluation (rollout_weight = 0.0, 0.5, 1.0).
6. get_action_probs trả về phân phối xác suất hợp lệ tổng bằng 1.0.
7. Tính tất định khi dùng cùng seed.
"""

import unittest
import chess

from games.chess.board import ChessBoard
from agents.stage6_alphago.mcts import AlphaGoMCTS


def uniform_policy(board: ChessBoard):
    """Hàm policy giả lập phân phối đều cho các nước hợp lệ."""
    actions = board.get_legal_actions()
    n = len(actions)
    if n == 0:
        return {}
    return {m: 1.0 / n for m in actions}


def zero_value(board: ChessBoard):
    """Hàm value giả lập luôn trả về 0.0 (cân bằng)."""
    return 0.0


class TestAlphaGoMCTS(unittest.TestCase):
    def setUp(self):
        self.board = ChessBoard()

    def test_search_returns_legal_move_and_preserves_board(self):
        """search() trả về nước đi hợp lệ và không làm thay đổi trạng thái hay move_stack của board."""
        initial_fen = self.board.fen()
        mcts = AlphaGoMCTS(
            policy_fn=uniform_policy,
            value_fn=zero_value,
            simulations=20,
            seed=42,
        )
        move = mcts.search(self.board)

        self.assertIsInstance(move, chess.Move)
        self.assertIn(move, self.board.get_legal_actions())
        # Bàn cờ gốc không bị biến đổi
        self.assertEqual(self.board.fen(), initial_fen)

    def test_search_single_legal_move_fast_path(self):
        """Khi chỉ có 1 nước đi duy nhất, đi ngay lập tức không cần mô phỏng."""
        # Vua Đen ở a8 bị kẹp, chỉ có duy nhất nước a8b8
        fen = "k7/8/K1P5/8/8/8/8/8 b - - 0 1"
        board = ChessBoard(fen)
        self.assertEqual(len(board.get_legal_actions()), 1)
        only_move = board.get_legal_actions()[0]

        mcts = AlphaGoMCTS(policy_fn=uniform_policy, simulations=20)
        chosen = mcts.search(board)
        self.assertEqual(chosen, only_move)

    def test_search_terminal_raises_error(self):
        """Không thể tìm kiếm trên bàn cờ đã kết thúc."""
        # Bàn cờ Scholar's Mate đã chiếu hết
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        board = ChessBoard(fen)
        self.assertTrue(board.is_terminal())

        mcts = AlphaGoMCTS(policy_fn=uniform_policy, simulations=10)
        with self.assertRaises(ValueError):
            mcts.search(board)

    def test_search_mate_in_one(self):
        """
        Thế cờ Mate-in-one: Trắng có nước Qh7# chiếu hết ngay.
        Policy ưu tiên nhẹ cho Qh7 và MCTS phải tìm thấy và chọn nước này.
        """
        # Thế cờ Trắng đi Hậu h5 sang h7 chiếu hết
        fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 5"
        board = ChessBoard(fen)
        mate_move = chess.Move.from_uci("h5f7")  # Qxf7#
        self.assertIn(mate_move, board.get_legal_actions())

        def biased_policy(b: ChessBoard):
            actions = b.get_legal_actions()
            probs = {}
            for m in actions:
                probs[m] = 0.5 if m == mate_move else 0.5 / (len(actions) - 1)
            return probs

        mcts = AlphaGoMCTS(
            policy_fn=biased_policy,
            value_fn=zero_value,
            simulations=30,
            seed=42,
        )
        best = mcts.search(board)
        self.assertEqual(best, mate_move)

    def test_hybrid_evaluation_modes(self):
        """Kiểm tra hoạt động trơn tru ở cả 3 chế độ: pure value, pure rollout, và hybrid."""
        for rollout_weight in [0.0, 0.5, 1.0]:
            mcts = AlphaGoMCTS(
                policy_fn=uniform_policy,
                value_fn=zero_value,
                simulations=10,
                rollout_weight=rollout_weight,
                max_rollout_depth=5,
                seed=42,
            )
            move = mcts.search(self.board)
            self.assertIn(move, self.board.get_legal_actions())

    def test_action_probs_distribution(self):
        """get_action_probs trả về dictionary xác suất có tổng bằng 1.0."""
        mcts = AlphaGoMCTS(
            policy_fn=uniform_policy,
            value_fn=zero_value,
            simulations=20,
            seed=42,
        )
        probs = mcts.get_action_probs(self.board, temperature=1.0)
        self.assertEqual(len(probs), len(self.board.get_legal_actions()))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_reproducibility_with_seed(self):
        """Hai động cơ MCTS cùng seed phải trả về chính xác cùng một nước đi."""
        mcts1 = AlphaGoMCTS(policy_fn=uniform_policy, value_fn=zero_value, simulations=25, seed=123)
        mcts2 = AlphaGoMCTS(policy_fn=uniform_policy, value_fn=zero_value, simulations=25, seed=123)

        move1 = mcts1.search(self.board)
        move2 = mcts2.search(self.board)
        self.assertEqual(move1, move2)


if __name__ == "__main__":
    unittest.main()

