"""
Kiểm thử tích hợp Đấu trường Arena và xác thực tính hợp lệ tuyệt đối của AlphaGo (Stage 6).
Kiểm tra:
1. Thi đấu cân bằng màu qua arena.evaluator.evaluate (không bị forfeit/lỗi cờ).
2. Tái tạo trạng thái ban đầu bằng restore_initial_board.
3. Ứng phó chính xác trong các thế cờ chiến thuật đặc biệt (en-passant, phong cấp, chiếu đôi).
"""

import unittest
import chess

from arena.evaluator import evaluate, play_game, restore_initial_board
from agents import build_agent
from agents.random_agent import RandomAgent
from agents.stage1_heuristics.think1_agent import Think1Agent
from agents.stage6_alphago import AlphaGoAgent
from games.chess.board import ChessBoard


class TestAlphaGoArenaIntegration(unittest.TestCase):
    def test_alphago_vs_random_balanced_match(self):
        """AlphaGo đấu 2 ván cân bằng màu với RandomAgent trong Arena."""
        alphago = AlphaGoAgent(simulations=15, seed=42, name="AlphaGo")
        random_bot = RandomAgent(seed=42)

        result = evaluate(alphago, random_bot, games=2, max_plies=50)

        self.assertEqual(result["games"], 2)
        self.assertIn("wins", result)
        self.assertIn("score_rate", result)

        # Kiểm tra không có ván nào bị xử thua do lỗi bot (forfeit)
        for record in result["records"]:
            self.assertEqual(record["error"], "", f"Phát hiện lỗi trong ván đấu: {record['error']}")
            self.assertNotIn("forfeit", record["reason"], f"Bot bị xử thua phạm luật: {record['reason']}")

            # Kiểm tra khả năng khôi phục FEN và lịch sử nước đi từ log đấu trường
            restored = restore_initial_board(record)
            self.assertEqual(restored.fen(), record["initial_fen"])

    def test_alphago_handles_special_chess_tactics(self):
        """AlphaGo xử lý chính xác các tình huống cờ đặc biệt mà không bao giờ ném ngoại lệ."""
        alphago = AlphaGoAgent(simulations=20, seed=42)

        # 1. Thế cờ phong cấp tốt (Pawn Promotion)
        promo_fen = "8/4P3/8/8/8/8/8/k1K5 w - - 0 1"
        board_promo = ChessBoard(promo_fen)
        move_promo = alphago.act(board_promo)
        self.assertIn(move_promo, board_promo.get_legal_actions())

        # 2. Thế cờ ăn tốt qua đường (En-Passant)
        ep_fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        board_ep = ChessBoard(ep_fen)
        move_ep = alphago.act(board_ep)
        self.assertIn(move_ep, board_ep.get_legal_actions())

        # 3. Thế cờ bị chiếu đôi (Double Check) - chỉ có nước chạy Vua
        double_check_fen = "rnb1k2r/pppp1ppp/8/8/1b1Pn3/8/PPPB1PPP/RN1QKB1R w KQkq - 3 7"
        board_dc = ChessBoard(double_check_fen)
        move_dc = alphago.act(board_dc)
        self.assertIn(move_dc, board_dc.get_legal_actions())

    def test_alphago_reset_consistency_between_rounds(self):
        """Gọi reset() giữa nhiều vòng đấu không làm suy giảm tính toàn vẹn của MCTS."""
        alphago = AlphaGoAgent(simulations=10, seed=42)
        board = ChessBoard()

        for _ in range(3):
            alphago.reset()
            move = alphago.act(board)
            self.assertIn(move, board.get_legal_actions())


if __name__ == "__main__":
    unittest.main()

