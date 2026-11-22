"""
Unit tests cho AlphaGoAgent (Stage 6).
Kiểm tra:
1. Khởi tạo agent với các cấu hình mạng và tham số khác nhau.
2. act() trả về nước cờ hợp lệ, chơi được cả 2 màu Trắng/Đen.
3. Ném ValueError trên thế cờ kết thúc.
4. Tương thích với checkpoint thực tế/dummy theo chuẩn HANDOFF.md.
5. Bảo toàn bàn cờ gốc và reset trạng thái an toàn.
"""

import io
import unittest
import chess

try:
    import torch
    from models.cnn_policy import CNNPolicyNetwork
    from models.value_net import ValueNetwork
except ImportError:
    torch = None
    CNNPolicyNetwork = None
    ValueNetwork = None

from games.chess.board import ChessBoard
from agents.stage6_alphago.alphago_agent import AlphaGoAgent


class TestAlphaGoAgent(unittest.TestCase):
    def setUp(self):
        self.board = ChessBoard()

    def test_agent_initialization_default(self):
        """Khởi tạo AlphaGoAgent mặc định thành công."""
        agent = AlphaGoAgent(simulations=10, seed=42)
        self.assertEqual(agent.simulations, 10)
        self.assertEqual(agent.seed, 42)
        self.assertIn("AlphaGo", agent.name)

    def test_agent_act_returns_legal_move_white(self):
        """act() khi cầm quân Trắng trả về nước cờ hợp lệ và không làm biến đổi bàn cờ."""
        agent = AlphaGoAgent(simulations=15, seed=42)
        initial_fen = self.board.fen()

        move = agent.act(self.board)
        self.assertIsInstance(move, chess.Move)
        self.assertIn(move, self.board.get_legal_actions())
        self.assertEqual(self.board.fen(), initial_fen)

    def test_agent_act_black(self):
        """act() khi cầm quân Đen sau khi Trắng đã đi 1.e4."""
        self.board.step("e4")
        self.assertEqual(self.board.turn, chess.BLACK)

        agent = AlphaGoAgent(simulations=15, seed=42)
        move = agent.act(self.board)
        self.assertIsInstance(move, chess.Move)
        self.assertIn(move, self.board.get_legal_actions())

    def test_agent_act_terminal_raises_error(self):
        """Ném ValueError khi gọi act() trên thế cờ đã kết thúc."""
        # Scholar's mate
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        board = ChessBoard(fen)
        self.assertTrue(board.is_terminal())

        agent = AlphaGoAgent(simulations=10)
        with self.assertRaises(ValueError):
            agent.act(board)

    @unittest.skipIf(torch is None, "Cần có PyTorch để kiểm tra nạp checkpoint")
    def test_agent_load_dummy_checkpoints(self):
        """Kiểm tra nạp checkpoint Policy và Value theo chuẩn HANDOFF.md."""
        policy_model = CNNPolicyNetwork()
        value_model = ValueNetwork()

        # Tạo dummy payload trong bộ nhớ buffer
        policy_payload = {
            "format_version": 1,
            "stage": 4,
            "model_class": "CNNPolicyNetwork",
            "model_config": policy_model.get_config(),
            "encoder_version": "hwc14-white-v1",
            "action_version": "az73-rankflip-v1",
            "num_actions": 4672,
            "state_dict": policy_model.state_dict(),
        }
        value_payload = {
            "format_version": 1,
            "stage": 5,
            "model_class": "ValueNetwork",
            "model_config": value_model.get_config(),
            "encoder_version": "hwc14-white-v1",
            "state_dict": value_model.state_dict(),
        }

        # Lưu tạm ra file để kiểm tra hàm load
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            p_path = Path(tmp_dir) / "policy.pt"
            v_path = Path(tmp_dir) / "value.pt"
            torch.save(policy_payload, p_path)
            torch.save(value_payload, v_path)

            agent = AlphaGoAgent(
                checkpoint_policy=p_path,
                checkpoint_value=v_path,
                simulations=15,
                seed=42,
            )
            move = agent.act(self.board)
            self.assertIn(move, self.board.get_legal_actions())

    def test_agent_reset(self):
        """Gọi reset() không ném ngoại lệ và làm mới bộ sinh số ngẫu nhiên."""
        agent = AlphaGoAgent(simulations=10, seed=42)
        agent.act(self.board)
        agent.reset()
        self.assertIsNone(agent.mcts.root)


if __name__ == "__main__":
    unittest.main()

