"""
Kiểm tra tích hợp CLI và Factory cho AlphaGoAgent (Stage 6).
"""

import unittest
import chess

from agents import build_agent, AGENT_REGISTRY, CLASSIC_AGENTS
from agents.stage6_alphago import AlphaGoAgent
from games.chess.board import ChessBoard


class TestAlphaGoCLIIntegration(unittest.TestCase):
    def test_alphago_in_registry_and_factory(self):
        """AlphaGo có mặt trong AGENT_REGISTRY và CLASSIC_AGENTS."""
        self.assertIn("alphago", AGENT_REGISTRY)
        self.assertIn("alphago", CLASSIC_AGENTS)
        self.assertIs(AGENT_REGISTRY["alphago"], AlphaGoAgent)

    def test_build_agent_alphago_depth_mapping(self):
        """build_agent('alphago', depth=2) ánh xạ đúng simulations và seed."""
        agent = build_agent("alphago", depth=2, seed=42)
        self.assertIsInstance(agent, AlphaGoAgent)
        self.assertEqual(agent.seed, 42)
        self.assertEqual(agent.simulations, 30)  # max(10, 2 * 15) = 30

    def test_alphago_plays_move_from_factory(self):
        """Agent tạo từ build_agent có thể đi nước cờ hợp lệ trên bàn cờ chuẩn."""
        agent = build_agent("alphago", depth=1, seed=42)
        board = ChessBoard()
        move = agent.act(board)
        self.assertIn(move, board.get_legal_actions())


if __name__ == "__main__":
    unittest.main()

