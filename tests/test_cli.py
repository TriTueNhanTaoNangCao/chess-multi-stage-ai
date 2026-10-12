import subprocess
import sys
import unittest
from agents.human_agent import HumanAgent
from games.chess.board import ChessBoard
from games.chess.renderer import ChessRenderer

class CLITests(unittest.TestCase):
    def test_human_san_and_uci(self):
        for text in ["e4","e2e4"]:
            self.assertEqual(HumanAgent(input_fn=lambda _:text).act(ChessBoard()).uci(),"e2e4")

    def test_eof_does_not_play_random_move(self):
        def eof(_):
            raise EOFError
        with self.assertRaises(EOFError):
            HumanAgent(input_fn=eof).act(ChessBoard())

    def test_ascii_renderer(self):
        text = ChessRenderer.render(ChessBoard().board,ascii_only=True,color=False)
        self.assertTrue(text.isascii())
        self.assertNotIn("\x1b",text)

    def test_cli_subprocess(self):
        result = subprocess.run([sys.executable,"main.py","--cli","--ascii","--agent","random"],
                                input="e4\nq\n",text=True,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("Random:",result.stdout)
        self.assertIn("Goodbye",result.stdout)
