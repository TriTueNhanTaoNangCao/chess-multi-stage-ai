import json
import tempfile
import unittest
from pathlib import Path
import chess
from agents import RandomAgent, Think1Agent
from arena.evaluator import play_game,evaluate
from arena.tournament import update_elo,expected_score,run_tournament
from games.chess.board import ChessBoard

class BrokenAgent(RandomAgent):
    def act(self,board):
        return chess.Move.from_uci("e2e5")

class CrashingAgent(RandomAgent):
    def act(self,board):
        raise RuntimeError("test error")

class ArenaTests(unittest.TestCase):
    def test_elo_arithmetic(self):
        self.assertEqual(expected_score(1200,1200),0.5)
        self.assertEqual(update_elo(1200,1200,1),(1216,1184))
        self.assertEqual(update_elo(1200,1200,0.5),(1200,1200))
        a,b = update_elo(1500,1100,0)
        self.assertAlmostEqual(a+b,2600)
        self.assertGreater(1500-a,16)

    def test_forfeit_and_exception(self):
        for agent in [BrokenAgent("bad"),CrashingAgent("bad")]:
            result = play_game(agent,RandomAgent("ok"))
            self.assertEqual(result.result,"0-1")
            self.assertEqual(result.reason,"forfeit")
            self.assertTrue(result.error)

    def test_mate_result_and_input_unchanged(self):
        board = ChessBoard("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = play_game(Think1Agent(),RandomAgent(),board=board,max_plies=2)
        self.assertEqual(result.result,"1-0")
        self.assertEqual(result.reason,"checkmate")
        self.assertEqual(len(board.board.move_stack),0)

    def test_ply_cap_is_explicit(self):
        result = play_game(RandomAgent(seed=1),RandomAgent(seed=2),max_plies=2)
        self.assertEqual(result.result,"1/2-1/2")
        self.assertEqual(result.reason,"max_plies_adjudication")
        self.assertEqual(result.white_calls,1)
        self.assertEqual(result.black_calls,1)

    def test_evaluator_alternates_colors(self):
        result = evaluate(BrokenAgent("bad"),RandomAgent("ok"),games=2)
        self.assertEqual(result["losses"],2)
        self.assertEqual([r["white"] for r in result["records"]],["bad","ok"])
        self.assertGreaterEqual(result["a_ms_per_move"],0)

    def test_round_robin_outputs_and_totals(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_tournament([RandomAgent(str(i),seed=i) for i in range(3)],
                                     games=2,max_plies=4,output_dir=folder)
            self.assertEqual(len(result["records"]),6)
            self.assertEqual(sum(r["games"] for r in result["leaderboard"]),12)
            self.assertTrue(all(r["elo"] == 1200 for r in result["leaderboard"]))
            for filename in ["leaderboard.md","leaderboard.csv","tournament_results.json"]:
                self.assertTrue((Path(folder)/filename).is_file())
            data=json.loads((Path(folder)/"tournament_results.json").read_text())
            self.assertEqual(len(data["records"]),6)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            run_tournament([RandomAgent("a"),RandomAgent("b")],games=3)
        with self.assertRaises(ValueError):
            run_tournament([RandomAgent(),RandomAgent()])
