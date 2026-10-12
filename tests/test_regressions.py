"""Regression coverage for the reviewed failures and compact GUI."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import chess
import numpy as np
import pygame
from agents import HumanAgent, RandomAgent, Think2Agent
from arena.evaluator import evaluate, play_game, restore_initial_board
from arena.tournament import main as tournament_main
from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder as E
from games.chess.gui import ChessGUI
from main import run_cli

PAWN_FEN = "7k/8/8/8/8/8/3p4/4K3 w - - 0 1"


class ResetFailure(RandomAgent):
    def reset(self):
        raise RuntimeError("reset failed")


class ActFailure(RandomAgent):
    def act(self, board):
        raise RuntimeError("search failed")


class ReviewRegressionTests(unittest.TestCase):
    def test_king_captures_pawn_for_every_seed(self):
        for raw in [chess.Board(PAWN_FEN), chess.Board(PAWN_FEN).mirror()]:
            board = ChessBoard(raw.fen())
            capture = next(m for m in board.legal_moves if raw.is_capture(m))
            for seed in range(30):
                self.assertEqual(Think2Agent(seed=seed).act(board), capture)

    def test_custom_fen_reset_survives_clone(self):
        original = ChessBoard(PAWN_FEN, claim_draw=False)
        original.step("Kf1")
        clone = original.clone()
        clone.reset()
        self.assertEqual(clone.fen(), PAWN_FEN)
        self.assertFalse(clone.claim_draw)
        self.assertNotEqual(original.fen(), clone.fen())

    def test_board_rejects_wrong_move_types(self):
        board = ChessBoard()
        for move in ["e2e4", 42, None, chess.Move.null()]:
            self.assertFalse(board.is_legal(move))
        self.assertTrue(board.is_legal(chess.Move.from_uci("e2e4")))

    def test_human_null_moves_reprompt(self):
        for null in ["0000", "--", "Z0", "@@@@"]:
            answers = iter([null, "e4"])
            human = HumanAgent(input_fn=lambda _: next(answers))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(human.act(ChessBoard()).uci(), "e2e4")
            self.assertIn("Illegal move", output.getvalue())

    def test_encoder_wrapper_state_and_reset_roundtrip(self):
        board = ChessBoard(PAWN_FEN, claim_draw=False)
        board.step("Kf1")
        np.testing.assert_array_equal(E.encode_board(board), E.encode_board(board.board))
        restored = E.decode_state(E.encode_state(board), as_wrapper=True)
        self.assertEqual(restored.fen(), board.fen())
        self.assertEqual(restored.board.move_stack, board.board.move_stack)
        self.assertFalse(restored.claim_draw)
        restored.reset()
        self.assertEqual(restored.fen(), PAWN_FEN)

    def test_terminal_policy_controls_mask_and_decode(self):
        board = ChessBoard(claim_draw=False)
        for san in ["Nf3", "Nf6", "Ng1", "Ng8"]*2:
            board.step(san)
        move = chess.Move.from_uci("g1f3")
        index = E.encode_move(move, board.turn)
        self.assertEqual(int(E.get_legal_action_mask(board).sum()), 20)
        board.claim_draw = True
        self.assertEqual(int(E.get_legal_action_mask(board).sum()), 0)
        self.assertEqual(E.legal_action_indices(board), [])
        self.assertIsNone(E.decode_move(index, board))
        self.assertEqual(E.decode_move(index, board.board, claim_draw=False), move)

    def test_result_relative_sign_at_mate(self):
        board = ChessBoard()
        for san in ["f3", "e5", "g4", "Qh4#"]:
            board.step(san)
        self.assertEqual(board.get_result_relative(), -1)
        self.assertEqual(-board.get_result_relative(), 1)

    def test_cli_bot_errors_are_reported_without_traceback(self):
        args = Namespace(fen=None, agent="random", depth=1, seed=1, side="black", ascii=True)
        for bot in [ActFailure(), RandomAgent()]:
            with contextlib.ExitStack() as stack:
                if not isinstance(bot, ActFailure):
                    stack.enter_context(patch.object(bot, "act", return_value="e2e4"))
                stack.enter_context(patch("main.build_agent", return_value=bot))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                error = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                self.assertEqual(run_cli(args), 1)
                self.assertIn("AI error", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())

    def test_cli_ascii_fallback_and_invalid_arguments(self):
        result = subprocess.run([sys.executable, "main.py", "--cli"], input="0000\ne4\nq\n",
                                text=True, capture_output=True, timeout=20,
                                env={**os.environ, "PYTHONIOENCODING": "ascii"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Illegal move", result.stdout)
        for command in [
            ["main.py", "--fen", "not a fen"],
            ["arena/tournament.py", "--games", "3"],
            ["arena/tournament.py", "--agents", "alphabeta:0", "random"],
            ["arena/tournament.py", "--agents", "random:2", "think3"],
        ]:
            result = subprocess.run([sys.executable, *command], text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)

    def test_reset_failures_forfeit_without_stopping_arena(self):
        for white, black, expected, reason in [
            (ResetFailure("bad"), RandomAgent("ok"), "0-1", "reset_forfeit"),
            (RandomAgent("ok"), ResetFailure("bad"), "1-0", "reset_forfeit"),
            (ResetFailure("a"), ResetFailure("b"), "1/2-1/2", "double_forfeit"),
        ]:
            result = play_game(white, black)
            self.assertEqual((result.result, result.reason), (expected, reason))
            self.assertEqual(result.moves, [])
            self.assertIn("reset failed", result.error)

    def test_arena_history_record_replays_draw(self):
        board = ChessBoard(claim_draw=False)
        for san in ["Nf3", "Nf6", "Ng1", "Ng8"]*2:
            board.step(san)
        board.claim_draw = True
        record = play_game(RandomAgent("a"), RandomAgent("b"), board=board)
        restored = restore_initial_board(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(restored.outcome().result(), record.result)
        self.assertEqual(restored.outcome().termination.name.lower(), record.reason)
        self.assertEqual(restored.board.move_stack, board.board.move_stack)

    def test_evaluator_requires_balanced_games(self):
        with self.assertRaises(ValueError):
            evaluate(RandomAgent("a"), RandomAgent("b"), games=3)

    def test_tournament_per_agent_depth_cli_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = tournament_main(["--agents", "alphabeta:1", "alphabeta:2",
                                      "--games", "2", "--max-plies", "2", "--output-dir", folder])
            self.assertEqual({r["agent"] for r in result["leaderboard"]},
                             {"AlphaBeta (Depth 1)", "AlphaBeta (Depth 2)"})
            self.assertEqual([a["depth"] for a in result["config"]["agents"]], [1, 2])
            self.assertTrue((Path(folder)/"leaderboard.csv").is_file())


class CompactGUIRegressionTests(unittest.TestCase):
    def setUp(self):
        self.gui = ChessGUI("random", board=ChessBoard(PAWN_FEN))

    def tearDown(self):
        self.gui.close()

    def test_small_displays_leave_title_bar_room(self):
        for desktop in [(1920,1080), (1366,768), (1024,600)]:
            width,height = ChessGUI.fit_window_size(desktop)
            self.assertLessEqual(width,800)
            self.assertLessEqual(height,736)
            self.assertLessEqual(width,desktop[0]-96)
            self.assertLessEqual(height,desktop[1]-140)

    def test_resize_mouse_mapping_and_close_event(self):
        self.gui.board = ChessBoard()
        for size in [(800,736),(600,500),(900,600)]:
            self.gui.handle_event(pygame.event.Event(pygame.VIDEORESIZE,w=size[0],h=size[1]))
            self.gui.board.reset()
            for square in [chess.E2,chess.E4]:
                x,y=self.gui.square_to_screen(square)
                viewport=self.gui.viewport
                pos=(round(viewport.x+(x+40)*viewport.width/1000),
                     round(viewport.y+(y+40)*viewport.height/920))
                self.gui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=pos))
            self.assertEqual(self.gui.board.board.peek().uci(),"e2e4")
            self.gui.render()
        self.gui.handle_event(pygame.event.Event(pygame.QUIT))
        self.assertFalse(self.gui.running)

    def test_resized_promotion_click(self):
        self.gui.board = ChessBoard("7k/P7/8/8/8/8/8/7K w - - 0 1")
        for square in [chess.A7,chess.A8]:
            x,y=self.gui.square_to_screen(square)
            self.gui.handle_click((x+40,y+40))
        viewport=self.gui.viewport
        pos=(round(viewport.x+650*viewport.width/1000),round(viewport.y+490*viewport.height/920))
        self.gui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=pos))
        self.assertEqual(self.gui.board.piece_at(chess.A8).piece_type,chess.KNIGHT)

    def test_custom_fen_preserved_by_all_restart_controls(self):
        for action in ["restart","flip","switch_bot"]:
            getattr(self.gui,action)()
            self.assertEqual(self.gui.board.fen(),PAWN_FEN)

    def test_worker_bad_types_reported(self):
        for move in ["e2e4",42,None,chess.Move.null()]:
            self.gui.results.put((self.gui.generation,move,0.1,""))
            self.gui.poll_worker()
            self.assertTrue(self.gui.error)
            self.assertEqual(self.gui.board.fen(),PAWN_FEN)
