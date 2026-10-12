"""Headless SDL interaction tests: no visible desktop window required."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import unittest
import chess
import pygame
from games.chess.gui import ChessGUI
from games.chess.board import ChessBoard

class GUITests(unittest.TestCase):
    def setUp(self):
        self.gui = ChessGUI("random",depth=1)

    def tearDown(self):
        self.gui.close()

    def click(self,square):
        x,y = self.gui.square_to_screen(square)
        self.gui.handle_click((x+40,y+40))

    def test_coordinate_mapping_both_sides(self):
        for color in chess.COLORS:
            self.gui.human_color = color
            for square in chess.SQUARES:
                x,y=self.gui.square_to_screen(square)
                self.assertEqual(self.gui.screen_to_square(x+40,y+40),square)
        self.assertIsNone(self.gui.screen_to_square(10,10))

    def test_click_select_move_and_undo(self):
        self.click(chess.E2)
        self.assertEqual(self.gui.selected,chess.E2)
        self.click(chess.E4)
        self.assertEqual(self.gui.board.board.peek().uci(),"e2e4")
        self.gui.board.step("e5")
        self.gui.undo()
        self.assertEqual(self.gui.board.fen(),chess.STARTING_FEN)

    def test_promotions_require_choice(self):
        for index,kind in enumerate((chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT)):
            self.gui.board = ChessBoard("7k/P7/8/8/8/8/8/7K w - - 0 1")
            self.click(chess.A7)
            self.click(chess.A8)
            self.assertEqual(len(self.gui.promotion_moves),4)
            self.gui.handle_click((350+index*100,490))
            self.assertEqual(self.gui.board.piece_at(chess.A8).piece_type,kind)

    def test_stale_worker_result_is_discarded(self):
        old = self.gui.generation
        self.gui.working = True
        self.gui.restart()
        self.gui.results.put((old,chess.Move.from_uci("e2e4"),0.01,""))
        self.gui.poll_worker()
        self.assertEqual(self.gui.board.fen(),chess.STARTING_FEN)
        self.assertFalse(self.gui.working)

    def test_keyboard_restart_switch_flip(self):
        for key in [pygame.K_m,pygame.K_f,pygame.K_r]:
            self.gui.handle_event(pygame.event.Event(pygame.KEYDOWN,key=key))
        self.assertEqual(self.gui.agent_key,"think1")
        self.assertEqual(self.gui.human_color,chess.BLACK)
        self.assertEqual(self.gui.board.fen(),chess.STARTING_FEN)

    def test_check_terminal_and_render(self):
        self.gui.board = ChessBoard("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertIn("Thắng",self.gui.status())
        self.gui.render()
        self.assertEqual(self.gui.screen.get_size(),(1000,920))

    def test_worker_delivers_legal_move(self):
        import time
        self.gui.human_color = chess.BLACK
        self.gui.start_worker()
        deadline = time.monotonic()+3
        while self.gui.working and time.monotonic()<deadline:
            self.gui.poll_worker()
            time.sleep(0.005)
        self.assertFalse(self.gui.working)
        self.assertFalse(self.gui.error)
        self.assertEqual(len(self.gui.board.board.move_stack),1)
