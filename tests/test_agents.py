import unittest
import chess
from agents import *
from agents.stage1_heuristics.pst import evaluate_board, evaluate_relative
from agents.stage1_heuristics.think2_agent import capture_score
from agents.stage2_tree_search.minimax_agent import INF
from games.chess.board import ChessBoard

class AgentTests(unittest.TestCase):
    def test_contract_legal_and_no_mutation(self):
        board = ChessBoard()
        for text in ["e4","e5","Nf3","Nc6"]:
            board.step(text)
        before = (board.fen(), list(board.board.move_stack))
        for key in CLASSIC_AGENTS:
            agent = build_agent(key, 2, 7)
            with self.subTest(agent=key):
                self.assertIn(agent.act(board), board.legal_moves)
                self.assertEqual((board.fen(), board.board.move_stack), before)

    def test_every_agent_rejects_terminal(self):
        board = ChessBoard("7k/8/8/8/8/8/8/K7 w - - 0 1")
        for key in CLASSIC_AGENTS:
            with self.assertRaises(ValueError):
                build_agent(key,1,7).act(board)

    def test_registry_lists_all_agents(self):
        self.assertIn("supervised", AGENT_REGISTRY)
        self.assertEqual(set(AGENT_REGISTRY.keys()), set(CLASSIC_AGENTS) | {"supervised"})
        with self.assertRaises(ValueError):
            build_agent("supervised", 2, 7)

    def test_mate_in_one_both_colors(self):
        position = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        for raw in [position, position.mirror()]:
            for key in ["think1","think2","think3","minimax","alphabeta"]:
                board = ChessBoard(raw.fen())
                board.step(build_agent(key,2,7).act(board))
                self.assertTrue(board.is_checkmate(), key)

    def test_think2_prevents_mate(self):
        board = ChessBoard()
        for move in ["f3","e5"]:
            board.step(move)
        for seed in range(8):
            move = Think2Agent(seed=seed).act(board)
            raw = board.board.copy()
            raw.push(move)
            self.assertIsNone(Think1Agent.immediate_mate(raw, list(raw.legal_moves)))

    def test_mvv_lva_and_en_passant(self):
        raw = chess.Board("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1")
        self.assertEqual(capture_score(raw,chess.Move.from_uci("e5d6")),900)
        board = ChessBoard("7k/8/8/8/8/8/q7/R6K w - - 0 1")
        self.assertEqual(Think2Agent(seed=1).act(board).uci(),"a1a2")

    def test_pst_orientation_and_symmetry(self):
        self.assertEqual(evaluate_board(chess.Board()),0)
        board = chess.Board()
        board.push_san("Nf3")
        self.assertGreater(evaluate_board(board),0)
        self.assertEqual(evaluate_board(board),-evaluate_board(board.mirror()))
        low = chess.Board("7k/8/8/8/8/8/P7/7K w - - 0 1")
        high = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        self.assertGreater(evaluate_board(high),evaluate_board(low))

    def test_alpha_beta_matches_minimax_without_qsearch(self):
        for fen in [chess.STARTING_FEN, "7k/8/8/8/8/2n5/4P3/KR6 w - - 0 1"]:
            board = ChessBoard(fen)
            mini = MinimaxAgent(depth=2)
            alpha = AlphaBetaAgent(depth=2,use_quiescence=False)
            mini.act(board)
            alpha.act(board)
            self.assertEqual(mini.last_score,alpha.last_score)
            self.assertLessEqual(alpha.nodes_visited,mini.nodes_visited)

    def test_quiescence_sees_recapture(self):
        board = chess.Board("3rk3/8/8/3p4/8/8/8/3QK3 w - - 0 1")
        board.push_uci("d1d5")
        agent = AlphaBetaAgent(depth=1)
        tactical = agent._quiescence_search(board,-INF,INF)
        self.assertGreater(tactical,evaluate_relative(board)+500)

    def test_quiescence_checks_terminal_and_quiet_evasions(self):
        agent = AlphaBetaAgent(depth=1)
        mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(agent._quiescence_search(mate,-INF,INF),-30000)
        checked = chess.Board("4k3/8/8/8/8/8/4R3/K7 b - - 0 1")
        self.assertTrue(checked.is_check())
        before = checked.fen()
        agent.nodes_visited = 0
        agent._quiescence_search(checked,-INF,INF,q_depth=2)
        self.assertGreater(agent.nodes_visited,1)
        self.assertEqual(before,checked.fen())

    def test_depth_validation_and_seed(self):
        for cls in [MinimaxAgent,AlphaBetaAgent]:
            with self.assertRaises(ValueError):
                cls(depth=0)
        board = ChessBoard()
        a,b = RandomAgent(seed=7),RandomAgent(seed=7)
        self.assertEqual([a.act(board) for _ in range(8)],[b.act(board) for _ in range(8)])
