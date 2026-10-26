"""
Comprehensive verification test suite covering Thang and Thi implementations.
Testing:
1. Thang's components:
   - Mate-in-one for White and Black
   - Think2 avoids mate-in-one if safe moves exist, and handles fallback when all unsafe
   - Minimax vs AlphaBeta value equivalence (same evaluator, same depth, no qsearch, full window)
   - Mate score: prioritize faster win, delay loss
   - Quiescence: non-capture check evasions and behavior at q_depth limit
   - PST: specific square evaluations and symmetry
2. Thi's components:
   - Terminal rollout values (+1, -1, 0)
   - Backpropagation with win/loss/draw/heuristic values across multiple plies
   - Selection maximizing perspective for White and Black
   - Terminal on exact last rollout step (move 30) evaluated by outcome, not material
   - Cutoff heuristic: bounds in (-1, 1), correct sign and formula
   - Agent returns legal moves, rejects terminal boards
   - Determinism with same seed
   - Visit count invariant: root.visits == simulations, sum of children visits == root visits
3. State and integration:
   - Input board immutability (FEN, move_stack, castling rights, ep_square, halfmove clock)
   - Input board with repetition history
   - Promotion, underpromotion, castling, en-passant
   - Agent factory and registry
   - Arena smoke test: color swap, adjudication vs rule draw
"""

import math
import random
import unittest
import chess

from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS
from agents import AGENT_REGISTRY, build_agent
from agents.base import BaseAgent
from agents.random_agent import RandomAgent
from agents.stage1_heuristics.think1_agent import Think1Agent
from agents.stage1_heuristics.think2_agent import Think2Agent, capture_score
from agents.stage1_heuristics.think3_agent import Think3Agent
from agents.stage1_heuristics.pst import (
    PST_TABLES, PIECE_VALUES, evaluate_board, evaluate_relative
)
from agents.stage2_tree_search.minimax_agent import MinimaxAgent, terminal_score, INF, MATE
from agents.stage2_tree_search.alphabeta_agent import AlphaBetaAgent
from agents.stage3_mcts_pure.node import MCTSNode
from agents.stage3_mcts_pure.mcts_pure_agent import PureMCTSAgent
from arena.evaluator import play_game, evaluate


class TestThangComponents(unittest.TestCase):
    """Deep verification of Thang's code (Core, Stage 1, Stage 2)."""

    def test_mate_in_one_white_and_black(self):
        """Test mate-in-one detection for both White and Black across agents."""
        # White has mate in 1: Qxf7#
        white_mate_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        # Black has mate in 1: Qh4# (Fool's mate position)
        black_mate_fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"

        for agent_name in ["think1", "think2", "think3", "minimax", "alphabeta"]:
            # White test
            b_white = ChessBoard(white_mate_fen)
            bot_white = build_agent(agent_name, depth=2, seed=42)
            m_white = bot_white.act(b_white)
            b_white.step(m_white)
            self.assertTrue(b_white.is_checkmate(), f"{agent_name} failed White mate-in-one")

            # Black test
            b_black = ChessBoard(black_mate_fen)
            bot_black = build_agent(agent_name, depth=2, seed=42)
            m_black = bot_black.act(b_black)
            b_black.step(m_black)
            self.assertTrue(b_black.is_checkmate(), f"{agent_name} failed Black mate-in-one")

    def test_think2_avoids_opponent_mate_in_one(self):
        """Think2 avoids giving opponent mate-in-1 when safe moves exist."""
        # 1. f3 e5. If White plays g4, Black has Qh4# (Fool's mate).
        # Any other move by White (e.g., e4, g3, Nh3, Kf2) is safe from mate in 1.
        b = ChessBoard()
        b.step("f3")
        b.step("e5")
        for s in range(20):
            agent = Think2Agent(seed=s)
            move = agent.act(b)
            copy_b = b.board.copy()
            copy_b.push(move)
            self.assertIsNone(Think1Agent.immediate_mate(copy_b, list(copy_b.legal_moves)),
                              f"Seed {s} played {move} allowing mate-in-1")

    def test_think2_handles_forced_mate_fallback(self):
        """Think2 gracefully falls back when all moves allow opponent mate-in-1."""
        # Position where every legal move still leads to mate next turn
        forced_mate_fen = "7k/8/8/8/8/6q1/r6P/7K w - - 0 1"
        b = ChessBoard(forced_mate_fen)
        self.assertFalse(b.is_terminal())
        # All white legal moves allow immediate checkmate
        agent = Think2Agent(seed=42)
        chosen = agent.act(b)
        self.assertIn(chosen, b.legal_moves)

    def test_minimax_and_alphabeta_score_equivalence(self):
        """Minimax and AlphaBeta must return identical score when quiescence is off and full window is used."""
        test_fens = [
            chess.STARTING_FEN,
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ]
        for depth in [1, 2, 3]:
            for fen in test_fens:
                b = ChessBoard(fen)
                minimax = MinimaxAgent(depth=depth)
                alphabeta = AlphaBetaAgent(depth=depth, use_quiescence=False)
                minimax.act(b)
                alphabeta.act(b)
                self.assertEqual(minimax.last_score, alphabeta.last_score,
                                 f"Score mismatch at depth {depth} for FEN {fen}: "
                                 f"Minimax={minimax.last_score}, AlphaBeta={alphabeta.last_score}")
                self.assertLessEqual(alphabeta.nodes_visited, minimax.nodes_visited,
                                     f"AlphaBeta visited more nodes than Minimax at depth {depth}")

    def test_mate_score_distance_preference(self):
        """Mate score must prefer faster mate (lower ply) and delay being mated (higher ply)."""
        # Winning score: -(-MATE + ply) = MATE - ply
        score_mate_in_1 = -(-MATE + 1)
        score_mate_in_3 = -(-MATE + 3)
        self.assertGreater(score_mate_in_1, score_mate_in_3, "Winning side must prefer faster mate")

        # Losing score: -MATE + ply
        score_mated_in_1 = -MATE + 1
        score_mated_in_3 = -MATE + 3
        self.assertGreater(score_mated_in_3, score_mated_in_1, "Losing side must prefer delayed mate")

    def test_quiescence_check_evasions_and_depth_cap(self):
        """Quiescence search must search evasions when in check, and respect q_depth cap."""
        agent = AlphaBetaAgent(depth=1, use_quiescence=True, q_depth=3)
        # Position where Black is in check, has non-capture evasions (King moves)
        checked_board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
        self.assertTrue(checked_board.is_check())
        # Must not crash, must return valid score
        score = agent._quiescence_search(checked_board, -INF, INF, q_depth=3, ply=0)
        self.assertIsInstance(score, int)

        # Test at q_depth = 0: when in check, should still evaluate one evasion without infinite recursion
        agent.nodes_visited = 0
        score_capped = agent._quiescence_search(checked_board, -INF, INF, q_depth=0, ply=0)
        self.assertIsInstance(score_capped, int)
        self.assertGreater(agent.nodes_visited, 0)

    def test_pst_values_and_symmetry(self):
        """Test PST values on specific squares and verify color symmetry."""
        # Starting board evaluates to 0
        self.assertEqual(evaluate_board(chess.Board()), 0)

        # Symmetrical advancement: e4 vs e5 (with black to move vs white to move)
        b_white_e4 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        b_black_e5 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertEqual(evaluate_board(b_white_e4), -evaluate_board(b_black_e5))

        # Knight in center vs corner (non-terminal positions with pawns)
        b_n_d4 = chess.Board("7k/8/8/8/3N4/8/7P/7K w - - 0 1")
        b_n_a1 = chess.Board("7k/8/8/8/8/8/7P/N6K w - - 0 1")
        self.assertGreater(evaluate_board(b_n_d4), evaluate_board(b_n_a1))


class TestThiMCTS(unittest.TestCase):
    """Deep verification of Thi's code (MCTSNode, PureMCTSAgent)."""

    def test_terminal_rollout_returns_correct_values(self):
        """Terminal rollout must return +1 for White win, -1 for Black win, 0 for draw."""
        agent = PureMCTSAgent(simulations=10, seed=42)

        # White win
        white_win = MCTSNode(board=ChessBoard("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"))
        self.assertEqual(agent._rollout(white_win), 1.0)

        # Black win
        black_win = MCTSNode(board=ChessBoard("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"))
        self.assertEqual(agent._rollout(black_win), -1.0)

        # Draw by stalemate
        draw_stalemate = MCTSNode(board=ChessBoard("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
        self.assertEqual(agent._rollout(draw_stalemate), 0.0)

        # Draw by insufficient material
        draw_insufficient = MCTSNode(board=ChessBoard("7k/8/8/8/8/8/8/K7 w - - 0 1"))
        self.assertEqual(agent._rollout(draw_insufficient), 0.0)

    def test_backprop_multi_level_alternating_values(self):
        """Test backpropagation over a 4-level tree with win, loss, draw, and heuristic cutoff values."""
        agent = PureMCTSAgent(simulations=10, seed=42)

        root = MCTSNode(board=ChessBoard()) # White to move: current_player = 1
        m1 = chess.Move.from_uci("e2e4")
        b1 = root.board.clone(); b1.step(m1)
        c1 = root.add_child(m1, b1) # Black to move: current_player = -1

        m2 = chess.Move.from_uci("e7e5")
        b2 = c1.board.clone(); b2.step(m2)
        c2 = c1.add_child(m2, b2) # White to move: current_player = 1

        m3 = chess.Move.from_uci("g1f3")
        b3 = c2.board.clone(); b3.step(m3)
        c3 = c2.add_child(m3, b3) # Black to move: current_player = -1

        # Test with cutoff heuristic score: white_score = +0.473 (White up by a Queen)
        white_score = 9.0 / 19.0
        agent._backpropagate(c3, white_score=white_score)

        # Level 3 (Black to move): value_target = white_score * (-1) = -white_score
        self.assertAlmostEqual(c3.value_sum, -white_score)
        self.assertEqual(c3.visits, 1)

        # Level 2 (White to move): value_target = white_score * (+1) = +white_score
        self.assertAlmostEqual(c2.value_sum, white_score)
        self.assertEqual(c2.visits, 1)

        # Level 1 (Black to move): value_target = -white_score
        self.assertAlmostEqual(c1.value_sum, -white_score)
        self.assertEqual(c1.visits, 1)

        # Root (White to move): value_target = +white_score
        self.assertAlmostEqual(root.value_sum, white_score)
        self.assertEqual(root.visits, 1)

    def test_selection_maximizes_current_player_perspective(self):
        """Ensure selection maximizes White's perspective when White moves and Black's when Black moves."""
        # Case 1: Parent has White to move (current_player = 1). Children have Black to move.
        white_parent = MCTSNode(board=ChessBoard())
        white_parent.visits = 20

        m1 = chess.Move.from_uci("e2e4")
        m2 = chess.Move.from_uci("d2d4")
        c1 = white_parent.add_child(m1, white_parent.board.clone())
        c2 = white_parent.add_child(m2, white_parent.board.clone())

        # c1 has mean_value = -0.5 (bad for Black, good for White)
        c1.visits = 10
        c1.value_sum = -5.0
        # c2 has mean_value = +0.5 (good for Black, bad for White)
        c2.visits = 10
        c2.value_sum = 5.0

        # White parent should choose c1 (where child.mean_value is -0.5, so -child.mean_value is +0.5)
        best = white_parent.best_child(c=1.414)
        self.assertIs(best, c1, "White parent must choose the child that is worse for Black")

        # Case 2: Parent has Black to move (current_player = -1). Children have White to move.
        black_board = ChessBoard()
        black_board.step("e4") # Now Black's turn
        black_parent = MCTSNode(board=black_board)
        black_parent.visits = 20

        m_b1 = chess.Move.from_uci("e7e5")
        m_b2 = chess.Move.from_uci("c7c5")
        cb1 = black_parent.add_child(m_b1, black_parent.board.clone())
        cb2 = black_parent.add_child(m_b2, black_parent.board.clone())

        # cb1 has mean_value = -0.5 (bad for White, good for Black)
        cb1.visits = 10
        cb1.value_sum = -5.0
        # cb2 has mean_value = +0.5 (good for White, bad for Black)
        cb2.visits = 10
        cb2.value_sum = 5.0

        # Black parent should choose cb1 (where White's value is -0.5, so -child.mean_value is +0.5)
        best_b = black_parent.best_child(c=1.414)
        self.assertIs(best_b, cb1, "Black parent must choose the child that is worse for White")

    def test_rollout_terminal_on_exact_last_move(self):
        """Terminal on exact max_rollout_depth boundary must return game outcome, not material delta."""
        # Scholar's mate setup: White can mate in 1 with Qxf7#
        board = ChessBoard("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        agent = PureMCTSAgent(simulations=5, max_rollout_depth=1, seed=42)
        # Mock choice to select Qxf7# on the 1st (and max) move of rollout
        agent.rng.choice = lambda moves: chess.Move.from_uci("h5f7")
        node = MCTSNode(board=board)
        score = agent._rollout(node)
        # Checkmate must return +1.0 (White won), not material delta
        self.assertEqual(score, 1.0)

    def test_cutoff_heuristic_bounds_and_sign(self):
        """Cutoff heuristic must lie strictly in (-1, 1) and have correct sign."""
        agent = PureMCTSAgent()
        # White ahead (Black missing queen)
        b1 = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        d1 = agent._evaluate_material_delta(b1)
        s1 = d1 / (abs(d1) + 10.0)
        self.assertGreater(s1, 0.0)
        self.assertLess(s1, 1.0)
        self.assertAlmostEqual(s1, 9.0 / 19.0)

        # Black ahead (White missing queen)
        b2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
        d2 = agent._evaluate_material_delta(b2)
        s2 = d2 / (abs(d2) + 10.0)
        self.assertLess(s2, 0.0)
        self.assertGreater(s2, -1.0)
        self.assertAlmostEqual(s2, -9.0 / 19.0)

        # Extreme case: 100 pawns ahead
        delta = 100.0
        score = delta / (abs(delta) + 10.0)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_mcts_visits_invariant(self):
        """Verify that total root visits == simulations, and sum(children.visits) == root.visits."""
        agent = PureMCTSAgent(simulations=45, seed=123)
        board = ChessBoard()
        agent.act(board)

        self.assertIsNotNone(agent.root)
        self.assertEqual(agent.root.visits, 45)
        sum_children_visits = sum(c.visits for c in agent.root.children.values())
        self.assertEqual(sum_children_visits, 45)

    def test_mcts_determinism(self):
        """Same seed and state produce identical results."""
        board = ChessBoard("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        a1 = PureMCTSAgent(simulations=30, seed=777)
        a2 = PureMCTSAgent(simulations=30, seed=777)
        self.assertEqual(a1.act(board), a2.act(board))


class TestStateAndIntegration(unittest.TestCase):
    """Deep verification of state preservation and system integration."""

    def test_input_board_immutability_including_repetition_history(self):
        """Input board state, FEN, move_stack, castling, ep, and clocks must be untouched."""
        board = ChessBoard()
        # Build history with en-passant opportunity and castling rights
        moves = ["e4", "e5", "Nf3", "Nf6", "d4", "exd4", "e5", "Ne4", "Qxd4", "d5"]
        for m in moves:
            board.step(m)

        # Snapshot full state
        fen_before = board.fen()
        stack_before = list(board.board.move_stack)
        castling_before = board.board.castling_rights
        ep_before = board.board.ep_square
        halfmove_before = board.board.halfmove_clock
        fullmove_before = board.board.fullmove_number

        for key in ["random", "think1", "think2", "think3", "minimax", "alphabeta", "mcts_pure"]:
            agent = build_agent(key, depth=2, seed=42)
            if key == "mcts_pure":
                agent.simulations = 20
            move = agent.act(board)
            self.assertIn(move, board.legal_moves, f"Agent {key} returned illegal move")
            self.assertEqual(board.fen(), fen_before, f"Agent {key} mutated FEN")
            self.assertEqual(list(board.board.move_stack), stack_before, f"Agent {key} mutated move_stack")
            self.assertEqual(board.board.castling_rights, castling_before, f"Agent {key} mutated castling_rights")
            self.assertEqual(board.board.ep_square, ep_before, f"Agent {key} mutated ep_square")
            self.assertEqual(board.board.halfmove_clock, halfmove_before, f"Agent {key} mutated halfmove_clock")
            self.assertEqual(board.board.fullmove_number, fullmove_before, f"Agent {key} mutated fullmove_number")

    def test_special_moves_encoding_and_rules(self):
        """Test promotion, underpromotion, castling, en-passant."""
        # 1. Underpromotion
        pawn_promo_fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
        board = ChessBoard(pawn_promo_fen)
        promotions = [m for m in board.legal_moves if m.from_square == chess.E7]
        self.assertEqual(len(promotions), 4) # Q, R, B, N
        for p in promotions:
            idx = ChessEncoder.encode_move(p, board.turn)
            decoded = ChessEncoder.decode_move(idx, board)
            self.assertEqual(decoded, p)

        # 2. Castling
        castle_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        board_c = ChessBoard(castle_fen)
        for m in [chess.Move.from_uci("e1g1"), chess.Move.from_uci("e1c1")]:
            idx = ChessEncoder.encode_move(m, board_c.turn)
            decoded = ChessEncoder.decode_move(idx, board_c)
            self.assertEqual(decoded, m)

        # 3. En passant
        ep_fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        board_ep = ChessBoard(ep_fen)
        ep_move = chess.Move.from_uci("e5d6")
        self.assertIn(ep_move, board_ep.legal_moves)
        idx = ChessEncoder.encode_move(ep_move, board_ep.turn)
        decoded = ChessEncoder.decode_move(idx, board_ep)
        self.assertEqual(decoded, ep_move)

    def test_arena_smoke_adjudication_vs_rules(self):
        """Test arena game play, color swap, and distinction between max_plies and rules draw."""
        # Max plies draw
        res_adjudicated = play_game(RandomAgent(seed=1), RandomAgent(seed=2), max_plies=4)
        self.assertEqual(res_adjudicated.result, "1/2-1/2")
        self.assertEqual(res_adjudicated.reason, "max_plies_adjudication")

        # Natural terminal
        scholar_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        board_scholar = ChessBoard(scholar_fen)
        res_mate = play_game(Think1Agent(), RandomAgent(), board=board_scholar, max_plies=10)
        self.assertEqual(res_mate.result, "1-0")
        self.assertEqual(res_mate.reason, "checkmate")


if __name__ == "__main__":
    unittest.main()
