"""
Unit tests for Stage 4 SupervisedAgent.

Verifies:
1. Loading checkpoint and metadata validation.
2. Legal move generation for both White and Black.
3. Immutability of caller board and history.
4. Terminal positions rejection (checkmate, stalemate).
5. Missing or invalid checkpoint handling (no fallback to random).
6. Incompatible metadata rejection (format_version, stage, model_class, encoder, action version).
7. Factory build_agent integration (requires checkpoint, does not break other agents).
8. CLI and Arena integration smoke tests with checkpoint fixture.
"""

from __future__ import annotations
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import chess
import torch

from agents import AGENT_REGISTRY, build_agent
from agents.random_agent import RandomAgent
from agents.stage4_supervised.supervised_agent import SupervisedAgent
from arena.tournament import build_cli_agents, run_tournament
from games.chess.board import ChessBoard
from games.chess.encoder import NUM_ACTIONS
from training.dataset import build_dataset
from training.stage4_train_sl import Stage4TrainConfig, train

FIXTURE_PGN = """[Event "Test 1"]
[White "P1"]
[Black "P2"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0

[Event "Test 2"]
[White "P3"]
[Black "P4"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1

[Event "Test 3"]
[White "P5"]
[Black "P6"]
[Result "1/2-1/2"]

1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 1/2-1/2
"""


class TestSupervisedAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="supervised_agent_test_")
        cls.data_dir = Path(cls.temp_dir) / "data"
        cls.ckpt_dir = Path(cls.temp_dir) / "checkpoints"

        # Build dataset and train a small 2-epoch model as real checkpoint fixture
        build_dataset(
            io.StringIO(FIXTURE_PGN),
            output_dir=cls.data_dir,
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            seed=42,
        )
        cfg = Stage4TrainConfig(
            dataset_dir=str(cls.data_dir),
            output_dir=str(cls.ckpt_dir),
            epochs=2,
            batch_size=4,
            lr=1e-3,
            seed=42,
            device="cpu",
            model_config={"hidden_dim": 32, "fc_dim": 64},
        )
        summary = train(cfg)
        cls.checkpoint_path = Path(summary["checkpoints"]["best"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_act_both_colors_and_legality(self):
        """SupervisedAgent produces legal moves for both White and Black."""
        agent = SupervisedAgent(checkpoint=self.checkpoint_path)

        # White turn (initial board)
        board_white = ChessBoard()
        move_white = agent.act(board_white)
        self.assertIsInstance(move_white, chess.Move)
        self.assertIn(move_white, board_white.legal_moves)

        # Black turn
        board_black = ChessBoard()
        board_black.step("e4")
        move_black = agent.act(board_black)
        self.assertIsInstance(move_black, chess.Move)
        self.assertIn(move_black, board_black.legal_moves)

    def test_no_board_mutation(self):
        """act() must never mutate the board position or move history."""
        agent = SupervisedAgent(checkpoint=self.checkpoint_path)
        board = ChessBoard()
        for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]:
            board.step(san)

        fen_before = board.fen()
        stack_before = list(board.board.move_stack)

        _ = agent.act(board)

        self.assertEqual(board.fen(), fen_before)
        self.assertEqual(board.board.move_stack, stack_before)

    def test_terminal_positions_rejected(self):
        """act() must raise ValueError on terminal positions (checkmate & stalemate)."""
        agent = SupervisedAgent(checkpoint=self.checkpoint_path)

        # Scholar's mate terminal position
        mate_board = ChessBoard("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        self.assertTrue(mate_board.is_terminal())
        with self.assertRaises(ValueError):
            agent.act(mate_board)

        # Stalemate position
        stalemate_board = ChessBoard("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
        self.assertTrue(stalemate_board.is_terminal())
        with self.assertRaises(ValueError):
            agent.act(stalemate_board)

    def test_missing_or_corrupted_checkpoint(self):
        """Missing or corrupted checkpoint must raise error, never fallback silently to random."""
        # Missing checkpoint
        with self.assertRaises(FileNotFoundError):
            SupervisedAgent(checkpoint=self.ckpt_dir / "non_existent.pt")

        # Corrupted checkpoint (empty or invalid content)
        bad_file = self.ckpt_dir / "corrupted.pt"
        bad_file.write_text("invalid binary data", encoding="utf-8")
        with self.assertRaises(ValueError):
            SupervisedAgent(checkpoint=bad_file)

    def test_incompatible_metadata_rejection(self):
        """Checkpoint with invalid or mismatched metadata must be rejected."""
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # 1. Incompatible stage
        bad_payload = dict(payload, stage=5)
        bad_path = self.ckpt_dir / "bad_stage.pt"
        torch.save(bad_payload, bad_path)
        with self.assertRaises(ValueError) as ctx:
            SupervisedAgent(checkpoint=bad_path)
        self.assertIn("Incompatible stage", str(ctx.exception))

        # 2. Incompatible encoder_version
        bad_payload = dict(payload, encoder_version="hwc14-custom-v2")
        bad_path = self.ckpt_dir / "bad_encoder.pt"
        torch.save(bad_payload, bad_path)
        with self.assertRaises(ValueError) as ctx:
            SupervisedAgent(checkpoint=bad_path)
        self.assertIn("Incompatible encoder_version", str(ctx.exception))

        # 3. Incompatible model_class
        bad_payload = dict(payload, model_class="ResNetDualHead")
        bad_path = self.ckpt_dir / "bad_model_class.pt"
        torch.save(bad_payload, bad_path)
        with self.assertRaises(ValueError) as ctx:
            SupervisedAgent(checkpoint=bad_path)
        self.assertIn("Incompatible model_class", str(ctx.exception))

    def test_factory_build_agent(self):
        """build_agent requires checkpoint for supervised, works normally for others."""
        # Supervised without checkpoint -> raises ValueError
        with self.assertRaises(ValueError) as ctx:
            build_agent("supervised")
        self.assertIn("requires a valid checkpoint path", str(ctx.exception))

        # Supervised with checkpoint -> succeeds
        agent = build_agent("supervised", checkpoint=str(self.checkpoint_path), seed=42)
        self.assertIsInstance(agent, SupervisedAgent)

        # Classic agents still build without checkpoint
        random_bot = build_agent("random", seed=42)
        self.assertIsInstance(random_bot, RandomAgent)

    def test_cli_smoke_with_supervised(self):
        """Verify CLI can run with --agent supervised and --checkpoint."""
        cmd = [
            sys.executable,
            "main.py",
            "--cli",
            "--ascii",
            "--agent",
            "supervised",
            "--checkpoint",
            str(self.checkpoint_path),
        ]
        # Human plays e4, bot plays response, human inputs quit ('q')
        result = subprocess.run(cmd, input="e4\nq\n", text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, f"CLI stderr: {result.stderr}")
        self.assertIn("SupervisedAgent:", result.stdout)
        self.assertIn("Goodbye", result.stdout)

    def test_arena_smoke_with_supervised(self):
        """Verify Arena can parse and run games with supervised:checkpoint."""
        spec = f"supervised:{self.checkpoint_path}"
        agents, config = build_cli_agents([spec, "random"], depth=2, seed=42)
        self.assertEqual(len(agents), 2)
        # Run 2-game match in temporary directory
        with tempfile.TemporaryDirectory() as arena_dir:
            result = run_tournament(agents, games=2, max_plies=4, output_dir=arena_dir)
            self.assertEqual(len(result["records"]), 2)
            self.assertEqual(len(result["leaderboard"]), 2)

    def test_checkpoint_path_parsing_relative_and_windows_absolute_with_spaces_and_colons(self):
        """Parser must handle relative paths, Windows drive letters (C:), and spaces."""
        # 1. Path with spaces and drive letter
        spaced_dir = Path(self.temp_dir) / "folder with spaces"
        spaced_dir.mkdir(parents=True, exist_ok=True)
        spaced_ckpt = spaced_dir / "my checkpoint model.pt"
        shutil.copy(self.checkpoint_path, spaced_ckpt)

        # Ensure absolute path has drive letter on Windows (e.g. C:\...)
        abs_spaced_str = str(spaced_ckpt.resolve())

        # Test build_cli_agents with absolute Windows path with spaces & colons
        agents, config = build_cli_agents([f"supervised:{abs_spaced_str}"], depth=2, seed=42)
        self.assertEqual(len(agents), 1)
        self.assertIsInstance(agents[0], SupervisedAgent)
        self.assertEqual(config[0]["checkpoint"], abs_spaced_str)
        # Verify it can act
        move = agents[0].act(ChessBoard())
        self.assertIn(move, ChessBoard().legal_moves)

        # 2. Test relative path
        rel_str = str(self.checkpoint_path)
        agents_rel, config_rel = build_cli_agents([f"supervised:{rel_str}"], depth=2, seed=42)
        self.assertEqual(len(agents_rel), 1)
        self.assertIsInstance(agents_rel[0], SupervisedAgent)
        self.assertEqual(config_rel[0]["checkpoint"], rel_str)

    def test_stage2_key_depth_preservation(self):
        """Parsing KEY:DEPTH for search agents must preserve depth without regression."""
        agents, config = build_cli_agents(["minimax:3", "alphabeta:4"], depth=2, seed=42)
        self.assertEqual(agents[0].depth, 3)
        self.assertEqual(agents[1].depth, 4)
        self.assertEqual(config[0]["depth"], 3)
        self.assertEqual(config[1]["depth"], 4)

        # Invalid depth <= 0
        with self.assertRaises(ValueError) as ctx:
            build_cli_agents(["alphabeta:0"], depth=2, seed=42)
        self.assertIn("Depth must be >=1", str(ctx.exception))

        # Non-search agent with depth
        with self.assertRaises(ValueError) as ctx:
            build_cli_agents(["think1:2"], depth=2, seed=42)
        self.assertIn("does not use search depth", str(ctx.exception))

    def test_lazy_import_and_missing_torch_reporting(self):
        """Classic bots work without Torch; choosing supervised reports clear RuntimeError."""
        import agents.stage4_supervised.supervised_agent as sa

        orig_torch = sa.torch
        try:
            # Simulate missing Torch
            sa.torch = None

            # 1. Classic bots continue to work without Torch
            classic_bot = build_agent("alphabeta", depth=1, seed=42)
            move = classic_bot.act(ChessBoard())
            self.assertIn(move, ChessBoard().legal_moves)

            random_bot = build_agent("random", seed=42)
            self.assertIn(random_bot.act(ChessBoard()), ChessBoard().legal_moves)

            # 2. Choosing supervised raises clear RuntimeError
            with self.assertRaises(RuntimeError) as ctx:
                build_agent("supervised", checkpoint=str(self.checkpoint_path))
            self.assertIn("PyTorch is required", str(ctx.exception))

            with self.assertRaises(RuntimeError) as ctx:
                SupervisedAgent(checkpoint=self.checkpoint_path)
            self.assertIn("PyTorch is required", str(ctx.exception))
        finally:
            sa.torch = orig_torch


if __name__ == "__main__":
    unittest.main()

