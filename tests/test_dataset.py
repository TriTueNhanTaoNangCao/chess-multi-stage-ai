"""
Unit tests for Stage 4 PGN Dataset pipeline in training/dataset.py.
Verifies:
1. State-before-move extraction and target roundtrip encode/decode.
2. Special chess moves: both colors, castling (K & Q), en-passant, underpromotion.
3. Outcome handling: distinction between known results (1-0, 0-1, 1/2) and unknown/unfinished (*).
4. Variant validation: standard chess accepted, non-standard variants rejected.
5. Rejection of corrupted games (empty, illegal move, syntax errors) and stats tracking.
6. Duplicate games grouping (zero data leakage between splits).
7. Deterministic split reproducibility with seed.
8. Dataset serialization and reload with allow_pickle=False and Unicode game_ids.
9. PyTorch DataLoader integration: CHW in __getitem__, NCHW (B,14,8,8) when batched, correct dtypes.
"""

import io
import json
import shutil
import tempfile
import unittest
import chess
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

from games.chess.encoder import ChessEncoder, NUM_ACTIONS, NUM_CHANNELS
from training.dataset import (
    ChessDataset,
    ChessSample,
    build_dataset,
    load_dataset,
    parse_game,
    parse_outcome,
)

# Fixture 1: Standard game with castling (both sides) and checkmate (1-0)
PGN_STANDARD_SCHOLAR = """[Event "Scholar's Mate"]
[Site "Local"]
[Date "2026.10.24"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

# Fixture 2: Game containing Kingside & Queenside Castling, En-passant, and Underpromotion (1/2-1/2)
PGN_COMPLEX_SPECIAL_MOVES = """[Event "Special Moves Showcase"]
[Site "Local"]
[Date "2026.10.24"]
[Round "2"]
[White "White"]
[Black "Black"]
[Result "1/2-1/2"]
[FEN "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"]
[SetUp "1"]

1. O-O O-O-O 2. d6 cxd6 3. Bxa6+ Kb8 1/2-1/2
"""

# Fixture 3: En-passant specific test with unknown result (*)
PGN_EN_PASSANT_UNKNOWN_RESULT = """[Event "En Passant Test"]
[Site "Local"]
[Date "2026.10.24"]
[Round "3"]
[White "White"]
[Black "Black"]
[Result "*"]

1. e4 c6 2. e5 d5 3. exd6 *
"""

# Fixture 4: Underpromotion test (with Black pawn so game does not immediately draw by insufficient material)
PGN_UNDERPROMOTION = """[Event "Underpromotion Test"]
[Site "Local"]
[Date "2026.10.24"]
[Round "4"]
[White "White"]
[Black "Black"]
[Result "1-0"]
[FEN "8/4P3/8/8/8/8/7p/4K2k w - - 0 1"]
[SetUp "1"]

1. e8=N Kg2 2. Nf6 Kf3 1-0
"""

# Fixture 5: Non-standard variant game (e.g. Chess960)
PGN_CHESS960 = """[Event "Fischer Random"]
[Variant "Chess960"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

# Fixture 6: Corrupted games
PGN_EMPTY_GAME = """[Event "Empty Game"]
[Result "*"]

*
"""

PGN_ILLEGAL_MOVE = """[Event "Illegal Move Game"]
[Result "*"]

1. e4 e5 2. Qh5 Ke7 3. Qxe5# 4. h4 *
"""

PGN_SYNTAX_ERROR = """[Event "Illegal move error"]
[Result "*"]

1. e4 e5 2. e5 *
"""

PGN_CLAIMABLE_THREEFOLD_CONTINUES = """[Event "Claimable Threefold Continues"]
[Result "*"]

1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. e4 *
"""

PGN_CLAIMABLE_FIFTY_MOVE_CONTINUES = """[Event "Claimable Fifty Move Continues"]
[SetUp "1"]
[FEN "8/8/8/8/8/8/1R6/K6k w - - 98 50"]
[Result "*"]

50. Rb3 Kg1 *
"""


class TestPGNDatasetPipeline(unittest.TestCase):
    """Test suite for PGN parsing, sample extraction, splitting, and loading."""

    def test_state_before_move_and_target_roundtrip(self):
        """Samples must capture state BEFORE the move; target must roundtrip decode to original move."""
        stream = io.StringIO(PGN_STANDARD_SCHOLAR)
        game = chess.pgn.read_game(stream)
        samples, err, fp = parse_game(game, "game_0")

        self.assertIsNone(err)
        self.assertIsNotNone(samples)
        self.assertEqual(len(samples), 7)  # 7 plies in Scholar's mate

        # Verify board reconstructed ply-by-ply matches state before move
        board = chess.Board()
        expected_moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("d1h5"),
            chess.Move.from_uci("b8c6"),
            chess.Move.from_uci("f1c4"),
            chess.Move.from_uci("g8f6"),
            chess.Move.from_uci("h5f7"),
        ]

        for ply, (sample, move) in enumerate(zip(samples, expected_moves)):
            # 1. State must match board BEFORE move
            expected_state = ChessEncoder.encode_board(board)
            self.assertTrue(np.array_equal(sample.state, expected_state), f"Ply {ply} state mismatch")
            self.assertEqual(sample.ply_index, ply)
            self.assertEqual(sample.game_id, "game_0")
            self.assertEqual(sample.outcome, 1.0)
            self.assertTrue(sample.outcome_known)

            # 2. Target must decode back to exact move
            decoded_move = ChessEncoder.decode_move(sample.target, board)
            self.assertEqual(decoded_move, move, f"Ply {ply} decode mismatch")

            # 3. Target must be marked in legal mask
            self.assertTrue(sample.legal_mask[sample.target], f"Ply {ply} target not in mask")
            self.assertEqual(sample.legal_mask.shape, (NUM_ACTIONS,))
            self.assertEqual(sample.legal_mask.dtype, np.bool_)

            # Push move to advance board
            board.push(move)

    def test_outcome_parsing_and_unknown_handling(self):
        """Unknown or missing results ('*') must have outcome_known=False, not defaulted to draw."""
        # 1. Known results
        self.assertEqual(parse_outcome("1-0"), (1.0, True))
        self.assertEqual(parse_outcome("0-1"), (-1.0, True))
        self.assertEqual(parse_outcome("1/2-1/2"), (0.0, True))
        self.assertEqual(parse_outcome("1/2"), (0.0, True))

        # 2. Unknown results
        self.assertEqual(parse_outcome("*"), (0.0, False))
        self.assertEqual(parse_outcome(""), (0.0, False))
        self.assertEqual(parse_outcome("unknown"), (0.0, False))

        # 3. Verify game with '*' is still parsed for policy training with outcome_known=False
        stream = io.StringIO(PGN_EN_PASSANT_UNKNOWN_RESULT)
        game = chess.pgn.read_game(stream)
        samples, err, _ = parse_game(game, "game_star")
        self.assertIsNone(err)
        self.assertEqual(len(samples), 5)
        for s in samples:
            self.assertFalse(s.outcome_known)
            self.assertEqual(s.outcome, 0.0)

    def test_claimable_threefold_history_can_continue(self):
        game = chess.pgn.read_game(io.StringIO(PGN_CLAIMABLE_THREEFOLD_CONTINUES))
        moves = list(game.mainline_moves())
        samples, err, _ = parse_game(game, "threefold_continues")

        self.assertIsNone(err)
        self.assertEqual(len(samples), len(moves))

        board = game.board()
        for move in moves[:8]:
            board.push(move)
        self.assertEqual(len(board.move_stack), 8)
        self.assertTrue(board.can_claim_threefold_repetition())
        self.assertIsNotNone(board.outcome(claim_draw=True))
        self.assertIsNone(board.outcome(claim_draw=False))

        continuation = moves[8]
        sample = samples[8]
        self.assertIsNone(ChessEncoder.decode_move(sample.target, board))
        self.assertEqual(
            ChessEncoder.decode_move(sample.target, board, claim_draw=False),
            continuation,
        )
        self.assertTrue(sample.legal_mask[sample.target])

    def test_claimable_fifty_move_history_can_continue(self):
        game = chess.pgn.read_game(io.StringIO(PGN_CLAIMABLE_FIFTY_MOVE_CONTINUES))
        moves = list(game.mainline_moves())
        samples, err, _ = parse_game(game, "fifty_move_continues")

        self.assertIsNone(err)
        self.assertEqual(len(samples), len(moves))

        board = game.board()
        board.push(moves[0])
        self.assertEqual(len(board.move_stack), 1)
        self.assertTrue(board.can_claim_fifty_moves())
        self.assertIsNotNone(board.outcome(claim_draw=True))
        self.assertIsNone(board.outcome(claim_draw=False))

        continuation = moves[1]
        sample = samples[1]
        self.assertIsNone(ChessEncoder.decode_move(sample.target, board))
        self.assertEqual(
            ChessEncoder.decode_move(sample.target, board, claim_draw=False),
            continuation,
        )
        self.assertTrue(sample.legal_mask[sample.target])

    def test_variant_validation_and_fingerprint(self):
        """Standard variants must be accepted; non-standard variants must be cleanly rejected."""
        # Standard chess accepted
        game_std = chess.pgn.read_game(io.StringIO(PGN_STANDARD_SCHOLAR))
        samples, err, fp_std = parse_game(game_std, "g1")
        self.assertIsNone(err)
        self.assertGreater(len(fp_std), 0)

        # Chess960 rejected
        game_960 = chess.pgn.read_game(io.StringIO(PGN_CHESS960))
        samples_960, err_960, _ = parse_game(game_960, "g960")
        self.assertIsNone(samples_960)
        self.assertIn("unsupported_variant", err_960)
        self.assertIn("Chess960", err_960)

    def test_special_moves_castling_en_passant_underpromotion(self):
        """Verify pipeline handles White and Black castling, en-passant, and underpromotions."""
        # 1. Castling test (both colors: White O-O, Black O-O-O)
        stream_c = io.StringIO(PGN_COMPLEX_SPECIAL_MOVES)
        game_c = chess.pgn.read_game(stream_c)
        samples_c, err_c, _ = parse_game(game_c, "castling_game")
        self.assertIsNone(err_c)
        self.assertGreater(len(samples_c), 0)

        # White O-O (ply 0)
        white_o_o = samples_c[0]
        b_c = game_c.board()
        dec_w_castle = ChessEncoder.decode_move(white_o_o.target, b_c)
        self.assertEqual(dec_w_castle, chess.Move.from_uci("e1g1"))

        # Black O-O-O (ply 1)
        b_c.push(dec_w_castle)
        black_o_o_o = samples_c[1]
        dec_b_castle = ChessEncoder.decode_move(black_o_o_o.target, b_c)
        self.assertEqual(dec_b_castle, chess.Move.from_uci("e8c8"))

        # 2. En-passant test
        stream_ep = io.StringIO(PGN_EN_PASSANT_UNKNOWN_RESULT)
        game_ep = chess.pgn.read_game(stream_ep)
        samples_ep, err_ep, _ = parse_game(game_ep, "ep_game")
        self.assertIsNone(err_ep)
        self.assertEqual(len(samples_ep), 5)  # 1. e4 c6 2. e5 d5 3. exd6
        ep_sample = samples_ep[4]
        # Replay to ply 4
        b_ep = chess.Board()
        for m in [chess.Move.from_uci("e2e4"), chess.Move.from_uci("c7c6"),
                  chess.Move.from_uci("e4e5"), chess.Move.from_uci("d7d5")]:
            b_ep.push(m)
        self.assertEqual(b_ep.ep_square, chess.D6)
        ep_move = ChessEncoder.decode_move(ep_sample.target, b_ep)
        self.assertEqual(ep_move, chess.Move.from_uci("e5d6"))

        # 3. Underpromotion test (e8=N)
        stream_promo = io.StringIO(PGN_UNDERPROMOTION)
        game_promo = chess.pgn.read_game(stream_promo)
        samples_promo, err_promo, _ = parse_game(game_promo, "promo_game")
        self.assertIsNone(err_promo)
        promo_sample = samples_promo[0]
        b_promo = game_promo.board()
        promo_move = ChessEncoder.decode_move(promo_sample.target, b_promo)
        self.assertEqual(promo_move, chess.Move.from_uci("e7e8n"))

    def test_corrupted_games_rejection_and_stats(self):
        """Pipeline must reject invalid games, track error reasons, and report manifest stats."""
        multi_pgn = "\n\n".join([
            PGN_STANDARD_SCHOLAR,
            PGN_EMPTY_GAME,
            PGN_ILLEGAL_MOVE,
            PGN_SYNTAX_ERROR,
            PGN_CHESS960,
            PGN_EN_PASSANT_UNKNOWN_RESULT,
        ])
        stream = io.StringIO(multi_pgn)
        summary = build_dataset(stream, split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)

        stats = summary["stats"]
        self.assertEqual(stats["valid_games"], 2)  # Scholar + En-passant
        self.assertEqual(stats["rejected_games"], 4)  # Empty + Illegal + Syntax + Chess960
        self.assertIn("empty_game", stats["rejection_reasons"])
        self.assertIn("illegal_move", stats["rejection_reasons"])
        self.assertIn("unsupported_variant", stats["rejection_reasons"])

    def test_duplicate_games_grouping_and_no_leakage(self):
        """Identical games must share the same fingerprint and never cross split boundaries."""
        game_a = PGN_STANDARD_SCHOLAR
        game_b = PGN_EN_PASSANT_UNKNOWN_RESULT
        combined_pgn = "\n\n".join([game_a, game_b, game_a, game_b, game_a, game_b])

        stream = io.StringIO(combined_pgn)
        summary = build_dataset(
            stream,
            split_ratios={"train": 0.5, "val": 0.5, "test": 0.0},
            seed=42,
        )

        # 6 valid games total, but only 2 unique games
        self.assertEqual(summary["stats"]["valid_games"], 6)
        self.assertEqual(summary["stats"]["unique_games"], 2)
        self.assertEqual(summary["stats"]["duplicate_games"], 4)
        self.assertEqual(summary["duplicate_handling"], "kept_grouped_in_same_split")
        self.assertIn("leakage_note", summary)

        # Each unique game has 3 copies; splits must contain multiples of 3 games
        for split in ["train", "val"]:
            count = summary["num_games"][split]
            self.assertIn(count, [0, 3, 6], f"Split {split} has split instances of duplicates: {count}")

    def test_variant_alias_standard_and_chess(self):
        """Variant aliases 'Standard', 'Chess', and omitted variant must normalize to 'standard' and share fingerprints."""
        game_standard = '[Event "G1"]\n[Variant "Standard"]\n\n1. e4 e5 2. Nf3 Nc6 *'
        game_chess = '[Event "G2"]\n[Variant "Chess"]\n\n1. e4 e5 2. Nf3 Nc6 *'
        game_omitted = '[Event "G3"]\n\n1. e4 e5 2. Nf3 Nc6 *'

        res_std, err_std, fp_std = parse_game(chess.pgn.read_game(io.StringIO(game_standard)), game_id="g1")
        res_chs, err_chs, fp_chs = parse_game(chess.pgn.read_game(io.StringIO(game_chess)), game_id="g2")
        res_omt, err_omt, fp_omt = parse_game(chess.pgn.read_game(io.StringIO(game_omitted)), game_id="g3")

        self.assertIsNotNone(res_std)
        self.assertIsNotNone(res_chs)
        self.assertIsNotNone(res_omt)
        self.assertEqual(fp_std, fp_chs)
        self.assertEqual(fp_std, fp_omt)
        self.assertEqual(len(fp_std), 64)

        # When processed together in build_dataset, they should count as 1 unique game with 2 duplicates
        combined = "\n\n".join([game_standard, game_chess, game_omitted])
        summary = build_dataset(io.StringIO(combined), seed=42)
        self.assertEqual(summary["stats"]["valid_games"], 3)
        self.assertEqual(summary["stats"]["unique_games"], 1)
        self.assertEqual(summary["stats"]["duplicate_games"], 2)

    def test_seed_determinism(self):
        """Dataset build must produce identical split assignments with same seed."""
        moves_list = [
            "1. e4 e5", "1. d4 d5", "1. c4 e5", "1. Nf3 d5", "1. f4 d5",
            "1. g3 e5", "1. b3 e5", "1. e4 c5", "1. d4 Nf6", "1. e4 e6"
        ]
        pgn_text = "\n\n".join([f'[Event "G{i}"]\n\n{m} *' for i, m in enumerate(moves_list)])

        res1 = build_dataset(io.StringIO(pgn_text), split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=123)
        res2 = build_dataset(io.StringIO(pgn_text), split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=123)
        res3 = build_dataset(io.StringIO(pgn_text), split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=999)

        self.assertEqual(res1["num_games"], res2["num_games"])
        self.assertEqual(res1["num_samples"], res2["num_samples"])

    def test_save_and_load_roundtrip_with_allow_pickle_false_and_dataloader(self):
        """Verify saving dataset, allow_pickle=False loading, Unicode game_ids, and DataLoader batching."""
        temp_dir = tempfile.mkdtemp(prefix="chess_dataset_test_")
        try:
            combined_pgn = "\n\n".join([
                PGN_STANDARD_SCHOLAR,
                PGN_COMPLEX_SPECIAL_MOVES,
                PGN_EN_PASSANT_UNKNOWN_RESULT,
                PGN_UNDERPROMOTION,
            ])
            summary = build_dataset(
                io.StringIO(combined_pgn),
                output_dir=temp_dir,
                split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
                seed=42,
            )

            # 1. Check manifest metadata
            manifest_path = temp_dir + "/manifest.json"
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["encoder_version"], "hwc14-white-v1")
            self.assertEqual(manifest["action_version"], "az73-rankflip-v1")
            self.assertEqual(manifest["num_actions"], 4672)
            self.assertEqual(manifest["duplicate_handling"], "kept_grouped_in_same_split")
            self.assertIn("leakage_note", manifest)
            self.assertEqual(
                manifest["pgn_draw_claim_policy"],
                {
                    "claim_draw": False,
                    "description": "Continue through claimable draws while validating the recorded PGN mainline.",
                },
            )
            self.assertEqual(manifest["stats"]["total_samples_kept"], summary["stats"]["total_samples_kept"])

            # 2. Check allow_pickle=False works directly on .npz
            train_file = temp_dir + "/train.npz"
            raw_data = np.load(train_file, allow_pickle=False)
            self.assertTrue(raw_data["game_ids"].dtype.kind == "U", f"game_ids must be Unicode, got {raw_data['game_ids'].dtype}")
            self.assertEqual(raw_data["states"].dtype, np.float32)
            self.assertEqual(raw_data["targets"].dtype, np.int32)
            self.assertEqual(raw_data["legal_masks"].dtype, np.bool_)
            self.assertEqual(raw_data["outcome_known"].dtype, np.bool_)

            # 3. Load with load_dataset (which enforces allow_pickle=False)
            train_ds = load_dataset(temp_dir, split="train", to_torch=True)
            self.assertEqual(len(train_ds), summary["num_samples"]["train"])

            if len(train_ds) > 0:
                item = train_ds[0]
                self.assertIn("state", item)
                self.assertIn("target", item)
                self.assertIn("legal_mask", item)
                self.assertIn("outcome", item)
                self.assertIn("outcome_known", item)

                # Verify single item has CHW (14, 8, 8)
                if torch is not None:
                    self.assertIsInstance(item["state"], torch.Tensor)
                    self.assertEqual(item["state"].shape, (14, 8, 8))
                    self.assertEqual(item["state"].dtype, torch.float32)

                    self.assertIsInstance(item["target"], torch.Tensor)
                    self.assertEqual(item["target"].dtype, torch.long)

                    self.assertIsInstance(item["legal_mask"], torch.Tensor)
                    self.assertEqual(item["legal_mask"].shape, (NUM_ACTIONS,))
                    self.assertEqual(item["legal_mask"].dtype, torch.bool)

                    self.assertIsInstance(item["outcome_known"], torch.Tensor)
                    self.assertEqual(item["outcome_known"].dtype, torch.bool)

                    # 4. Verify PyTorch DataLoader collates into NCHW (B, 14, 8, 8)
                    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
                    batch = next(iter(loader))
                    b_size = batch["target"].shape[0]
                    self.assertEqual(batch["state"].shape, (b_size, 14, 8, 8))
                    self.assertEqual(batch["state"].dtype, torch.float32)
                    self.assertEqual(batch["target"].dtype, torch.long)
                    self.assertEqual(batch["legal_mask"].dtype, torch.bool)
                    self.assertEqual(batch["outcome"].shape, (b_size, 1))
                    self.assertEqual(batch["outcome_known"].dtype, torch.bool)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_flat_npy_split_as_memmap(self):
        """NPY arrays stay memory-mapped while preserving the NPZ dataset contract."""
        temp_dir = tempfile.mkdtemp(prefix="chess_dataset_npy_test_")
        try:
            arrays = {
                "states": np.zeros((2, 8, 8, NUM_CHANNELS), dtype=np.float32),
                "targets": np.array([1, 2], dtype=np.int32),
                "legal_masks": np.ones((2, NUM_ACTIONS), dtype=np.bool_),
                "outcomes": np.zeros((2,), dtype=np.float32),
            }
            for key, array in arrays.items():
                np.save(f"{temp_dir}/train_{key}.npy", array, allow_pickle=False)

            dataset = load_dataset(temp_dir, split="train", to_torch=False)

            self.assertEqual(len(dataset), 2)
            self.assertIsInstance(dataset.states, np.memmap)
            self.assertIsInstance(dataset.targets, np.memmap)
            self.assertIsNone(dataset.game_ids)
            self.assertEqual(dataset[1]["target"], 2)
            if torch is not None:
                torch_dataset = load_dataset(temp_dir, split="train", to_torch=True)
                self.assertEqual(torch_dataset[0]["state"].shape, (NUM_CHANNELS, 8, 8))
                self.assertEqual(torch_dataset[0]["legal_mask"].dtype, torch.bool)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
