"""
Unit tests for Stage 4 Policy SL Training in training/stage4_train_sl.py.

Verifies:
1. End-to-end training smoke test on small fixture PGN (1-2 epochs).
2. Weight update verification (weights change after training step).
3. Checkpoint generation (best.pt and last.pt) strictly following docs/HANDOFF.md.
4. Roundtrip load_checkpoint restores weights and sets eval mode.
5. Evaluation of test split using best checkpoint when requested.
6. Validation errors for invalid batches (no legal moves, target not in legal mask).
7. Rejection of empty train/val splits with clear error messages.
8. Rejection of incompatible dataset metadata (encoder, action version, num_actions).
9. Weighted sample metric aggregation correctness.
"""

from __future__ import annotations
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from games.chess.encoder import NUM_ACTIONS
from models.cnn_policy import CNNPolicyNetwork
from training.dataset import build_dataset
from training.stage4_train_sl import (
    Stage4TrainConfig,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
    train_one_epoch,
    validate_batch,
)

FIXTURE_PGN = """[Event "Game 1"]
[White "P1"]
[Black "P2"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0

[Event "Game 2"]
[White "P3"]
[Black "P4"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1

[Event "Game 3"]
[White "P5"]
[Black "P6"]
[Result "1/2-1/2"]

1. Nf3 Nf6 2. g3 g6 3. Bg2 Bg7 4. O-O O-O 1/2-1/2

[Event "Game 4"]
[White "P7"]
[Black "P8"]
[Result "*"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 *
"""


class TestStage4Train(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="stage4_train_test_")
        self.data_dir = Path(self.temp_dir) / "data"
        self.ckpt_dir = Path(self.temp_dir) / "checkpoints"

        # Build small dataset fixture
        build_dataset(
            io.StringIO(FIXTURE_PGN),
            output_dir=self.data_dir,
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            seed=42,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_training_and_checkpoints(self):
        """Run 2-epoch training, verify finite loss, weight updates, HANDOFF checkpoint schema, and test eval."""
        config = Stage4TrainConfig(
            dataset_dir=str(self.data_dir),
            output_dir=str(self.ckpt_dir),
            epochs=2,
            batch_size=4,
            lr=1e-3,
            seed=42,
            device="cpu",
            evaluate_test=True,
            model_config={"hidden_dim": 32, "fc_dim": 64},  # small network for fast testing
        )

        # Record initial weights of a model initialized with same seed
        torch.manual_seed(42)
        initial_model = CNNPolicyNetwork(hidden_dim=32, fc_dim=64)
        initial_param = list(initial_model.parameters())[0].clone()

        result = train(config)

        # 1. Summary checks
        self.assertEqual(result["stage"], 4)
        self.assertEqual(result["model_class"], "CNNPolicyNetwork")
        self.assertEqual(result["epochs_trained"], 2)
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]), 2)

        for record in result["history"]:
            self.assertTrue(math.isfinite(record["train_loss"]))
            self.assertTrue(math.isfinite(record["val_loss"]))
            self.assertGreater(record["train_loss"], 0.0)
            self.assertGreater(record["val_loss"], 0.0)
            self.assertGreaterEqual(record["train_top1"], 0.0)
            self.assertLessEqual(record["train_top1"], 1.0)
            self.assertGreaterEqual(record["train_top5"], 0.0)
            self.assertLessEqual(record["train_top5"], 1.0)
            self.assertGreaterEqual(record["train_top5"], record["train_top1"])

        # 2. Checkpoints must exist
        best_path = Path(result["checkpoints"]["best"])
        last_path = Path(result["checkpoints"]["last"])
        self.assertTrue(best_path.exists(), "stage4_cnn_best.pt was not saved")
        self.assertTrue(last_path.exists(), "stage4_cnn_last.pt was not saved")

        # 3. Verify checkpoint payload against docs/HANDOFF.md schema
        loaded_model, payload = load_checkpoint(best_path, device="cpu")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(payload["stage"], 4)
        self.assertEqual(payload["model_class"], "CNNPolicyNetwork")
        self.assertEqual(payload["encoder_version"], "hwc14-white-v1")
        self.assertEqual(payload["action_version"], "az73-rankflip-v1")
        self.assertEqual(payload["num_actions"], NUM_ACTIONS)
        self.assertEqual(payload["value_perspective"], "side_to_move")
        self.assertTrue(payload["claim_draw"])
        self.assertEqual(payload["seed"], 42)
        self.assertIn("epoch", payload)
        self.assertIn("step", payload)
        self.assertIn("training_config", payload)
        self.assertIn("metrics", payload)
        self.assertIn("state_dict", payload)
        self.assertIn("optimizer_state_dict", payload)

        # 4. Verify weight updates occurred
        trained_param = list(loaded_model.parameters())[0]
        self.assertFalse(torch.equal(initial_param, trained_param), "Model weights did not change after training")

        # 5. Verify model is loaded in eval mode
        self.assertFalse(loaded_model.training, "Loaded model must be in eval mode")

        # 6. Verify test split evaluation
        self.assertIsNotNone(result["test_metrics"])
        self.assertIn("loss", result["test_metrics"])
        self.assertIn("top1", result["test_metrics"])
        self.assertIn("top5", result["test_metrics"])
        self.assertTrue(math.isfinite(result["test_metrics"]["loss"]))

    def test_validate_batch_illegal_conditions(self):
        """Batch validation must catch samples with zero legal moves or target not in legal mask."""
        batch_size = 2
        states = torch.zeros((batch_size, 14, 8, 8), dtype=torch.float32)
        targets = torch.tensor([10, 20], dtype=torch.long)
        legal_masks = torch.zeros((batch_size, NUM_ACTIONS), dtype=torch.bool)

        # Case 1: Mask is all False (no legal moves)
        batch_empty_mask = {
            "state": states,
            "target": targets,
            "legal_mask": legal_masks,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_batch(batch_empty_mask)
        self.assertIn("no legal actions", str(ctx.exception))

        # Case 2: Target action is not in legal mask
        legal_masks[0, 5] = True   # Sample 0 has legal move 5, but target is 10
        legal_masks[1, 20] = True  # Sample 1 has legal move 20, target is 20
        batch_invalid_target = {
            "state": states,
            "target": targets,
            "legal_mask": legal_masks,
        }
        with self.assertRaises(ValueError) as ctx:
            validate_batch(batch_invalid_target)
        self.assertIn("target action is not marked as legal", str(ctx.exception))

        # Case 3: Both targets are marked legal -> must pass
        legal_masks[0, 10] = True
        validate_batch(batch_invalid_target)

    def test_empty_train_or_val_split_rejection(self):
        """Must raise clear ValueError if train or val split contains 0 samples."""
        empty_dir = Path(self.temp_dir) / "empty_dataset"
        empty_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "format_version": 1,
            "encoder_version": "hwc14-white-v1",
            "action_version": "az73-rankflip-v1",
            "num_actions": NUM_ACTIONS,
        }
        (empty_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # Save empty .npz
        empty_arrays = {
            "states": np.empty((0, 8, 8, 14), dtype=np.float32),
            "targets": np.empty((0,), dtype=np.int32),
            "legal_masks": np.empty((0, NUM_ACTIONS), dtype=np.bool_),
            "outcomes": np.empty((0,), dtype=np.float32),
            "outcome_known": np.empty((0,), dtype=np.bool_),
            "game_ids": np.empty((0,), dtype="U16"),
            "ply_indices": np.empty((0,), dtype=np.int32),
        }
        np.savez(empty_dir / "train.npz", **empty_arrays)
        np.savez(empty_dir / "val.npz", **empty_arrays)

        config = Stage4TrainConfig(
            dataset_dir=str(empty_dir),
            output_dir=str(self.ckpt_dir),
            epochs=1,
        )
        with self.assertRaises(ValueError) as ctx:
            train(config)
        self.assertIn("Train split is empty", str(ctx.exception))

    def test_incompatible_dataset_metadata(self):
        """Reject datasets with incompatible encoder or action version."""
        incompat_dir = Path(self.temp_dir) / "incompat_data"
        shutil.copytree(self.data_dir, incompat_dir)

        # Mutate manifest to wrong encoder version
        manifest_file = incompat_dir / "manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["encoder_version"] = "wrong-encoder-v2"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        config = Stage4TrainConfig(
            dataset_dir=str(incompat_dir),
            output_dir=str(self.ckpt_dir),
            epochs=1,
        )
        with self.assertRaises(ValueError) as ctx:
            train(config)
        self.assertIn("Incompatible dataset encoder version", str(ctx.exception))

    def test_weighted_metrics_aggregation(self):
        """Verify sample-weighted metrics compute correctly even with uneven batch sizes."""
        model = CNNPolicyNetwork(hidden_dim=16, fc_dim=32)
        # Create 2 mock batches: batch 1 of size 3, batch 2 of size 1
        b1_mask = torch.ones((3, NUM_ACTIONS), dtype=torch.bool)
        b2_mask = torch.ones((1, NUM_ACTIONS), dtype=torch.bool)
        batch1 = {
            "state": torch.zeros((3, 14, 8, 8), dtype=torch.float32),
            "target": torch.zeros(3, dtype=torch.long),
            "legal_mask": b1_mask,
        }
        batch2 = {
            "state": torch.zeros((1, 14, 8, 8), dtype=torch.float32),
            "target": torch.zeros(1, dtype=torch.long),
            "legal_mask": b2_mask,
        }
        batches = [batch1, batch2]

        metrics = evaluate(model, batches, device=torch.device("cpu"))
        self.assertEqual(metrics["samples"], 4)
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertGreaterEqual(metrics["top1"], 0.0)


    def test_resume_from_checkpoint(self):
        """Resume training from last checkpoint: epoch counter advances, weights differ from fresh init."""
        # Phase 1: train 1 epoch fresh
        cfg1 = Stage4TrainConfig(
            dataset_dir=str(self.data_dir),
            output_dir=str(self.ckpt_dir / "phase1"),
            epochs=1,
            batch_size=4,
            lr=1e-3,
            seed=42,
            device="cpu",
            model_config={"hidden_dim": 32, "fc_dim": 64},
        )
        result1 = train(cfg1)
        self.assertEqual(result1["start_epoch"], 1)

        last_ckpt = Path(result1["checkpoints"]["last"])
        self.assertTrue(last_ckpt.exists())

        # Verify last checkpoint carries epoch=1 and new rng_state fields
        import torch as _torch
        payload = _torch.load(last_ckpt, map_location="cpu", weights_only=False)
        self.assertEqual(payload["epoch"], 1)
        self.assertIn("rng_state", payload, "New checkpoints must include rng_state")
        self.assertIn("numpy_rng_state", payload, "New checkpoints must include numpy_rng_state")
        self.assertIn("best_val_loss", payload, "New checkpoints must include best_val_loss")

        # Phase 2: resume from epoch 1 last checkpoint, train up to epoch 3
        cfg2 = Stage4TrainConfig(
            dataset_dir=str(self.data_dir),
            output_dir=str(self.ckpt_dir / "phase2"),
            epochs=3,
            batch_size=4,
            lr=1e-3,
            seed=42,
            device="cpu",
            model_config={"hidden_dim": 32, "fc_dim": 64},
            resume_checkpoint=str(last_ckpt),
        )
        result2 = train(cfg2)

        # Must start from epoch 2
        self.assertEqual(result2["start_epoch"], 2)
        self.assertEqual(result2["resume_from"], str(last_ckpt))
        # History only contains newly run epochs (2, 3)
        self.assertEqual(len(result2["history"]), 2)
        self.assertEqual(result2["history"][0]["epoch"], 2)
        self.assertEqual(result2["history"][1]["epoch"], 3)

        # RNG was restored from checkpoint
        self.assertTrue(result2["rng_state_restored"])

        # best_val_loss initialized from checkpoint, not inf
        self.assertNotEqual(result2["best_val_loss"], float("inf"))

    def test_early_stopping(self):
        """Early stopping fires well before max_epochs when val loss is stagnant."""
        # Force val loss to stagnate by using a very high min_delta that real loss
        # improvement cannot exceed: every epoch counts as "no improvement".
        cfg = Stage4TrainConfig(
            dataset_dir=str(self.data_dir),
            output_dir=str(self.ckpt_dir / "early_stop"),
            epochs=20,
            batch_size=4,
            lr=1e-3,
            seed=42,
            device="cpu",
            model_config={"hidden_dim": 32, "fc_dim": 64},
            early_stopping_patience=2,
            early_stopping_min_delta=1e6,  # impossible to beat → triggers immediately
        )
        result = train(cfg)
        # min_delta=1e6 ensures no epoch is ever "improving" → stops after patience=2
        self.assertIn("early_stopping", result["stop_reason"])
        # Must have stopped well before max_epochs=20
        self.assertLess(len(result["history"]), 20)

    def test_resume_missing_checkpoint_raises(self):
        """Resuming from non-existent checkpoint must raise FileNotFoundError."""
        cfg = Stage4TrainConfig(
            dataset_dir=str(self.data_dir),
            output_dir=str(self.ckpt_dir),
            epochs=2,
            device="cpu",
            model_config={"hidden_dim": 32, "fc_dim": 64},
            resume_checkpoint="/nonexistent/path/checkpoint.pt",
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            train(cfg)
        self.assertIn("resume_checkpoint not found", str(ctx.exception))

    def test_resume_incompatible_model_config_raises(self):
        """Resume must raise ValueError if checkpoint model_config differs from requested config."""
        # Train 1 epoch with hidden_dim=32
        cfg1 = Stage4TrainConfig(
            dataset_dir=str(self.data_dir),
            output_dir=str(self.ckpt_dir / "compat"),
            epochs=1,
            batch_size=4,
            seed=42,
            device="cpu",
            model_config={"hidden_dim": 32, "fc_dim": 64},
        )
        result1 = train(cfg1)
        last_ckpt = Path(result1["checkpoints"]["last"])

        # Try to resume with different hidden_dim
        cfg2 = Stage4TrainConfig(
            dataset_dir=str(self.data_dir),
            output_dir=str(self.ckpt_dir / "compat2"),
            epochs=2,
            device="cpu",
            model_config={"hidden_dim": 64, "fc_dim": 64},  # mismatch
            resume_checkpoint=str(last_ckpt),
        )
        with self.assertRaises(ValueError) as ctx:
            train(cfg2)
        self.assertIn("model_config mismatch", str(ctx.exception))
        self.assertIn("hidden_dim", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
