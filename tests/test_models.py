"""
Unit tests for Stage 4 CNNPolicyNetwork and Stage 5 ValueNetwork.
Verifies:
1. Forward pass shapes and finite values (batch 1 and batch > 1).
2. Integration with real ChessEncoder input (HWC (8,8,14) -> NCHW (1,14,8,8)).
3. Backward pass and parameter updates (Cross-Entropy for policy, MSE for value).
4. Save/load state_dict with configuration and exact eval mode output matching.
5. Error handling on invalid input shapes.
"""

import io
import unittest
import chess
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS, NUM_CHANNELS


@unittest.skipIf(torch is None, "PyTorch is not installed in the environment")
class TestNeuralModels(unittest.TestCase):
    """Test suite for CNNPolicyNetwork and ValueNetwork."""

    def setUp(self):
        # Import lazily inside setUp / module
        from models.cnn_policy import CNNPolicyNetwork
        from models.value_net import ValueNetwork
        self.CNNPolicyNetwork = CNNPolicyNetwork
        self.ValueNetwork = ValueNetwork
        torch.manual_seed(42)

    def test_cnn_policy_forward_shapes_and_finite(self):
        """CNNPolicyNetwork forward produces (B, 4672) finite raw logits for batch 1 and batch 4."""
        model = self.CNNPolicyNetwork()
        model.eval()

        # Batch 1
        x1 = torch.randn(1, 14, 8, 8, dtype=torch.float32)
        with torch.no_grad():
            out1 = model(x1)
        self.assertEqual(out1.shape, (1, 4672))
        self.assertTrue(torch.isfinite(out1).all().item())

        # Batch 4
        x4 = torch.randn(4, 14, 8, 8, dtype=torch.float32)
        with torch.no_grad():
            out4 = model(x4)
        self.assertEqual(out4.shape, (4, 4672))
        self.assertTrue(torch.isfinite(out4).all().item())

    def test_cnn_policy_rejects_invalid_inputs(self):
        """CNNPolicyNetwork rejects 3D inputs and incorrect channel counts."""
        model = self.CNNPolicyNetwork()
        with self.assertRaises(ValueError):
            model(torch.randn(14, 8, 8))  # 3D tensor
        with self.assertRaises(ValueError):
            model(torch.randn(1, 10, 8, 8))  # 10 channels instead of 14

    def test_value_net_forward_shapes_and_bounds(self):
        """ValueNetwork forward produces (B, 1) finite values bounded in [-1.0, 1.0]."""
        model = self.ValueNetwork()
        model.eval()

        # Batch 1
        x1 = torch.randn(1, 14, 8, 8, dtype=torch.float32)
        with torch.no_grad():
            val1 = model(x1)
        self.assertEqual(val1.shape, (1, 1))
        self.assertTrue(torch.isfinite(val1).all().item())
        self.assertTrue((-1.0 <= val1).all().item())
        self.assertTrue((val1 <= 1.0).all().item())

        # Batch 4
        x4 = torch.randn(4, 14, 8, 8, dtype=torch.float32)
        with torch.no_grad():
            val4 = model(x4)
        self.assertEqual(val4.shape, (4, 1))
        self.assertTrue(torch.isfinite(val4).all().item())
        self.assertTrue((-1.0 <= val4).all().item())
        self.assertTrue((val4 <= 1.0).all().item())

    def test_value_net_rejects_invalid_inputs(self):
        """ValueNetwork rejects 3D inputs and incorrect channel counts."""
        model = self.ValueNetwork()
        with self.assertRaises(ValueError):
            model(torch.randn(14, 8, 8))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 12, 8, 8))

    def test_real_encoder_input_pipeline(self):
        """Test feeding real ChessEncoder output into both models."""
        policy_net = self.CNNPolicyNetwork()
        value_net = self.ValueNetwork()
        policy_net.eval()
        value_net.eval()

        test_fens = [
            chess.STARTING_FEN,
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "8/8/8/8/8/8/4k3/4K2R w - - 0 1",
        ]

        tensors = []
        for fen in test_fens:
            board = chess.Board(fen)
            # 1. Via encode_board() HWC -> transpose to NCHW
            hwc_planes = ChessEncoder.encode_board(board)
            self.assertEqual(hwc_planes.shape, (8, 8, 14))
            nchw = torch.from_numpy(hwc_planes.transpose(2, 0, 1).copy()).unsqueeze(0)
            self.assertEqual(nchw.shape, (1, 14, 8, 8))

            # 2. Via encode_board_tensor()
            nchw_direct = ChessEncoder.encode_board_tensor(board)
            self.assertTrue(torch.equal(nchw, nchw_direct))
            tensors.append(nchw)

        batch_tensor = torch.cat(tensors, dim=0)
        self.assertEqual(batch_tensor.shape, (3, 14, 8, 8))

        with torch.no_grad():
            logits = policy_net(batch_tensor)
            values = value_net(batch_tensor)

        self.assertEqual(logits.shape, (3, 4672))
        self.assertEqual(values.shape, (3, 1))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(values).all().item())

    def test_cnn_policy_backward_and_weight_update(self):
        """Verify Cross-Entropy backward pass produces finite gradients and updates parameters."""
        model = self.CNNPolicyNetwork()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 14, 8, 8, dtype=torch.float32)
        targets = torch.randint(0, 4672, (4,), dtype=torch.long)

        # Snapshot weights before step
        orig_fc2_weights = model.fc2.weight.clone().detach()

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertTrue(torch.isfinite(param.grad).all().item(), f"Parameter {name} gradient is not finite")

        optimizer.step()

        # Check that weights actually changed
        new_fc2_weights = model.fc2.weight.detach()
        self.assertFalse(torch.equal(orig_fc2_weights, new_fc2_weights), "Weights did not update after optimizer step")

    def test_value_net_backward_and_weight_update(self):
        """Verify MSE backward pass produces finite gradients and updates parameters."""
        model = self.ValueNetwork()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 14, 8, 8, dtype=torch.float32)
        targets = torch.tensor([[1.0], [-1.0], [0.0], [0.5]], dtype=torch.float32)

        orig_fc3_weights = model.fc3.weight.clone().detach()

        optimizer.zero_grad()
        pred_value = model(x)
        loss = F.mse_loss(pred_value, targets)
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertTrue(torch.isfinite(param.grad).all().item(), f"Parameter {name} gradient is not finite")

        optimizer.step()

        new_fc3_weights = model.fc3.weight.detach()
        self.assertFalse(torch.equal(orig_fc3_weights, new_fc3_weights), "Weights did not update after optimizer step")

    def test_save_load_state_dict_and_config(self):
        """Verify checkpoint serialization with architecture config reproduces identical eval outputs."""
        for model_cls in [self.CNNPolicyNetwork, self.ValueNetwork]:
            model1 = model_cls()
            config = model1.get_config()

            # Save state_dict in memory buffer
            buffer = io.BytesIO()
            torch.save({"config": config, "state_dict": model1.state_dict()}, buffer)
            buffer.seek(0)

            # Reconstruct from config and load weights
            checkpoint = torch.load(buffer, weights_only=True)
            model2 = model_cls.from_config(checkpoint["config"])
            model2.load_state_dict(checkpoint["state_dict"])

            model1.eval()
            model2.eval()

            x = torch.randn(3, 14, 8, 8, dtype=torch.float32)
            with torch.no_grad():
                out1 = model1(x)
                out2 = model2(x)

            self.assertTrue(torch.allclose(out1, out2, atol=1e-6), f"{model_cls.__name__} eval outputs do not match after reload")


if __name__ == "__main__":
    unittest.main()
