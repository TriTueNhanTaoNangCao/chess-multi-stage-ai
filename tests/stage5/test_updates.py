"""Stage 5 Policy and Value update regressions."""
from tests.stage5.helpers import *

class TestStage5PolicyUpdate(unittest.TestCase):
    def test_positive_and_negative_targets_reverse_gradient_direction(self):
        positive_model = LogitPolicy()
        positive_optimizer = CountingSGD(positive_model.parameters(), lr=0.1)
        positive_before = positive_model.logits.detach().clone()
        positive_metrics = reinforce_policy_update(
            positive_model,
            positive_optimizer,
            [make_trajectory([make_step(target=1.0)])],
        )
        self.assertGreater(positive_model.logits[0].item(), positive_before[0].item())
        self.assertLess(positive_model.logits[1].item(), positive_before[1].item())
        self.assertEqual(positive_optimizer.step_calls, 1)
        self.assertEqual(positive_metrics["decisions"], 1)

        negative_model = LogitPolicy()
        negative_optimizer = CountingSGD(negative_model.parameters(), lr=0.1)
        negative_before = negative_model.logits.detach().clone()
        negative_metrics = reinforce_policy_update(
            negative_model,
            negative_optimizer,
            [make_trajectory([make_step(target=-1.0)])],
        )
        self.assertLess(negative_model.logits[0].item(), negative_before[0].item())
        self.assertGreater(negative_model.logits[1].item(), negative_before[1].item())
        self.assertEqual(negative_optimizer.step_calls, 1)
        self.assertLess(negative_metrics["policy_loss"], 0.0)

    def test_legal_mask_excludes_invalid_logits_and_rejects_invalid_target_action(self):
        model = LogitPolicy()
        with torch.no_grad():
            model.logits[2] = 100.0
        optimizer = CountingSGD(model.parameters(), lr=0.0)
        metrics = reinforce_policy_update(
            model,
            optimizer,
            [make_trajectory([make_step(action=0, legal_actions=(0, 1))])],
        )
        self.assertAlmostEqual(metrics["policy_loss"], np.log(2.0), places=6)

        invalid_model = LogitPolicy()
        invalid_optimizer = CountingSGD(invalid_model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "present in its legal mask"):
            reinforce_policy_update(
                invalid_model,
                invalid_optimizer,
                [make_trajectory([make_step(action=2, legal_actions=(0, 1))])],
            )
        self.assertEqual(invalid_optimizer.step_calls, 0)

    def test_loss_reduction_is_mean_of_per_game_sums(self):
        model = LogitPolicy()
        optimizer = CountingSGD(model.parameters(), lr=0.0)
        short_game = make_trajectory([make_step()])
        long_game = make_trajectory([make_step(), make_step()])
        metrics = reinforce_policy_update(
            model,
            optimizer,
            [short_game, long_game],
            chunk_size=1,
        )
        self.assertAlmostEqual(metrics["policy_loss"], 1.5 * np.log(2.0), places=6)
        self.assertEqual(metrics["games_used"], 2)
        self.assertEqual(metrics["games_skipped"], 0)
        self.assertEqual(metrics["decisions"], 3)
        self.assertEqual(optimizer.step_calls, 1)

    def test_truncated_and_empty_batches_do_not_step(self):
        model = LogitPolicy()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        truncated = make_trajectory([make_step(target=None)], truncated=True)
        no_decisions = make_trajectory([])
        metrics = reinforce_policy_update(
            model, optimizer, [truncated, no_decisions]
        )
        self.assertIsNone(metrics["policy_loss"])
        self.assertEqual(metrics["games_used"], 0)
        self.assertEqual(metrics["games_skipped"], 2)
        self.assertEqual(metrics["games_truncated"], 1)
        self.assertEqual(metrics["decisions"], 0)
        self.assertEqual(optimizer.step_calls, 0)

        empty_metrics = reinforce_policy_update(model, optimizer, [])
        self.assertIsNone(empty_metrics["policy_loss"])
        self.assertEqual(empty_metrics["games_used"], 0)
        self.assertEqual(optimizer.step_calls, 0)

    def test_batchnorm_buffers_and_modes_are_preserved_during_update(self):
        model = LogitPolicy()
        model.train()
        original_modes = {id(module): module.training for module in model.modules()}
        original_buffers = {
            name: value.detach().clone() for name, value in model.named_buffers()
        }
        optimizer = CountingSGD(model.parameters(), lr=0.1)

        before = model.logits.detach().clone()
        reinforce_policy_update(
            model,
            optimizer,
            [make_trajectory([make_step(target=1.0)])],
        )

        self.assertFalse(torch.equal(model.logits.detach(), before))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, original_buffers[name]), name)
        for module in model.modules():
            self.assertEqual(module.training, original_modes[id(module)])

    def test_none_baseline_matches_old_behavior_and_zero_baseline(self):
        trajectory = make_trajectory([make_step(target=1.0), make_step(target=1.0)])
        without_baseline = LogitPolicy()
        with_zero_baseline = LogitPolicy()
        with_zero_baseline.load_state_dict(without_baseline.state_dict())
        optimizer_none = CountingSGD(without_baseline.parameters(), lr=0.1)
        optimizer_zero = CountingSGD(with_zero_baseline.parameters(), lr=0.1)

        metrics_none = reinforce_policy_update(
            without_baseline, optimizer_none, [trajectory]
        )
        metrics_zero = reinforce_policy_update(
            with_zero_baseline,
            optimizer_zero,
            [trajectory],
            value_model=ScalarValueModel(0.0),
        )

        self.assertAlmostEqual(
            metrics_none["policy_loss"], metrics_zero["policy_loss"], places=7
        )
        for left, right in zip(
            without_baseline.parameters(), with_zero_baseline.parameters()
        ):
            self.assertTrue(torch.allclose(left, right, atol=1e-7, rtol=1e-7))

    def test_baseline_equal_target_produces_zero_policy_gradient(self):
        model = LogitPolicy()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        metrics = reinforce_policy_update(
            model,
            optimizer,
            [make_trajectory([make_step(target=1.0)])],
            value_model=ScalarValueModel(1.0),
        )

        self.assertAlmostEqual(metrics["policy_loss"], 0.0, places=7)
        self.assertEqual(optimizer.step_calls, 1)
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()))
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))

    def test_positive_and_negative_advantages_reverse_update_direction(self):
        positive_model = LogitPolicy()
        positive_optimizer = CountingSGD(positive_model.parameters(), lr=0.1)
        reinforce_policy_update(
            positive_model,
            positive_optimizer,
            [make_trajectory([make_step(target=1.0)])],
            value_model=ScalarValueModel(0.5),
        )
        self.assertGreater(positive_model.logits[0].item(), 0.0)
        self.assertLess(positive_model.logits[1].item(), 0.0)

        negative_model = LogitPolicy()
        negative_optimizer = CountingSGD(negative_model.parameters(), lr=0.1)
        reinforce_policy_update(
            negative_model,
            negative_optimizer,
            [make_trajectory([make_step(target=0.0)])],
            value_model=ScalarValueModel(0.5),
        )
        self.assertLess(negative_model.logits[0].item(), 0.0)
        self.assertGreater(negative_model.logits[1].item(), 0.0)

    def test_value_baseline_is_frozen_and_mode_is_restored(self):
        policy = LogitPolicy()
        optimizer = CountingSGD(policy.parameters(), lr=0.1)
        value_model = ScalarValueModel(0.25)
        value_model.train()
        value_model.bn.eval()
        original_modes = {
            id(module): module.training for module in value_model.modules()
        }
        original_parameters = [
            parameter.detach().clone() for parameter in value_model.parameters()
        ]
        original_buffers = {
            name: value.detach().clone()
            for name, value in value_model.named_buffers()
        }

        reinforce_policy_update(
            policy,
            optimizer,
            [make_trajectory([make_step(target=1.0)])],
            value_model=value_model,
        )

        for old, new in zip(original_parameters, value_model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()))
            self.assertIsNone(new.grad)
        for name, value in value_model.named_buffers():
            self.assertTrue(torch.equal(value, original_buffers[name]), name)
        for module in value_model.modules():
            self.assertEqual(module.training, original_modes[id(module)])

    def test_chunked_and_unchunked_baseline_updates_are_equivalent(self):
        trajectories = [
            make_trajectory([make_step(target=1.0)]),
            make_trajectory(
                [make_step(target=-1.0), make_step(target=-1.0)]
            ),
        ]
        unchunked = LogitPolicy()
        chunked = LogitPolicy()
        chunked.load_state_dict(unchunked.state_dict())
        optimizer_unchunked = CountingSGD(unchunked.parameters(), lr=0.05)
        optimizer_chunked = CountingSGD(chunked.parameters(), lr=0.05)

        unchunked_metrics = reinforce_policy_update(
            unchunked,
            optimizer_unchunked,
            trajectories,
            value_model=ScalarValueModel(0.25),
        )
        chunked_metrics = reinforce_policy_update(
            chunked,
            optimizer_chunked,
            trajectories,
            chunk_size=1,
            value_model=ScalarValueModel(0.25),
        )

        self.assertAlmostEqual(
            unchunked_metrics["policy_loss"],
            chunked_metrics["policy_loss"],
            places=7,
        )
        self.assertEqual(optimizer_unchunked.step_calls, 1)
        self.assertEqual(optimizer_chunked.step_calls, 1)
        for left, right in zip(unchunked.parameters(), chunked.parameters()):
            self.assertTrue(torch.allclose(left, right, atol=1e-7, rtol=1e-7))

    def test_nonfinite_baseline_aborts_policy_step(self):
        policy = LogitPolicy()
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        optimizer = CountingSGD(policy.parameters(), lr=0.1)
        value_model = NonFiniteValueModel()
        original_mode = value_model.training

        with self.assertRaisesRegex(ValueError, "Non-finite value baseline"):
            reinforce_policy_update(
                policy,
                optimizer,
                [make_trajectory([make_step(target=1.0)])],
                value_model=value_model,
            )

        self.assertEqual(optimizer.step_calls, 0)
        self.assertEqual(value_model.training, original_mode)
        for old, new in zip(before, policy.parameters()):
            self.assertTrue(torch.equal(old, new.detach()))

class TestStage5ValueUpdate(unittest.TestCase):
    def test_value_batch_uses_at_most_one_sample_per_terminal_game(self):
        trajectories = [
            make_trajectory(
                [
                    make_step(target=1.0, state_value=10.0),
                    make_step(target=1.0, state_value=11.0),
                ]
            ),
            make_trajectory(
                [
                    make_step(target=-1.0, state_value=20.0),
                    make_step(target=-1.0, state_value=21.0),
                    make_step(target=-1.0, state_value=22.0),
                ]
            ),
            make_trajectory([make_step(target=0.0, state_value=30.0)]),
            make_trajectory([make_step(target=None)], truncated=True),
        ]
        batch = build_value_batch(trajectories, seed=9)

        self.assertEqual(batch.states.shape, (3, 14, 8, 8))
        self.assertEqual(batch.states.dtype, torch.float32)
        self.assertEqual(batch.targets.shape, (3, 1))
        self.assertEqual(batch.targets.dtype, torch.float32)
        self.assertEqual(batch.targets[:, 0].tolist(), [1.0, -1.0, 0.0])
        self.assertIn(batch.states[0, 0, 0, 0].item(), (10.0, 11.0))
        self.assertIn(batch.states[1, 0, 0, 0].item(), (20.0, 21.0, 22.0))
        self.assertEqual(batch.states[2, 0, 0, 0].item(), 30.0)
        self.assertEqual(batch.games_used, 3)
        self.assertEqual(batch.games_skipped, 1)
        self.assertEqual(batch.games_truncated, 1)

    def test_value_batch_seed_is_reproducible_without_global_rng_changes(self):
        trajectories = [
            make_trajectory(
                [make_step(state_value=float(index)) for index in range(5)]
            )
            for _ in range(4)
        ]
        numpy_state = np.random.get_state()
        numpy_state_copy = (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
        torch_state = torch.random.get_rng_state().clone()

        first = build_value_batch(trajectories, seed=42)
        second = build_value_batch(trajectories, seed=42)

        self.assertTrue(torch.equal(first.states, second.states))
        self.assertTrue(torch.equal(first.targets, second.targets))
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state_copy[0])
        np.testing.assert_array_equal(current_numpy_state[1], numpy_state_copy[1])
        self.assertEqual(current_numpy_state[2:], numpy_state_copy[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))

    def test_empty_value_batch_does_not_step(self):
        batch = build_value_batch(
            [make_trajectory([make_step(target=None)], truncated=True)], seed=1
        )
        self.assertEqual(batch.states.shape, (0, 14, 8, 8))
        self.assertEqual(batch.targets.shape, (0, 1))
        model = ScalarValueModel()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        metrics = update_value_network(model, optimizer, batch)
        self.assertIsNone(metrics["value_loss"])
        self.assertEqual(metrics["samples_used"], 0)
        self.assertEqual(optimizer.step_calls, 0)

    def test_mse_formula_gradient_weight_update_and_mode_restore(self):
        batch = ValueBatch(
            states=torch.zeros((2, 14, 8, 8), dtype=torch.float32),
            targets=torch.tensor([[-1.0], [1.0]], dtype=torch.float32),
            games_used=2,
            games_skipped=0,
            games_truncated=0,
        )
        model = ScalarValueModel(initial_value=0.5)
        model.eval()
        original_modes = {id(module): module.training for module in model.modules()}
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        before = model.value.detach().clone()

        metrics = update_value_network(model, optimizer, batch)

        self.assertAlmostEqual(metrics["value_loss"], 1.25, places=6)
        self.assertEqual(metrics["samples_used"], 2)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertTrue(torch.isfinite(model.value.grad).all())
        self.assertFalse(torch.equal(model.value.detach(), before))
        for module in model.modules():
            self.assertEqual(module.training, original_modes[id(module)])

    def test_invalid_value_target_is_rejected_without_step(self):
        batch = ValueBatch(
            states=torch.zeros((1, 14, 8, 8), dtype=torch.float32),
            targets=torch.tensor([[0.5]], dtype=torch.float32),
            games_used=1,
            games_skipped=0,
            games_truncated=0,
        )
        model = ScalarValueModel()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "only -1, 0, or 1"):
            update_value_network(model, optimizer, batch)
        self.assertEqual(optimizer.step_calls, 0)

    def test_real_value_network_one_small_update(self):
        from models.value_net import ValueNetwork

        batch = ValueBatch(
            states=torch.zeros((2, 14, 8, 8), dtype=torch.float32),
            targets=torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
            games_used=2,
            games_skipped=0,
            games_truncated=0,
        )
        model = ValueNetwork(hidden_dim=4, conv2_dim=4, fc1_dim=8, fc2_dim=4)
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        metrics = update_value_network(model, optimizer, batch)

        self.assertTrue(np.isfinite(metrics["value_loss"]))
        self.assertEqual(metrics["samples_used"], 2)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )
