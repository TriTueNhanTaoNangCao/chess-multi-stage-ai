"""Stage 5 configuration, orchestration, and opponent-pool regressions."""
from tests.stage5.helpers import *
from training.stage5_train_rl import (
    _collect_trajectory_batch_with_opponent_pool,
    _snapshot_policy,
)

class TestStage5Trainer(unittest.TestCase):
    def test_mixed_start_integrates_with_opponent_pool(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                manifest = root / "mixed.json"
                write_small_stage4_checkpoint(source)
                write_mixed_start_manifest(manifest)
                result = train(
                    Stage5TrainConfig(
                        stage4_checkpoint=source,
                        output_dir=root / "stage5",
                        iterations=1,
                        games_per_iteration=32,
                        max_plies=0,
                        baseline="zero",
                        value_mode="disabled",
                        opponent_mode="pool",
                        opponent_snapshot_interval=1,
                        opponent_pool_size=2,
                        start_mode="mixed",
                        mixed_start_manifest=manifest,
                    )
                )
                row = result.history[0]
                self.assertEqual(
                    row["start_games"],
                    {"standard": 16, "pgn": 12, "mate_in_one": 4},
                )
                self.assertEqual(sum(row["opponent_games"].values()), 32)
                self.assertEqual(
                    row["opponent_pool_ids"],
                    ["stage4", "learner_iter_000001"],
                )

    def test_config_validation(self):
            with self.assertRaisesRegex(ValueError, "iterations"):
                Stage5TrainConfig("unused.pt", iterations=0).validate()
            with self.assertRaisesRegex(ValueError, "baseline"):
                Stage5TrainConfig("unused.pt", baseline="unknown").validate()
            with self.assertRaisesRegex(ValueError, "value_mode"):
                Stage5TrainConfig("unused.pt", value_mode="unknown").validate()
            with self.assertRaisesRegex(ValueError, "requires baseline='zero'"):
                Stage5TrainConfig("unused.pt", value_mode="disabled").validate()
            with self.assertRaisesRegex(ValueError, "requires baseline='value'"):
                Stage5TrainConfig(
                    "unused.pt", value_mode="frozen", baseline="zero"
                ).validate()
            with self.assertRaisesRegex(ValueError, "requires value_checkpoint"):
                Stage5TrainConfig(
                    "unused.pt", value_mode="frozen", baseline="value"
                ).validate()
            with self.assertRaisesRegex(ValueError, "only valid"):
                Stage5TrainConfig(
                    "unused.pt", value_checkpoint="unused-value.pt"
                ).validate()
            with self.assertRaisesRegex(ValueError, "reject value_checkpoint"):
                Stage5TrainConfig(
                    resume_checkpoint="resume.pt",
                    value_mode="frozen",
                    baseline="value",
                    value_checkpoint="replacement.pt",
                ).validate()
            with self.assertRaisesRegex(ValueError, "policy_lr"):
                Stage5TrainConfig("unused.pt", policy_lr=0.0).validate()
            with self.assertRaisesRegex(ValueError, "opponent_mode"):
                Stage5TrainConfig("unused.pt", opponent_mode="adaptive").validate()
            with self.assertRaisesRegex(ValueError, "opponent_pool_size"):
                Stage5TrainConfig("unused.pt", opponent_pool_size=0).validate()
            with self.assertRaisesRegex(ValueError, "start_mode"):
                Stage5TrainConfig("unused.pt", start_mode="unknown").validate()
            with self.assertRaisesRegex(ValueError, "games_per_iteration=32"):
                Stage5TrainConfig(
                    "unused.pt",
                    start_mode="mixed",
                    mixed_start_manifest="missing.json",
                ).validate()
            with self.assertRaisesRegex(ValueError, "mixed_start_manifest"):
                Stage5TrainConfig(
                    "unused.pt",
                    games_per_iteration=32,
                    start_mode="mixed",
                ).validate()

    def test_orchestration_order_fixed_opponent_and_continuous_rng(self):
            import training.stage5.trainer as stage5

            learner = LogitPolicy()
            opponent = LogitPolicy()
            opponent_state = {
                key: value.detach().clone() for key, value in opponent.state_dict().items()
            }
            metadata = {"claim_draw": True}
            events = []
            collected_batches = []
            move_draws = []
            move_rng_ids = []
            value_rng_ids = []
            value_rng_states = []
            policy_baselines = []
            value_after_updates = []
            original_policy_update = stage5.reinforce_policy_update
            original_build_value_batch = stage5.build_value_batch
            original_value_update = stage5.update_value_network

            def terminal_game(color, winner, target, marker):
                return GameTrajectory(
                    steps=[
                        make_step(target=target, state_value=marker),
                        make_step(target=target, state_value=marker + 0.5),
                    ],
                    learner_color=color,
                    truncated=False,
                    termination_reason="checkmate" if winner else "stalemate",
                    winner_white=winner,
                    plies_played=4,
                )

            def fake_collect(*args, **kwargs):
                iteration = len(collected_batches)
                rng = kwargs["rng"]
                move_rng_ids.append(id(rng))
                move_draws.append(float(torch.rand((), generator=rng)))
                marker = float(iteration * 10)
                batch = [
                    terminal_game(1, 1, 1.0, marker + 1),
                    terminal_game(-1, -1, 1.0, marker + 2),
                    terminal_game(-1, 1, -1.0, marker + 3),
                    terminal_game(1, 0, 0.0, marker + 4),
                    GameTrajectory(
                        steps=[make_step(target=None, state_value=marker + 5)],
                        learner_color=1,
                        truncated=True,
                        termination_reason="max_plies",
                        winner_white=None,
                        plies_played=1,
                    ),
                ]
                collected_batches.append(batch)
                events.append("collect")
                return batch

            def policy_update_wrapper(*args, **kwargs):
                self.assertIs(args[2], collected_batches[-1])
                policy_baselines.append(float(kwargs["value_model"].value.detach()))
                events.append("policy_update")
                return original_policy_update(*args, **kwargs)

            def build_value_wrapper(*args, **kwargs):
                rng = kwargs["rng"]
                value_rng_ids.append(id(rng))
                value_rng_states.append(repr(rng.bit_generator.state))
                events.append("value_batch")
                return original_build_value_batch(*args, **kwargs)

            def value_update_wrapper(*args, **kwargs):
                events.append("value_update")
                result = original_value_update(*args, **kwargs)
                value_after_updates.append(float(args[0].value.detach()))
                return result

            def save_wrapper(*args, **kwargs):
                events.append("save")

            config = Stage5TrainConfig(
                stage4_checkpoint="mock-stage4.pt",
                seed=17,
                iterations=2,
                games_per_iteration=5,
                max_plies=4,
                chunk_size=2,
                policy_lr=0.01,
                value_lr=0.01,
                baseline="value",
            )
            self.assertEqual(config.value_mode, "joint")
            with (
                patch.object(
                    stage5,
                    "load_checkpoint",
                    side_effect=[(learner, metadata), (opponent, metadata)],
                ),
                patch.object(stage5, "ValueNetwork", ScalarValueModel),
                patch.object(
                    stage5, "_collect_trajectory_batch_with_rng", side_effect=fake_collect
                ),
                patch.object(
                    stage5, "reinforce_policy_update", side_effect=policy_update_wrapper
                ),
                patch.object(
                    stage5, "build_value_batch", side_effect=build_value_wrapper
                ),
                patch.object(
                    stage5, "update_value_network", side_effect=value_update_wrapper
                ),
                patch.object(
                    stage5, "save_stage5_checkpoint", side_effect=save_wrapper
                ),
            ):
                result = train(config)

            self.assertEqual(
                events,
                [
                    "collect",
                    "policy_update",
                    "value_batch",
                    "value_update",
                    "save",
                    "collect",
                    "policy_update",
                    "value_batch",
                    "value_update",
                    "save",
                ],
            )
            self.assertIsNot(collected_batches[0], collected_batches[1])
            self.assertEqual(len({id(batch) for batch in collected_batches}), 2)
            self.assertEqual(len(set(move_rng_ids)), 1)
            expected_rng = torch.Generator(device="cpu")
            expected_rng.manual_seed(17)
            expected_draws = [float(torch.rand((), generator=expected_rng)) for _ in range(2)]
            self.assertEqual(move_draws, expected_draws)
            self.assertEqual(len(set(value_rng_ids)), 1)
            self.assertNotEqual(value_rng_states[0], value_rng_states[1])
            self.assertEqual(policy_baselines[0], 0.0)
            self.assertAlmostEqual(policy_baselines[1], value_after_updates[0], places=7)
            self.assertNotEqual(value_after_updates[0], policy_baselines[0])
            for record in result.history:
                self.assertEqual(
                    (record["wins"], record["draws"], record["losses"], record["truncated"]),
                    (2, 1, 1, 1),
                )
                self.assertEqual(record["games"], 5)
                self.assertEqual(record["decisions"], 8)
                self.assertEqual(record["value_samples_used"], 4)
            for key, value in result.opponent_model.state_dict().items():
                self.assertTrue(torch.equal(value, opponent_state[key]), key)
            self.assertTrue(all(not parameter.requires_grad for parameter in opponent.parameters()))

    def test_disabled_value_is_dormant_while_policy_updates(self):
            import training.stage5.trainer as stage5
            from models.value_net import ValueNetwork

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                output = root / "stage5"
                write_small_stage4_checkpoint(source)
                source_payload = torch.load(
                    source, map_location="cpu", weights_only=False
                )
                value_model = ValueNetwork(
                    hidden_dim=4, conv2_dim=4, fc1_dim=8, fc2_dim=4
                )
                value_before = {
                    key: tensor.detach().clone()
                    for key, tensor in value_model.state_dict().items()
                }
                trajectory = make_trajectory([make_step(target=1.0)])

                with (
                    patch.object(stage5, "ValueNetwork", return_value=value_model),
                    patch.object(
                        stage5,
                        "_collect_trajectory_batch_with_rng",
                        return_value=[trajectory],
                    ),
                    patch.object(
                        stage5,
                        "build_value_batch",
                        side_effect=AssertionError("Value batch must stay disabled"),
                    ),
                    patch.object(
                        stage5,
                        "update_value_network",
                        side_effect=AssertionError("Value update must stay disabled"),
                    ),
                ):
                    result = train(
                        Stage5TrainConfig(
                            stage4_checkpoint=source,
                            output_dir=output,
                            iterations=1,
                            games_per_iteration=1,
                            max_plies=1,
                            policy_lr=0.01,
                            value_lr=0.01,
                            baseline="zero",
                            value_mode="disabled",
                        )
                    )

                self.assertIsNone(result.history[0]["value_loss"])
                self.assertEqual(result.history[0]["value_samples_used"], 0)
                for key, tensor in result.value_model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, value_before[key]), key)
                self.assertTrue(
                    any(
                        not torch.equal(tensor, source_payload["state_dict"][key])
                        for key, tensor in result.policy_model.state_dict().items()
                    )
                )
                payload = torch.load(
                    output / "last.pt", map_location="cpu", weights_only=False
                )
                self.assertEqual(payload["value_mode"], "disabled")
                self.assertEqual(
                    payload["training_config"]["value_mode"], "disabled"
                )
                self.assertFalse(payload["value_optimizer_state_dict"]["state"])

    def test_frozen_value_is_loaded_used_and_never_updated(self):
            import training.stage5.trainer as stage5

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                value_checkpoint = root / "value-only.pt"
                output = root / "stage5"
                write_small_stage4_checkpoint(source)
                write_value_only_checkpoint(value_checkpoint)
                source_policy = torch.load(
                    source, map_location="cpu", weights_only=False
                )["state_dict"]
                source_value = torch.load(
                    value_checkpoint, map_location="cpu", weights_only=False
                )["state_dict"]
                trajectory = make_trajectory([make_step(target=1.0)])
                baselines = []
                original_policy_update = stage5.reinforce_policy_update

                def capture_baseline(*args, **kwargs):
                    baselines.append(kwargs["value_model"])
                    return original_policy_update(*args, **kwargs)

                with (
                    patch.object(
                        stage5,
                        "_collect_trajectory_batch_with_rng",
                        return_value=[trajectory],
                    ),
                    patch.object(
                        stage5,
                        "reinforce_policy_update",
                        side_effect=capture_baseline,
                    ),
                    patch.object(
                        stage5,
                        "build_value_batch",
                        side_effect=AssertionError(
                            "Frozen Value must not build a training batch"
                        ),
                    ),
                    patch.object(
                        stage5,
                        "update_value_network",
                        side_effect=AssertionError(
                            "Frozen Value must not perform an optimizer step"
                        ),
                    ),
                ):
                    result = train(
                        Stage5TrainConfig(
                            stage4_checkpoint=source,
                            value_checkpoint=value_checkpoint,
                            output_dir=output,
                            iterations=1,
                            games_per_iteration=1,
                            max_plies=1,
                            policy_lr=0.01,
                            value_lr=0.01,
                            baseline="value",
                            value_mode="frozen",
                        )
                    )

                self.assertEqual(baselines, [result.value_model])
                self.assertFalse(result.value_model.training)
                self.assertTrue(
                    all(
                        not parameter.requires_grad
                        for parameter in result.value_model.parameters()
                    )
                )
                self.assertTrue(
                    all(
                        parameter.grad is None
                        for parameter in result.value_model.parameters()
                    )
                )
                for key, tensor in result.value_model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, source_value[key]), key)
                self.assertTrue(
                    any(
                        not torch.equal(tensor, source_policy[key])
                        for key, tensor in result.policy_model.state_dict().items()
                    )
                )
                self.assertIsNone(result.history[0]["value_loss"])
                self.assertEqual(result.history[0]["value_samples_used"], 0)
                payload = torch.load(
                    output / "last.pt", map_location="cpu", weights_only=False
                )
                self.assertEqual(payload["value_mode"], "frozen")
                self.assertEqual(
                    payload["value_source"]["checkpoint_type"], "value_only"
                )
                self.assertEqual(
                    len(payload["value_source"]["sha256"]), 64
                )
                self.assertFalse(payload["value_optimizer_state_dict"]["state"])

    def test_all_truncated_finishes_and_stage4_checkpoint_is_unchanged(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = Path(temp_dir) / "stage4-small.pt"
                write_small_stage4_checkpoint(checkpoint_path)
                checkpoint_before = checkpoint_path.read_bytes()
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                config = Stage5TrainConfig(
                    stage4_checkpoint=checkpoint_path,
                    output_dir=Path(temp_dir) / "stage5",
                    seed=3,
                    iterations=2,
                    games_per_iteration=2,
                    max_plies=0,
                    policy_lr=0.01,
                    value_lr=0.01,
                    baseline="value",
                )

                result = train(config)

                self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)
                self.assertEqual(len(result.history), 2)
                for record in result.history:
                    self.assertEqual(record["truncated"], 2)
                    self.assertEqual(record["wins"], 0)
                    self.assertEqual(record["draws"], 0)
                    self.assertEqual(record["losses"], 0)
                    self.assertEqual(record["decisions"], 0)
                    self.assertIsNone(record["policy_loss"])
                    self.assertIsNone(record["value_loss"])
                for model in (result.policy_model, result.opponent_model):
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, payload["state_dict"][key]), key)

    def test_opponent_snapshot_is_frozen_and_has_independent_storage(self):
            from models.cnn_policy import CNNPolicyNetwork

            learner = CNNPolicyNetwork(hidden_dim=2, fc_dim=8, dropout=0.0)
            snapshot = _snapshot_policy(
                learner,
                opponent_id="learner_iter_000001",
                iteration=1,
                device=torch.device("cpu"),
            )
            self.assertEqual(snapshot.opponent_id, "learner_iter_000001")
            self.assertFalse(snapshot.model.training)
            learner_parameters = dict(learner.named_parameters())
            snapshot_parameters = dict(snapshot.model.named_parameters())
            frozen_before = {
                name: value.detach().clone()
                for name, value in snapshot_parameters.items()
            }
            for name, learner_parameter in learner_parameters.items():
                frozen_parameter = snapshot_parameters[name]
                self.assertTrue(torch.equal(learner_parameter, frozen_parameter))
                self.assertNotEqual(
                    learner_parameter.untyped_storage().data_ptr(),
                    frozen_parameter.untyped_storage().data_ptr(),
                )
                self.assertFalse(frozen_parameter.requires_grad)
            with torch.no_grad():
                next(iter(learner.parameters())).add_(1.0)
            for name, frozen_parameter in snapshot.model.named_parameters():
                self.assertTrue(torch.equal(frozen_parameter, frozen_before[name]))

    def test_opponent_selection_rng_is_reproducible(self):
            pool = [
                OpponentSnapshot("stage4", 0, TinyPolicy()),
                OpponentSnapshot("learner_iter_000001", 1, TinyPolicy()),
                OpponentSnapshot("learner_iter_000002", 2, TinyPolicy()),
            ]

            def collect_counts(seed):
                move_rng = torch.Generator(device="cpu")
                move_rng.manual_seed(999)
                trajectories, counts = _collect_trajectory_batch_with_opponent_pool(
                    TinyPolicy(),
                    pool,
                    30,
                    device=torch.device("cpu"),
                    move_rng=move_rng,
                    opponent_rng=np.random.default_rng(seed),
                    first_learner_color=chess.WHITE,
                    max_plies=0,
                    claim_draw=True,
                    initial_board=None,
                )
                self.assertEqual(len(trajectories), 30)
                return counts

            first = collect_counts(2026)
            second = collect_counts(2026)
            self.assertEqual(first, second)
            self.assertEqual(sum(first.values()), 30)
            self.assertGreater(sum(count > 0 for count in first.values()), 1)

    def test_pool_limit_and_resume_restore_order_history_and_rng(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                write_small_stage4_checkpoint(source)

                def config(output, iterations, resume=None):
                    return Stage5TrainConfig(
                        stage4_checkpoint=None if resume else source,
                        output_dir=output,
                        resume_checkpoint=resume,
                        seed=19,
                        iterations=iterations,
                        games_per_iteration=6,
                        max_plies=0,
                        policy_lr=0.001,
                        value_lr=0.001,
                        baseline="value",
                        opponent_mode="pool",
                        opponent_snapshot_interval=1,
                        opponent_pool_size=3,
                    )

                continuous_dir = root / "continuous_pool"
                continuous = train(config(continuous_dir, 5))
                interrupted_dir = root / "interrupted_pool"
                train(config(interrupted_dir, 3))
                resumed = train(
                    config(
                        interrupted_dir,
                        5,
                        interrupted_dir / "last.pt",
                    )
                )

                expected_ids = ["stage4", "learner_iter_000004", "learner_iter_000005"]
                self.assertEqual(
                    [entry.opponent_id for entry in continuous.opponent_pool],
                    expected_ids,
                )
                self.assertEqual(
                    [entry.opponent_id for entry in resumed.opponent_pool],
                    expected_ids,
                )
                self.assertEqual(continuous.history, resumed.history)
                for record in resumed.history:
                    self.assertEqual(sum(record["opponent_games"].values()), 6)
                    self.assertIn("stage4", record["opponent_pool_ids"])
                    self.assertLessEqual(len(record["opponent_pool_ids"]), 3)
                for left, right in zip(
                    continuous.opponent_pool, resumed.opponent_pool
                ):
                    self.assertEqual(left.opponent_id, right.opponent_id)
                    self.assertEqual(left.iteration, right.iteration)
                    for key, value in left.model.state_dict().items():
                        self.assertTrue(
                            torch.equal(value, right.model.state_dict()[key]), key
                        )
                continuous_payload = torch.load(
                    continuous_dir / "last.pt", map_location="cpu", weights_only=False
                )
                resumed_payload = torch.load(
                    interrupted_dir / "last.pt", map_location="cpu", weights_only=False
                )
                self.assertEqual(continuous_payload["opponent_mode"], "pool")
                self.assertEqual(
                    [entry["id"] for entry in resumed_payload["opponent_pool"]],
                    expected_ids,
                )
                self.assertEqual(
                    repr(continuous_payload["opponent_rng_state"]),
                    repr(resumed_payload["opponent_rng_state"]),
                )
