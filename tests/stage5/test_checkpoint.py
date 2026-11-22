"""Stage 5 checkpoint and resume regressions."""
import copy

from tests.stage5.helpers import *

class TestStage5Checkpoint(unittest.TestCase):
    def test_stage5_checkpoint_roundtrip_and_bad_metadata(self):
            from models.cnn_policy import CNNPolicyNetwork
            from models.value_net import ValueNetwork

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                checkpoint_path = root / "last.pt"
                policy = CNNPolicyNetwork(hidden_dim=2, fc_dim=8, dropout=0.0)
                opponent = CNNPolicyNetwork.from_config(policy.get_config())
                opponent.load_state_dict(policy.state_dict(), strict=True)
                value = ValueNetwork(hidden_dim=4, conv2_dim=4, fc1_dim=8, fc2_dim=4)
                policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
                value_optimizer = torch.optim.Adam(value.parameters(), lr=0.02)
                for parameter in policy.parameters():
                    parameter.grad = torch.full_like(parameter, 0.01)
                for parameter in value.parameters():
                    parameter.grad = torch.full_like(parameter, 0.02)
                policy_optimizer.step()
                value_optimizer.step()
                policy_optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)
                collector_rng = torch.Generator(device="cpu")
                collector_rng.manual_seed(7)
                torch.rand((), generator=collector_rng)
                value_rng = np.random.default_rng(7)
                value_rng.integers(100)
                config = Stage5TrainConfig(
                    stage4_checkpoint="source.pt",
                    output_dir=root,
                    seed=7,
                    iterations=1,
                    policy_lr=0.01,
                    value_lr=0.02,
                )

                save_stage5_checkpoint(
                    checkpoint_path,
                    policy_model=policy,
                    value_model=value,
                    opponent_model=opponent,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    config=config,
                    completed_iteration=1,
                    history=[{"iteration": 1}],
                    source_checkpoint={"path": "source.pt", "stage": 4},
                    collector_rng=collector_rng,
                    value_rng=value_rng,
                    claim_draw=True,
                )
                loaded = load_stage5_checkpoint(checkpoint_path, device="cpu")

                self.assertEqual(loaded.payload["stage"], 5)
                self.assertEqual(loaded.payload["completed_iteration"], 1)
                self.assertEqual(loaded.payload["value_mode"], "joint")
                self.assertEqual(
                    loaded.payload["training_config"]["value_mode"], "joint"
                )
                self.assertEqual(loaded.payload["encoder_version"], "hwc14-white-v1")
                for original, restored in (
                    (policy, loaded.policy_model),
                    (value, loaded.value_model),
                    (opponent, loaded.opponent_model),
                ):
                    for key, tensor in original.state_dict().items():
                        self.assertTrue(
                            torch.equal(tensor, restored.state_dict()[key]), key
                        )
                self.assertTrue(
                    loaded.policy_optimizer.state_dict()["state"]
                )
                self.assertTrue(loaded.value_optimizer.state_dict()["state"])
                self.assertFalse(loaded.opponent_model.training)
                self.assertTrue(
                    all(
                        not parameter.requires_grad
                        for parameter in loaded.opponent_model.parameters()
                    )
                )
                self.assertTrue(
                    torch.equal(
                        loaded.payload["collector_rng_state"], collector_rng.get_state()
                    )
                )
                self.assertEqual(
                    repr(loaded.payload["value_rng_state"]),
                    repr(value_rng.bit_generator.state),
                )
                self.assertFalse((root / ".last.pt.tmp").exists())

                bad_path = root / "bad.pt"
                bad_payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                bad_payload["stage"] = 4
                torch.save(bad_payload, bad_path)
                with self.assertRaisesRegex(ValueError, "stage"):
                    load_stage5_checkpoint(bad_path)

    def test_checkpoint_errors_for_missing_existing_and_incompatible_resume(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                with self.assertRaises(FileNotFoundError):
                    load_stage5_checkpoint(root / "missing.pt")

                occupied = root / "occupied"
                occupied.mkdir()
                (occupied / "last.pt").write_bytes(b"already here")
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    train(
                        Stage5TrainConfig(
                            stage4_checkpoint=root / "unused.pt",
                            output_dir=occupied,
                        )
                    )

    def test_value_only_schema_validation_and_frozen_resume_without_source(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                value_source = root / "value-only.pt"
                output = root / "fresh"
                write_small_stage4_checkpoint(source)
                write_value_only_checkpoint(value_source)

                loaded_value, provenance = load_value_only_checkpoint(
                    value_source, device="cpu", claim_draw=True
                )
                self.assertFalse(loaded_value.training)
                self.assertTrue(
                    all(
                        not parameter.requires_grad
                        for parameter in loaded_value.parameters()
                    )
                )
                self.assertEqual(provenance["checkpoint_type"], "value_only")
                self.assertEqual(len(provenance["sha256"]), 64)

                original = torch.load(
                    value_source, map_location="cpu", weights_only=False
                )
                invalid_cases = {
                    "checkpoint_type": "rl_policy",
                    "encoder_version": "wrong-encoder",
                    "value_perspective": "white",
                    "claim_draw": False,
                    "model_config": {
                        **original["model_config"],
                        "in_channels": 13,
                    },
                }
                for key, bad_value in invalid_cases.items():
                    bad_payload = copy.deepcopy(original)
                    bad_payload[key] = bad_value
                    bad_path = root / f"bad-{key}.pt"
                    torch.save(bad_payload, bad_path)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        load_value_only_checkpoint(
                            bad_path, device="cpu", claim_draw=True
                        )

                train(
                    Stage5TrainConfig(
                        stage4_checkpoint=source,
                        value_checkpoint=value_source,
                        output_dir=output,
                        iterations=1,
                        games_per_iteration=1,
                        max_plies=0,
                        baseline="value",
                        value_mode="frozen",
                    )
                )
                stage5_checkpoint = output / "last.pt"
                loaded_stage5 = load_stage5_checkpoint(stage5_checkpoint)
                self.assertFalse(loaded_stage5.value_model.training)
                self.assertTrue(
                    all(
                        not parameter.requires_grad
                        for parameter in loaded_stage5.value_model.parameters()
                    )
                )
                embedded_value_state = {
                    key: tensor.detach().clone()
                    for key, tensor in loaded_stage5.value_model.state_dict().items()
                }
                value_source.unlink()

                resumed = train(
                    Stage5TrainConfig(
                        output_dir=root / "resumed",
                        resume_checkpoint=stage5_checkpoint,
                        iterations=2,
                        games_per_iteration=1,
                        max_plies=0,
                        baseline="value",
                        value_mode="frozen",
                    )
                )
                self.assertEqual(len(resumed.history), 2)
                self.assertFalse(resumed.value_model.training)
                for key, tensor in resumed.value_model.state_dict().items():
                    self.assertTrue(
                        torch.equal(tensor, embedded_value_state[key]), key
                    )
                resumed_payload = torch.load(
                    root / "resumed" / "last.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(resumed_payload["value_mode"], "frozen")
                self.assertEqual(
                    resumed_payload["value_source"]["sha256"],
                    loaded_stage5.payload["value_source"]["sha256"],
                )

                source = root / "stage4.pt"
                output = root / "stage5"
                write_small_stage4_checkpoint(source)
                train(
                    Stage5TrainConfig(
                        stage4_checkpoint=source,
                        output_dir=output,
                        iterations=1,
                        games_per_iteration=1,
                        max_plies=0,
                        baseline="zero",
                    )
                )
                with self.assertRaisesRegex(ValueError, "baseline"):
                    train(
                        Stage5TrainConfig(
                            output_dir=output,
                            resume_checkpoint=output / "last.pt",
                            iterations=2,
                            games_per_iteration=1,
                            max_plies=0,
                            baseline="value",
                        )
                    )
                with self.assertRaisesRegex(ValueError, "value_mode"):
                    train(
                        Stage5TrainConfig(
                            output_dir=output,
                            resume_checkpoint=output / "last.pt",
                            iterations=2,
                            games_per_iteration=1,
                            max_plies=0,
                            baseline="zero",
                            value_mode="disabled",
                        )
                    )

    def test_cpu_continuous_matches_interrupted_resume(self):
            import random
            import training.stage5.trainer as stage5

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                write_small_stage4_checkpoint(source)
                global_python_state = random.getstate()
                global_numpy_state = np.random.get_state()
                global_torch_state = torch.get_rng_state().clone()

                def fake_collect(*args, **kwargs):
                    rng = kwargs["rng"]
                    marker = float(torch.rand((), generator=rng))
                    color = 1 if kwargs["first_learner_color"] else -1
                    winner = 1 if color == 1 else -1
                    return [
                        GameTrajectory(
                            steps=[make_step(target=1.0, state_value=marker)],
                            learner_color=color,
                            truncated=False,
                            termination_reason="checkmate",
                            winner_white=winner,
                            plies_played=1,
                        )
                    ]

                continuous_dir = root / "continuous"
                with patch.object(
                    stage5, "_collect_trajectory_batch_with_rng", side_effect=fake_collect
                ):
                    continuous = train(
                        Stage5TrainConfig(
                            stage4_checkpoint=source,
                            output_dir=continuous_dir,
                            seed=23,
                            iterations=2,
                            games_per_iteration=1,
                            max_plies=1,
                            chunk_size=1,
                            policy_lr=0.001,
                            value_lr=0.001,
                            baseline="value",
                        )
                    )

                random.setstate(global_python_state)
                np.random.set_state(global_numpy_state)
                torch.set_rng_state(global_torch_state)
                resumed_dir = root / "resumed"
                with patch.object(
                    stage5, "_collect_trajectory_batch_with_rng", side_effect=fake_collect
                ):
                    train(
                        Stage5TrainConfig(
                            stage4_checkpoint=source,
                            output_dir=resumed_dir,
                            seed=23,
                            iterations=1,
                            games_per_iteration=1,
                            max_plies=1,
                            chunk_size=1,
                            policy_lr=0.001,
                            value_lr=0.001,
                            baseline="value",
                        )
                    )
                    resumed = train(
                        Stage5TrainConfig(
                            output_dir=resumed_dir,
                            resume_checkpoint=resumed_dir / "last.pt",
                            seed=23,
                            iterations=2,
                            games_per_iteration=1,
                            max_plies=1,
                            chunk_size=1,
                            policy_lr=0.001,
                            value_lr=0.001,
                            baseline="value",
                        )
                    )

                self.assertEqual(continuous.history, resumed.history)
                for continuous_model, resumed_model in (
                    (continuous.policy_model, resumed.policy_model),
                    (continuous.value_model, resumed.value_model),
                    (continuous.opponent_model, resumed.opponent_model),
                ):
                    for key, value in continuous_model.state_dict().items():
                        self.assertTrue(
                            torch.equal(value, resumed_model.state_dict()[key]), key
                        )
                continuous_payload = torch.load(
                    continuous_dir / "last.pt", map_location="cpu", weights_only=False
                )
                resumed_payload = torch.load(
                    resumed_dir / "last.pt", map_location="cpu", weights_only=False
                )
                self.assertEqual(continuous_payload["completed_iteration"], 2)
                self.assertEqual(resumed_payload["completed_iteration"], 2)
                self.assertTrue(
                    torch.equal(
                        continuous_payload["collector_rng_state"],
                        resumed_payload["collector_rng_state"],
                    )
                )
                self.assertEqual(
                    repr(continuous_payload["value_rng_state"]),
                    repr(resumed_payload["value_rng_state"]),
                )

    def test_mixed_start_resume_restores_source_rng_and_rejects_changed_manifest(self):
            import training.stage5.trainer as stage5

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                manifest = root / "mixed.json"
                write_small_stage4_checkpoint(source)
                write_mixed_start_manifest(manifest)

                def config(output, iterations, resume=None):
                    return Stage5TrainConfig(
                        stage4_checkpoint=None if resume else source,
                        output_dir=output,
                        resume_checkpoint=resume,
                        seed=29,
                        iterations=iterations,
                        games_per_iteration=32,
                        max_plies=0,
                        policy_lr=0.001,
                        value_lr=0.001,
                        baseline="zero",
                        value_mode="disabled",
                        start_mode="mixed",
                        mixed_start_manifest=manifest,
                    )

                continuous_dir = root / "continuous_mixed"
                original_collect = (
                    stage5._collect_trajectory_batch_from_starts_with_rng
                )
                continuous_schedule = []
                resumed_schedule = []

                def capture(store):
                    def wrapped(*args, **kwargs):
                        starts = args[2]
                        store.append(
                            [
                                (
                                    entry.source_type,
                                    entry.source_id,
                                    entry.learner_color,
                                )
                                for entry in starts
                            ]
                        )
                        return original_collect(*args, **kwargs)

                    return wrapped

                with patch.object(
                    stage5,
                    "_collect_trajectory_batch_from_starts_with_rng",
                    side_effect=capture(continuous_schedule),
                ):
                    continuous = train(config(continuous_dir, 2))
                resumed_dir = root / "resumed_mixed"
                with patch.object(
                    stage5,
                    "_collect_trajectory_batch_from_starts_with_rng",
                    side_effect=capture(resumed_schedule),
                ):
                    train(config(resumed_dir, 1))
                    resumed = train(
                        config(resumed_dir, 2, resumed_dir / "last.pt")
                    )
                self.assertEqual(continuous_schedule, resumed_schedule)
                self.assertEqual(continuous.history, resumed.history)
                self.assertEqual(
                    [row["start_games"] for row in resumed.history],
                    [
                        {"standard": 16, "pgn": 12, "mate_in_one": 4},
                        {"standard": 16, "pgn": 12, "mate_in_one": 4},
                    ],
                )
                continuous_payload = torch.load(
                    continuous_dir / "last.pt", map_location="cpu", weights_only=False
                )
                resumed_payload = torch.load(
                    resumed_dir / "last.pt", map_location="cpu", weights_only=False
                )
                self.assertEqual(continuous_payload["start_mode"], "mixed")
                self.assertEqual(
                    continuous_payload["start_source"],
                    resumed_payload["start_source"],
                )
                self.assertEqual(
                    repr(continuous_payload["start_rng_state"]),
                    repr(resumed_payload["start_rng_state"]),
                )
                self.assertNotIn("positions", resumed_payload["start_source"])

                changed = json.loads(manifest.read_text(encoding="utf-8"))
                changed["purpose"] = "changed"
                manifest.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest or mix"):
                    train(config(resumed_dir, 3, resumed_dir / "last.pt"))

    def test_legacy_fixed_checkpoint_stays_fixed(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "stage4.pt"
                output = root / "fixed"
                write_small_stage4_checkpoint(source)
                train(
                    Stage5TrainConfig(
                        stage4_checkpoint=source,
                        output_dir=output,
                        iterations=1,
                        games_per_iteration=1,
                        max_plies=0,
                        opponent_mode="fixed",
                    )
                )
                payload = torch.load(
                    output / "last.pt", map_location="cpu", weights_only=False
                )
                for key in (
                    "opponent_mode",
                    "opponent_pool",
                    "opponent_rng_state",
                    "value_mode",
                    "value_source",
                    "start_mode",
                    "start_source",
                    "start_rng_state",
                ):
                    payload.pop(key)
                for key in (
                    "opponent_mode",
                    "opponent_snapshot_interval",
                    "opponent_pool_size",
                    "value_mode",
                    "value_checkpoint",
                    "start_mode",
                    "mixed_start_manifest",
                ):
                    payload["training_config"].pop(key)
                legacy_path = root / "legacy_fixed.pt"
                torch.save(payload, legacy_path)

                loaded = load_stage5_checkpoint(legacy_path)
                self.assertEqual(loaded.opponent_mode, "fixed")
                self.assertEqual(loaded.payload["value_mode"], "joint")
                self.assertEqual(
                    loaded.payload["training_config"]["value_mode"], "joint"
                )
                self.assertEqual(
                    [entry.opponent_id for entry in loaded.opponent_pool], ["stage4"]
                )
                result = train(
                    Stage5TrainConfig(
                        output_dir=root / "legacy_resume",
                        resume_checkpoint=legacy_path,
                        iterations=1,
                        games_per_iteration=1,
                        max_plies=0,
                        opponent_mode="fixed",
                    )
                )
                self.assertEqual(
                    [entry.opponent_id for entry in result.opponent_pool], ["stage4"]
                )
                with self.assertRaisesRegex(ValueError, "opponent_mode"):
                    train(
                        Stage5TrainConfig(
                            output_dir=root / "wrong_mode",
                            resume_checkpoint=legacy_path,
                            iterations=2,
                            games_per_iteration=1,
                            max_plies=0,
                            opponent_mode="pool",
                        )
                    )
