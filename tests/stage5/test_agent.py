"""Stage 5 RLPolicyAgent regressions."""
from tests.stage5.helpers import *

class TestRLPolicyAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.cnn_policy import CNNPolicyNetwork
        from models.value_net import ValueNetwork

        cls.temp_context = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_context.name)
        cls.stage4_checkpoint = cls.root / "stage4.pt"
        write_small_stage4_checkpoint(cls.stage4_checkpoint)
        cls.checkpoint = cls.root / "last.pt"
        cls.saved_policy = CNNPolicyNetwork(hidden_dim=2, fc_dim=8, dropout=0.0)
        cls.saved_policy.eval()
        opponent = CNNPolicyNetwork.from_config(cls.saved_policy.get_config())
        opponent.load_state_dict(cls.saved_policy.state_dict(), strict=True)
        value = ValueNetwork(hidden_dim=4, conv2_dim=4, fc1_dim=8, fc2_dim=4)
        policy_optimizer = torch.optim.Adam(cls.saved_policy.parameters(), lr=0.001)
        value_optimizer = torch.optim.Adam(value.parameters(), lr=0.001)
        collector_rng = torch.Generator(device="cpu")
        collector_rng.manual_seed(31)
        value_rng = np.random.default_rng(31)
        config = Stage5TrainConfig(
            stage4_checkpoint=cls.stage4_checkpoint,
            output_dir=cls.root,
            seed=31,
            iterations=1,
            policy_lr=0.001,
            value_lr=0.001,
        )
        save_stage5_checkpoint(
            cls.checkpoint,
            policy_model=cls.saved_policy,
            value_model=value,
            opponent_model=opponent,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            config=config,
            completed_iteration=0,
            history=[],
            source_checkpoint={"path": str(cls.stage4_checkpoint), "stage": 4},
            collector_rng=collector_rng,
            value_rng=value_rng,
            claim_draw=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_context.cleanup()

    def test_legal_moves_both_colors_and_board_history_unchanged(self):
        agent = RLPolicyAgent(self.checkpoint, seed=99, name="Stage5Test")
        self.assertEqual(agent.name, "Stage5Test")
        white = ChessBoard()
        white_before = (white.fen(), list(white.board.move_stack))
        white_move = agent.act(white)
        self.assertIn(white_move, white.legal_moves)
        self.assertEqual((white.fen(), white.board.move_stack), white_before)

        black = ChessBoard()
        black.step("e4")
        black_before = (black.fen(), list(black.board.move_stack))
        black_move = agent.act(black)
        self.assertIn(black_move, black.legal_moves)
        self.assertEqual((black.fen(), black.board.move_stack), black_before)

    def test_terminal_rejected_and_saved_policy_matches(self):
        agent = RLPolicyAgent(self.checkpoint)
        terminal = ChessBoard("7k/8/8/8/8/8/8/K7 w - - 0 1")
        with self.assertRaises(ValueError):
            agent.act(terminal)

        state = ChessEncoder.encode_board(ChessBoard()).transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            expected = self.saved_policy(tensor)
            actual = agent.model(tensor)
        self.assertTrue(torch.equal(actual, expected))
        for key, value in self.saved_policy.state_dict().items():
            self.assertTrue(torch.equal(value, agent.model.state_dict()[key]), key)

    def test_constructor_does_not_change_global_rng(self):
        torch_state = torch.get_rng_state().clone()
        numpy_state = np.random.get_state()
        numpy_copy = (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
        RLPolicyAgent(self.checkpoint, seed=123)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_copy[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_copy[1])
        self.assertEqual(current_numpy[2:], numpy_copy[2:])

    def test_missing_corrupt_stage4_and_bad_metadata_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            RLPolicyAgent(self.root / "missing.pt")
        corrupt = self.root / "corrupt.pt"
        corrupt.write_bytes(b"not a torch checkpoint")
        with self.assertRaises(ValueError):
            RLPolicyAgent(corrupt)
        with self.assertRaisesRegex(ValueError, "stage"):
            RLPolicyAgent(self.stage4_checkpoint)

        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        bad_encoder = self.root / "bad_encoder.pt"
        torch.save(dict(payload, encoder_version="wrong"), bad_encoder)
        with self.assertRaisesRegex(ValueError, "encoder_version"):
            RLPolicyAgent(bad_encoder)
        missing_state = dict(payload)
        missing_state.pop("state_dict")
        missing_state_path = self.root / "missing_state.pt"
        torch.save(missing_state, missing_state_path)
        with self.assertRaisesRegex(ValueError, "missing fields"):
            RLPolicyAgent(missing_state_path)

    def test_factory_arena_parser_cli_and_defaults(self):
        self.assertIn("rl_policy", AGENT_REGISTRY)
        self.assertNotIn("rl_policy", CLASSIC_AGENTS)
        self.assertNotIn("rl_policy", DEFAULT_TOURNAMENT_AGENTS)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            build_agent("rl_policy")
        agent = build_agent(
            "rl_policy", checkpoint=str(self.checkpoint), seed=5, device="cpu"
        )
        self.assertIsInstance(agent, RLPolicyAgent)

        checkpoint_text = str(self.checkpoint.resolve())
        agents, config = build_cli_agents(
            [f"rl_policy:{checkpoint_text}", "alphabeta:2"], depth=1, seed=7
        )
        self.assertIsInstance(agents[0], RLPolicyAgent)
        self.assertEqual(config[0]["checkpoint"], checkpoint_text)
        self.assertEqual(agents[1].depth, 2)

        import main as main_module

        with patch.object(main_module, "run_cli", return_value=17) as run_cli:
            result = main_module.main(
                [
                    "--cli",
                    "--agent",
                    "rl_policy",
                    "--checkpoint",
                    checkpoint_text,
                ]
            )
        self.assertEqual(result, 17)
        self.assertEqual(run_cli.call_count, 1)

    def test_classic_agents_survive_simulated_missing_torch(self):
        import agents.stage5_rl.rl_agent as rl_module

        original_torch = rl_module.torch
        try:
            rl_module.torch = None
            classic = build_agent("random", seed=4)
            self.assertIn(classic.act(ChessBoard()), ChessBoard().legal_moves)
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                build_agent("rl_policy", checkpoint=str(self.checkpoint))
        finally:
            rl_module.torch = original_torch
