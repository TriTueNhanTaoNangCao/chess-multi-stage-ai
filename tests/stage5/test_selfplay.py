"""Stage 5 trajectory collector regressions."""
from collections import Counter
from tests.stage5.helpers import *
from training.stage5_train_rl import (
    _collect_trajectory,
    _collect_trajectory_batch_from_starts_with_rng,
)

class TestStage5TrajectoryCollection(unittest.TestCase):
    def test_mixed_start_schedule_colors_history_and_continuous_rng(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "mixed.json"
            write_mixed_start_manifest(manifest)
            source, metadata = load_mixed_start_manifest(
                manifest, claim_draw=True
            )
            self.assertEqual(metadata["mix"], {
                "standard": 16,
                "pgn": 12,
                "mate_in_one": 4,
            })

            rng = np.random.default_rng(2052)
            first = build_mixed_start_batch(source, rng=rng, claim_draw=True)
            second = build_mixed_start_batch(source, rng=rng, claim_draw=True)
            replay_rng = np.random.default_rng(2052)
            replay_first = build_mixed_start_batch(
                source, rng=replay_rng, claim_draw=True
            )
            replay_second = build_mixed_start_batch(
                source, rng=replay_rng, claim_draw=True
            )

            def signature(starts):
                return [
                    (entry.source_type, entry.source_id, entry.learner_color)
                    for entry in starts
                ]

            self.assertEqual(signature(first), signature(replay_first))
            self.assertEqual(signature(second), signature(replay_second))
            self.assertNotEqual(signature(first), signature(second))
            counts = Counter(entry.source_type for entry in first)
            self.assertEqual(
                counts,
                {"standard": 16, "pgn": 12, "mate_in_one": 4},
            )
            for source_type, expected in (("standard", 8), ("pgn", 6)):
                entries = [e for e in first if e.source_type == source_type]
                self.assertEqual(sum(e.learner_color == chess.WHITE for e in entries), expected)
                self.assertEqual(sum(e.learner_color == chess.BLACK for e in entries), expected)
            for entry in (e for e in first if e.source_type == "mate_in_one"):
                self.assertEqual(entry.learner_color, entry.board.turn)

            source_board = source["pgn"][1].board
            original_fen = source_board.fen()
            original_stack = [move.uci() for move in source_board.board.move_stack]
            trajectory = _collect_trajectory_batch_from_starts_with_rng(
                TinyPolicy(),
                TinyPolicy(),
                [source["pgn"][1]],
                device=torch.device("cpu"),
                rng=torch.Generator(device="cpu").manual_seed(7),
                max_plies=1,
                claim_draw=True,
            )[0]
            self.assertEqual(trajectory.steps[0].ply_index, 1)
            self.assertEqual(source_board.fen(), original_fen)
            self.assertEqual(
                [move.uci() for move in source_board.board.move_stack],
                original_stack,
            )

    def test_mixed_start_reward_comes_from_rollout_not_pgn_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "mixed.json"
            write_mixed_start_manifest(manifest)
            source, _ = load_mixed_start_manifest(manifest, claim_draw=True)
            start = source["pgn"][0]
            mate = chess.Move.from_uci("h5f7")
            action = ChessEncoder.encode_move(mate, start.board.turn)
            trajectory = _collect_trajectory(
                TinyPolicy([action]),
                TinyPolicy(),
                device=torch.device("cpu"),
                rng=torch.Generator(device="cpu").manual_seed(11),
                learner_color=start.board.turn,
                max_plies=1,
                claim_draw=True,
                initial_board=start.board,
            )
            self.assertEqual(trajectory.winner_white, 1)
            self.assertEqual([step.target for step in trajectory.steps], [1.0])
            self.assertEqual(trajectory.termination_reason, "checkmate")

    def test_state_before_move_action_mask_and_decode(self):
        board = ChessBoard()
        action = ChessEncoder.encode_move(chess.Move.from_uci("e2e4"), chess.WHITE)
        trajectory = collect_trajectory(
            TinyPolicy([action]),
            TinyPolicy(),
            seed=7,
            learner_color=chess.WHITE,
            max_plies=1,
            initial_board=board,
        )

        self.assertTrue(trajectory.truncated)
        self.assertEqual(len(trajectory.steps), 1)
        step = trajectory.steps[0]
        expected = ChessEncoder.encode_board(board).transpose(2, 0, 1)
        np.testing.assert_array_equal(step.state, expected)
        self.assertEqual(step.state.dtype, np.float32)
        self.assertEqual(step.legal_mask.dtype, np.bool_)
        self.assertTrue(step.legal_mask[step.action])
        self.assertEqual(step.current_player, 1)
        self.assertEqual(step.ply_index, 0)
        self.assertIsNone(step.target)
        self.assertEqual(
            ChessEncoder.decode_move(step.action, board, claim_draw=True).uci(),
            "e2e4",
        )

    def test_win_loss_target_sign_for_both_learner_colors(self):
        cases = [
            # Scholar's mate: White wins.
            (
                ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
                chess.WHITE,
                1.0,
            ),
            (
                ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
                chess.BLACK,
                -1.0,
            ),
            # Fool's mate: Black wins.
            (["f2f3", "e7e5", "g2g4", "d8h4"], chess.BLACK, 1.0),
            (["f2f3", "e7e5", "g2g4", "d8h4"], chess.WHITE, -1.0),
        ]
        for moves, learner_color, expected_target in cases:
            with self.subTest(moves=moves, learner_color=learner_color):
                white, black = scripted_indices(moves)
                learner_actions = white if learner_color == chess.WHITE else black
                opponent_actions = black if learner_color == chess.WHITE else white
                trajectory = collect_trajectory(
                    TinyPolicy(learner_actions),
                    TinyPolicy(opponent_actions),
                    seed=11,
                    learner_color=learner_color,
                    max_plies=len(moves),
                )
                self.assertFalse(trajectory.truncated)
                self.assertEqual(trajectory.termination_reason, "checkmate")
                self.assertTrue(trajectory.steps)
                self.assertTrue(
                    all(step.target == expected_target for step in trajectory.steps)
                )

    def test_draw_target_is_zero(self):
        board = ChessBoard("7k/8/8/8/8/8/2n5/KB6 w - - 0 1")
        move = chess.Move.from_uci("b1c2")
        action = ChessEncoder.encode_move(move, board.turn)
        trajectory = collect_trajectory(
            TinyPolicy([action]),
            TinyPolicy(),
            learner_color=chess.WHITE,
            max_plies=1,
            initial_board=board,
        )
        self.assertFalse(trajectory.truncated)
        self.assertEqual(trajectory.winner_white, 0)
        self.assertEqual(trajectory.termination_reason, "insufficient_material")
        self.assertEqual([step.target for step in trajectory.steps], [0.0])

    def test_terminal_on_limit_is_not_truncated(self):
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        white, black = scripted_indices(moves)
        terminal = collect_trajectory(
            TinyPolicy(black),
            TinyPolicy(white),
            learner_color=chess.BLACK,
            max_plies=4,
        )
        self.assertFalse(terminal.truncated)
        self.assertEqual(terminal.termination_reason, "checkmate")

        truncated = collect_trajectory(
            TinyPolicy(black[:1]),
            TinyPolicy(white),
            learner_color=chess.BLACK,
            max_plies=3,
        )
        self.assertTrue(truncated.truncated)
        self.assertEqual(truncated.termination_reason, "max_plies")
        self.assertIsNone(truncated.winner_white)
        self.assertTrue(all(step.target is None for step in truncated.steps))

    def test_batch_seed_reproducible_and_rng_continues_between_games(self):
        first = collect_trajectory_batch(
            TinyPolicy(), TinyPolicy(), 3, seed=42, max_plies=1
        )
        second = collect_trajectory_batch(
            TinyPolicy(), TinyPolicy(), 3, seed=42, max_plies=1
        )
        first_actions = [game.steps[0].action for game in (first[0], first[2])]
        second_actions = [game.steps[0].action for game in (second[0], second[2])]
        self.assertEqual(first_actions, second_actions)
        self.assertEqual([game.learner_color for game in first], [1, -1, 1])

        mask = torch.from_numpy(
            ChessEncoder.get_legal_action_mask(ChessBoard()).astype(np.bool_)
        )
        probabilities = torch.softmax(
            torch.zeros(NUM_ACTIONS).masked_fill(~mask, -torch.inf), dim=0
        )
        rng = torch.Generator(device="cpu")
        rng.manual_seed(42)
        expected = []
        for sample_index in range(3):
            action = int(torch.multinomial(probabilities, 1, generator=rng).item())
            if sample_index in (0, 2):
                expected.append(action)
        self.assertEqual(first_actions, expected)

    def test_board_models_global_rng_and_modes_are_preserved(self):
        board = ChessBoard(claim_draw=False)
        board.step("e4")
        original_fen = board.fen()
        original_stack = [move.uci() for move in board.board.move_stack]

        learner = TinyPolicy()
        opponent = TinyPolicy()
        learner.train()
        learner.bn.eval()
        opponent.eval()
        original_modes = {
            id(module): module.training
            for model in (learner, opponent)
            for module in model.modules()
        }
        original_states = [
            {key: value.detach().clone() for key, value in model.state_dict().items()}
            for model in (learner, opponent)
        ]
        global_rng = torch.random.get_rng_state().clone()

        trajectory = collect_trajectory(
            learner,
            opponent,
            seed=123,
            learner_color=chess.BLACK,
            max_plies=2,
            claim_draw=True,
            initial_board=board,
        )

        self.assertEqual(board.fen(), original_fen)
        self.assertEqual([move.uci() for move in board.board.move_stack], original_stack)
        self.assertFalse(board.claim_draw)
        self.assertEqual(trajectory.steps[0].ply_index, 1)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_rng))
        for model, state in zip((learner, opponent), original_states):
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]), key)
            for module in model.modules():
                self.assertEqual(module.training, original_modes[id(module)])
