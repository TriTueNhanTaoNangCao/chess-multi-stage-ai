"""Small deterministic tests for the separate AlphaGo-style Value collector."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder
from models.cnn_policy import CNNPolicyNetwork
from models.value_net import ValueNetwork
from scripts.stage5.stage5_alphago_value_data import (
    filter_cross_split_state_duplicates,
    restore_resume,
)
from tests.stage5.helpers import TinyPolicy, write_small_stage4_checkpoint
from training.stage5.alphago_value_data import (
    AlphaGoValueDataConfig,
    CollectionRNG,
    collect_value_game,
    load_value_data_policies,
    sample_uniform_legal_move,
)
from training.stage5.checkpoint import save_stage5_checkpoint
from training.stage5.types import Stage5TrainConfig


class FixedIntegers:
    def __init__(self, *values: int):
        self.values = list(values)

    def integers(self, *args, **kwargs):
        if not self.values:
            raise AssertionError("No fixed integer remains")
        return self.values.pop(0)


def policy_rng(seed: int = 1) -> torch.Generator:
    result = torch.Generator(device="cpu")
    result.manual_seed(seed)
    return result


def move_index(board: ChessBoard, uci: str) -> int:
    moves = sorted(board.get_legal_actions(), key=lambda move: move.uci())
    return [move.uci() for move in moves].index(uci)


def scripted_actions(moves: list[str], board: ChessBoard) -> list[int]:
    replay = board.clone()
    result = []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        result.append(ChessEncoder.encode_move(move, replay.turn))
        replay.step(move)
    return result


def write_disabled_stage5(path: Path) -> None:
    policy = CNNPolicyNetwork(hidden_dim=2, fc_dim=8, dropout=0.0)
    opponent = CNNPolicyNetwork.from_config(policy.get_config())
    opponent.load_state_dict(policy.state_dict())
    value = ValueNetwork(hidden_dim=4, conv2_dim=4, fc1_dim=8, fc2_dim=4)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    save_stage5_checkpoint(
        path,
        policy_model=policy,
        value_model=value,
        opponent_model=opponent,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        config=Stage5TrainConfig(
            stage4_checkpoint="source.pt",
            output_dir=path.parent,
            iterations=1,
            baseline="zero",
            value_mode="disabled",
        ),
        completed_iteration=0,
        history=[],
        source_checkpoint={"path": "source.pt", "stage": 4},
        collector_rng=policy_rng(7),
        value_rng=np.random.default_rng(7),
        claim_draw=True,
    )


def test_checkpoint_metadata_is_validated_and_policies_are_frozen():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sl_path, rl_path = root / "sl.pt", root / "rl.pt"
        write_small_stage4_checkpoint(sl_path)
        write_disabled_stage5(rl_path)
        loaded = load_value_data_policies(sl_path, rl_path)
        assert loaded.claim_draw is True
        assert loaded.provenance["rl"]["baseline"] == "zero"
        assert loaded.provenance["rl"]["value_mode"] == "disabled"
        assert len(loaded.provenance["sl"]["sha256"]) == 64
        assert not loaded.sl_policy.training and not loaded.rl_policy.training
        assert not any(p.requires_grad for p in loaded.sl_policy.parameters())
        payload = torch.load(rl_path, map_location="cpu", weights_only=False)
        payload["training_config"]["baseline"] = "value"
        torch.save(payload, rl_path)
        try:
            load_value_data_policies(sl_path, rl_path)
        except ValueError as exc:
            assert "baseline='zero'" in str(exc)
        else:
            raise AssertionError("invalid RL baseline was accepted")


def test_three_phases_save_post_uniform_state_and_target():
    board = ChessBoard(claim_draw=True)
    sl = TinyPolicy(scripted_actions(["f2f3"], board))
    after_sl = board.clone()
    after_sl.step("f2f3")
    uniform_index = move_index(after_sl, "e7e5")
    after_uniform = after_sl.clone()
    after_uniform.step("e7e5")
    rl = TinyPolicy(scripted_actions(["g2g4", "d8h4"], after_uniform))
    rng = CollectionRNG(FixedIntegers(2), FixedIntegers(uniform_index), policy_rng())
    game = collect_value_game(
        sl,
        rl,
        config=AlphaGoValueDataConfig(2, 2, 8, True),
        rng=rng,
        game_id="three-phase",
        initial_board=board,
    )
    assert game.moves_uci == ("f2f3", "e7e5", "g2g4", "d8h4")
    assert (game.sl_plies, game.rl_plies, sl.calls, rl.calls) == (1, 2, 1, 2)
    assert game.random_move_uci == "e7e5"
    assert game.saved_current_player == 1
    assert game.winner_white == -1 and game.target == -1.0
    expected = ChessEncoder.encode_board(after_uniform).transpose(2, 0, 1)
    np.testing.assert_array_equal(game.state, expected)


def test_target_sign_for_opposite_saved_side_and_full_history():
    board = ChessBoard(claim_draw=True)
    prefix = ("e2e4", "e7e5", "f1c4", "b8c6", "d1h5")
    for move in prefix:
        board.step(move)
    index = move_index(board, "g8f6")
    after_uniform = board.clone()
    after_uniform.step("g8f6")
    rl = TinyPolicy(scripted_actions(["h5f7"], after_uniform))
    game = collect_value_game(
        TinyPolicy(),
        rl,
        config=AlphaGoValueDataConfig(1, 1, 2, True),
        rng=CollectionRNG(FixedIntegers(1), FixedIntegers(index), policy_rng()),
        game_id="white-mates",
        initial_board=board,
    )
    assert game.status == "sampled" and game.target == 1.0
    assert game.saved_current_player == 1 and game.winner_white == 1
    assert game.moves_uci == prefix + ("g8f6", "h5f7")


def test_terminal_before_sample_and_both_truncation_sides_are_excluded():
    mate_board = ChessBoard(claim_draw=True)
    for move in ("f2f3", "e7e5", "g2g4"):
        mate_board.step(move)
    sl = TinyPolicy(scripted_actions(["d8h4"], mate_board))
    terminal = collect_value_game(
        sl, TinyPolicy(), config=AlphaGoValueDataConfig(2, 2, 2, True),
        rng=CollectionRNG(FixedIntegers(2), FixedIntegers(), policy_rng()),
        game_id="early-terminal", initial_board=mate_board,
    )
    assert terminal.status == "excluded_terminal_before_sample"
    assert terminal.state is None and terminal.target is None

    before = collect_value_game(
        TinyPolicy(), TinyPolicy(), config=AlphaGoValueDataConfig(2, 2, 1, True),
        rng=CollectionRNG(FixedIntegers(2), FixedIntegers(), policy_rng()),
        game_id="truncated-before",
    )
    assert before.status == "excluded_truncated_before_sample" and before.truncated
    assert before.winner_white is None and before.target is None

    start = ChessBoard(claim_draw=True)
    after = collect_value_game(
        TinyPolicy(), TinyPolicy(), config=AlphaGoValueDataConfig(1, 1, 1, True),
        rng=CollectionRNG(
            FixedIntegers(1), FixedIntegers(move_index(start, "e2e4")), policy_rng()
        ), game_id="truncated-after",
    )
    assert after.status == "excluded_truncated_after_sample" and after.truncated
    assert after.winner_white is None and after.state is None and after.target is None


def test_uniform_move_sampling_and_rng_resume_are_reproducible():
    board = ChessBoard(claim_draw=True)
    legal = sorted(move.uci() for move in board.get_legal_actions())
    rng = np.random.default_rng(1234)
    counts = {move: 0 for move in legal}
    for _ in range(20_000):
        counts[sample_uniform_legal_move(board, rng).uci()] += 1
    assert max(abs(count - 1000) for count in counts.values()) < 120

    config = AlphaGoValueDataConfig(1, 2, 1, True)
    first = CollectionRNG.from_seed(77)
    collect_value_game(TinyPolicy(), TinyPolicy(), config=config, rng=first, game_id="g0")
    saved = first.state_json()
    expected = collect_value_game(
        TinyPolicy(), TinyPolicy(), config=config, rng=first, game_id="g1"
    )
    resumed = CollectionRNG.from_seed(999)
    resumed.restore_json(saved)
    actual = collect_value_game(
        TinyPolicy(), TinyPolicy(), config=config, rng=resumed, game_id="g1"
    )
    assert (actual.u, actual.moves_uci, actual.status) == (
        expected.u, expected.moves_uci, expected.status
    )

    with tempfile.TemporaryDirectory() as directory:
        shard = Path(directory) / "shard_00000.npz"
        np.savez_compressed(
            shard,
            shard_index=np.asarray(0),
            config_sha256=np.asarray("c" * 64),
            sl_checkpoint_sha256=np.asarray("s" * 64),
            rl_checkpoint_sha256=np.asarray("r" * 64),
            games_completed=np.asarray(1),
            collection_rng_state_json=np.asarray(saved),
        )
        restored_from_shard = CollectionRNG.from_seed(0)
        completed = restore_resume(
            [shard], config_sha256="c" * 64, sl_sha256="s" * 64,
            rl_sha256="r" * 64, rng=restored_from_shard,
        )
        replayed = collect_value_game(
            TinyPolicy(), TinyPolicy(), config=config, rng=restored_from_shard,
            game_id="g1",
        )
        assert completed == 1
        assert (replayed.u, replayed.moves_uci, replayed.status) == (
            expected.u, expected.moves_uci, expected.status
        )


def test_cross_split_exact_tensor_duplicates_are_removed():
    row = lambda fingerprint: {"state_fingerprint": fingerprint}
    filtered = filter_cross_split_state_duplicates(
        {
            "train": [row("train"), row("shared-val"), row("shared-test")],
            "val": [row("shared-val"), row("shared-test")],
            "test": [row("shared-test")],
        }
    )
    assert [r["state_fingerprint"] for r in filtered["train"]] == ["train"]
    assert [r["state_fingerprint"] for r in filtered["val"]] == ["shared-val"]
    assert [r["state_fingerprint"] for r in filtered["test"]] == ["shared-test"]


def load_tests(loader, tests, pattern):
    """Expose the compact function tests to the repository's unittest runner."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
