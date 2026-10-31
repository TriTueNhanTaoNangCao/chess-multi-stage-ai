"""AlphaGo-2016-style three-phase Value data collection for chess.

This module only generates terminal state/outcome examples.  It does not train
ValueNetwork and is deliberately separate from the Stage 5 REINFORCE loop and
the generic Value-only training helpers.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import chess
import numpy as np
import torch
from torch import nn

from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder
from training.stage4_train_sl import load_checkpoint
from training.stage5.checkpoint import load_stage5_checkpoint
from training.stage5.selfplay import _sample_legal_action


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AlphaGoValueDataConfig:
    """Collection contract; U is a one-based generated-ply number.

    If U=1, no SL ply is played: the uniform legal move is the first generated
    ply.  If U=k, SL samples exactly k-1 plies, then the kth ply is uniform.
    ``max_plies`` counts generated plies from the supplied initial position.
    """

    u_min: int
    u_max: int
    max_plies: int
    claim_draw: bool

    def validate(self) -> None:
        for name in ("u_min", "u_max", "max_plies"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.u_min > self.u_max:
            raise ValueError("u_min must not exceed u_max")
        if not isinstance(self.claim_draw, bool):
            raise ValueError("claim_draw must be boolean")


@dataclass
class CollectionRNG:
    """Three private RNG streams: U, uniform move, and policy sampling."""

    u_rng: np.random.Generator
    uniform_rng: np.random.Generator
    policy_rng: torch.Generator

    @classmethod
    def from_seed(cls, seed: int) -> "CollectionRNG":
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        u_seed, uniform_seed, policy_seed = np.random.SeedSequence(seed).spawn(3)
        policy_value = int(policy_seed.generate_state(1, dtype=np.uint64)[0])
        policy_rng = torch.Generator(device="cpu")
        policy_rng.manual_seed(policy_value % (2**63 - 1))
        return cls(
            u_rng=np.random.default_rng(u_seed),
            uniform_rng=np.random.default_rng(uniform_seed),
            policy_rng=policy_rng,
        )

    def state(self) -> Dict[str, object]:
        return {
            "u_rng": copy.deepcopy(self.u_rng.bit_generator.state),
            "uniform_rng": copy.deepcopy(self.uniform_rng.bit_generator.state),
            "policy_rng": self.policy_rng.get_state().cpu().numpy().copy(),
        }

    def restore(self, state: Dict[str, object]) -> None:
        try:
            self.u_rng.bit_generator.state = copy.deepcopy(state["u_rng"])
            self.uniform_rng.bit_generator.state = copy.deepcopy(
                state["uniform_rng"]
            )
            self.policy_rng.set_state(
                torch.from_numpy(np.asarray(state["policy_rng"], dtype=np.uint8).copy())
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid collection RNG state: {exc}") from exc

    def state_json(self) -> str:
        state = self.state()
        state["policy_rng"] = np.asarray(state["policy_rng"]).tolist()
        return json.dumps(state, sort_keys=True, separators=(",", ":"))

    def restore_json(self, payload: str) -> None:
        state = json.loads(payload)
        state["policy_rng"] = np.asarray(state["policy_rng"], dtype=np.uint8)
        self.restore(state)


@dataclass
class ValueDataPolicies:
    sl_policy: nn.Module
    rl_policy: nn.Module
    claim_draw: bool
    provenance: Dict[str, object]


@dataclass
class AlphaGoValueGame:
    game_id: str
    u: int
    root_fen: str
    moves_uci: tuple[str, ...]
    status: str
    termination_reason: str
    truncated: bool
    winner_white: Optional[int]
    saved_current_player: Optional[int]
    random_move_uci: Optional[str]
    state: Optional[np.ndarray]
    target: Optional[float]
    sl_plies: int
    rl_plies: int


def load_value_data_policies(
    sl_checkpoint: str | Path,
    rl_checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ValueDataPolicies:
    """Strictly load the explicitly supplied SL and zero-baseline RL policies."""
    sl_path, rl_path = Path(sl_checkpoint), Path(rl_checkpoint)
    if not sl_path.is_file():
        raise FileNotFoundError(f"SL Policy checkpoint not found: {sl_path}")
    if not rl_path.is_file():
        raise FileNotFoundError(f"RL Policy checkpoint not found: {rl_path}")
    sl_policy, sl_metadata = load_checkpoint(sl_path, device=device)
    rl_state = load_stage5_checkpoint(rl_path, device=device)
    rl_metadata = rl_state.payload
    training_config = rl_metadata.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("RL checkpoint lacks training_config metadata")
    if training_config.get("baseline") != "zero":
        raise ValueError("RL checkpoint must have baseline='zero'")
    value_mode = rl_metadata.get("value_mode", training_config.get("value_mode"))
    if value_mode != "disabled":
        raise ValueError("RL checkpoint must have value_mode='disabled'")
    sl_claim_draw = sl_metadata.get("claim_draw")
    rl_claim_draw = rl_metadata.get("claim_draw")
    if not isinstance(sl_claim_draw, bool) or sl_claim_draw != rl_claim_draw:
        raise ValueError("SL and RL checkpoint claim_draw metadata must match")
    sl_policy.eval().requires_grad_(False)
    rl_state.policy_model.eval().requires_grad_(False)
    return ValueDataPolicies(
        sl_policy=sl_policy,
        rl_policy=rl_state.policy_model,
        claim_draw=sl_claim_draw,
        provenance={
            "sl": {
                "path": str(sl_path),
                "sha256": sha256_file(sl_path),
                "stage": sl_metadata.get("stage"),
                "model_class": sl_metadata.get("model_class"),
                "encoder_version": sl_metadata.get("encoder_version"),
                "action_version": sl_metadata.get("action_version"),
            },
            "rl": {
                "path": str(rl_path),
                "sha256": sha256_file(rl_path),
                "stage": rl_metadata.get("stage"),
                "model_class": rl_metadata.get("model_class"),
                "encoder_version": rl_metadata.get("encoder_version"),
                "action_version": rl_metadata.get("action_version"),
                "baseline": training_config.get("baseline"),
                "value_mode": value_mode,
            },
            "claim_draw": sl_claim_draw,
        },
    )


def sample_uniform_legal_move(
    board: ChessBoard, rng: np.random.Generator
) -> chess.Move:
    """Sample uniformly from the complete legal move set in stable UCI order."""
    moves = sorted(board.get_legal_actions(), key=lambda move: move.uci())
    if not moves:
        raise ValueError("Cannot sample a uniform move from a terminal position")
    return moves[int(rng.integers(len(moves)))]


def _policy_move(
    policy: nn.Module,
    board: ChessBoard,
    device: torch.device,
    rng: torch.Generator,
) -> chess.Move:
    _, action, _ = _sample_legal_action(policy, board, device, rng)
    move = ChessEncoder.decode_move(action, board, claim_draw=board.claim_draw)
    if move is None or not board.board.is_legal(move):
        raise RuntimeError("Policy sampled an action that did not decode as legal")
    return move


def collect_value_game(
    sl_policy: nn.Module,
    rl_policy: nn.Module,
    *,
    config: AlphaGoValueDataConfig,
    rng: CollectionRNG,
    game_id: str,
    device: str | torch.device = "cpu",
    initial_board: Optional[ChessBoard] = None,
) -> AlphaGoValueGame:
    """Generate at most one terminal Value example with SL/uniform/RL phases."""
    config.validate()
    if not isinstance(game_id, str) or not game_id:
        raise ValueError("game_id must be a non-empty string")
    board = initial_board.clone() if initial_board is not None else ChessBoard()
    board.claim_draw = config.claim_draw
    root_fen = board.board.root().fen(en_passant="fen")
    start_stack = len(board.board.move_stack)
    u = int(rng.u_rng.integers(config.u_min, config.u_max + 1))
    sl_plies = rl_plies = 0
    random_move: Optional[chess.Move] = None
    saved_state: Optional[np.ndarray] = None
    saved_player: Optional[int] = None

    def finish(
        status: str,
        reason: str,
        *,
        truncated: bool,
        winner: Optional[int] = None,
        state: Optional[np.ndarray] = None,
        target: Optional[float] = None,
    ) -> AlphaGoValueGame:
        return AlphaGoValueGame(
            game_id=game_id,
            u=u,
            root_fen=root_fen,
            moves_uci=tuple(move.uci() for move in board.board.move_stack),
            status=status,
            termination_reason=reason,
            truncated=truncated,
            winner_white=winner,
            saved_current_player=saved_player,
            random_move_uci=None if random_move is None else random_move.uci(),
            state=state,
            target=target,
            sl_plies=sl_plies,
            rl_plies=rl_plies,
        )

    device_obj = torch.device(device)
    with torch.no_grad():
        while sl_plies < u - 1:
            outcome = board.outcome()
            if outcome is not None:
                return finish(
                    "excluded_terminal_before_sample",
                    f"terminal_during_sl:{outcome.termination.name.lower()}",
                    truncated=False,
                )
            if len(board.board.move_stack) - start_stack >= config.max_plies:
                return finish(
                    "excluded_truncated_before_sample", "max_plies_before_uniform", truncated=True
                )
            board.step(_policy_move(sl_policy, board, device_obj, rng.policy_rng))
            sl_plies += 1

        outcome = board.outcome()
        if outcome is not None:
            return finish(
                "excluded_terminal_before_sample",
                f"terminal_before_uniform:{outcome.termination.name.lower()}",
                truncated=False,
            )
        if len(board.board.move_stack) - start_stack >= config.max_plies:
            return finish(
                "excluded_truncated_before_sample", "max_plies_before_uniform", truncated=True
            )

        random_move = sample_uniform_legal_move(board, rng.uniform_rng)
        board.step(random_move)
        saved_state = ChessEncoder.encode_board(board).transpose(2, 0, 1).copy()
        saved_state = saved_state.astype(np.float32, copy=False)
        saved_player = board.current_player

        outcome = board.outcome()
        while outcome is None and len(board.board.move_stack) - start_stack < config.max_plies:
            board.step(_policy_move(rl_policy, board, device_obj, rng.policy_rng))
            rl_plies += 1
            outcome = board.outcome()

    if outcome is None:
        return finish(
            "excluded_truncated_after_sample", "max_plies_after_uniform", truncated=True
        )
    winner = 0 if outcome.winner is None else (1 if outcome.winner else -1)
    target = float(winner * saved_player)
    return finish(
        "sampled",
        outcome.termination.name.lower(),
        truncated=False,
        winner=winner,
        state=saved_state,
        target=target,
    )
