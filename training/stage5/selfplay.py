"""Stage 5 trajectory sampling and opponent-pool collection."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import chess
import numpy as np
import torch
from torch import nn
from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS
from models.cnn_policy import CNNPolicyNetwork
from training.stage5._utils import _models_in_eval
from training.stage5.types import (
    GameStart,
    GameTrajectory,
    OpponentSnapshot,
    TrajectoryStep,
)


MIXED_START_MIX = {"standard": 16, "pgn": 12, "mate_in_one": 4}


def load_mixed_start_manifest(
    path: str | Path,
    *,
    claim_draw: bool,
) -> tuple[Dict[str, List[GameStart]], Dict[str, object]]:
    """Load and validate the small mixed-start manifest without using labels."""
    manifest_path = Path(path)
    raw_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        manifest = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid mixed-start manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Mixed-start manifest must have schema_version=1")
    positions = manifest.get("positions")
    if not isinstance(positions, list):
        raise ValueError("Mixed-start manifest positions must be a list")

    grouped: Dict[str, List[GameStart]] = {"pgn": [], "mate_in_one": []}
    source_ids = set()
    for index, entry in enumerate(positions):
        if not isinstance(entry, dict):
            raise ValueError(f"Mixed-start position {index} must be an object")
        source_type = entry.get("type")
        if source_type not in grouped:
            raise ValueError(
                f"Mixed-start position {index} has unsupported type {source_type!r}"
            )
        root_fen = entry.get("root_fen")
        prefix_uci = entry.get("prefix_uci")
        current_fen = entry.get("current_fen")
        if not isinstance(root_fen, str) or not isinstance(prefix_uci, list) or not all(
            isinstance(move, str) for move in prefix_uci
        ):
            raise ValueError(
                f"Mixed-start position {index} lacks root_fen/prefix_uci"
            )
        try:
            board = ChessBoard(root_fen, claim_draw=claim_draw)
            for uci in prefix_uci:
                move = chess.Move.from_uci(uci)
                if not board.board.is_legal(move):
                    raise ValueError(f"illegal prefix move {uci}")
                board.board.push(move)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot reconstruct mixed-start position {index}: {exc}"
            ) from exc
        reconstructed_fen = board.board.fen(en_passant="fen")
        if isinstance(current_fen, str) and reconstructed_fen != current_fen:
            raise ValueError(
                f"Mixed-start position {index} current_fen does not match history"
            )
        if not board.board.is_valid():
            raise ValueError(f"Mixed-start position {index} is invalid")
        if board.board.outcome(claim_draw=claim_draw) is not None:
            raise ValueError(
                f"Mixed-start position {index} is terminal under claim_draw={claim_draw}"
            )
        source_id = entry.get("law_state_sha256")
        if not isinstance(source_id, str) or not source_id:
            source_id = hashlib.sha256(
                json.dumps(
                    {"root_fen": root_fen, "prefix_uci": prefix_uci},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        if source_id in source_ids:
            raise ValueError("Mixed-start manifest contains duplicate source IDs")
        source_ids.add(source_id)
        grouped[source_type].append(
            GameStart(
                board=board,
                learner_color=board.turn,
                source_type=source_type,
                source_id=source_id,
            )
        )

    if len(grouped["pgn"]) < MIXED_START_MIX["pgn"]:
        raise ValueError("Mixed-start manifest requires at least 12 PGN positions")
    if len(grouped["mate_in_one"]) != MIXED_START_MIX["mate_in_one"]:
        raise ValueError("Mixed-start manifest requires exactly four mate fixtures")
    metadata: Dict[str, object] = {
        "mode": "mixed",
        "manifest_sha256": manifest_sha256,
        "schema_version": manifest["schema_version"],
        "purpose": manifest.get("purpose"),
        "position_counts": {
            "pgn": len(grouped["pgn"]),
            "mate_in_one": len(grouped["mate_in_one"]),
        },
        "mix": dict(MIXED_START_MIX),
    }
    return grouped, metadata


def build_mixed_start_batch(
    source: Dict[str, List[GameStart]],
    *,
    rng: np.random.Generator,
    claim_draw: bool,
) -> List[GameStart]:
    """Build one 16/12/4 schedule with independently balanced learner colors."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("mixed-start rng must be a numpy.random.Generator")
    pgn_source = source.get("pgn", [])
    mate_source = source.get("mate_in_one", [])
    if len(pgn_source) < MIXED_START_MIX["pgn"] or len(mate_source) != 4:
        raise ValueError("Mixed-start source does not satisfy the 16/12/4 mix")

    standard_colors = np.array(
        [chess.WHITE] * 8 + [chess.BLACK] * 8, dtype=np.bool_
    )
    pgn_colors = np.array(
        [chess.WHITE] * 6 + [chess.BLACK] * 6, dtype=np.bool_
    )
    rng.shuffle(standard_colors)
    rng.shuffle(pgn_colors)
    selected_indices = rng.choice(len(pgn_source), size=12, replace=False)

    starts: List[GameStart] = [
        GameStart(
            board=ChessBoard(claim_draw=claim_draw),
            learner_color=bool(color),
            source_type="standard",
            source_id=f"standard:{index}",
        )
        for index, color in enumerate(standard_colors)
    ]
    starts.extend(
        GameStart(
            board=pgn_source[int(source_index)].board,
            learner_color=bool(color),
            source_type="pgn",
            source_id=pgn_source[int(source_index)].source_id,
        )
        for source_index, color in zip(selected_indices, pgn_colors)
    )
    starts.extend(
        GameStart(
            board=entry.board,
            learner_color=entry.board.turn,
            source_type="mate_in_one",
            source_id=entry.source_id,
        )
        for entry in mate_source
    )
    order = rng.permutation(len(starts))
    return [starts[int(index)] for index in order]

def _validate_learner_color(learner_color: chess.Color) -> chess.Color:
    if not isinstance(learner_color, bool):
        raise ValueError("learner_color must be chess.WHITE or chess.BLACK")
    return learner_color

def _sample_legal_action(
    policy: nn.Module,
    board: ChessBoard,
    device: torch.device,
    rng: torch.Generator,
) -> tuple[np.ndarray, int, np.ndarray]:
    state = ChessEncoder.encode_board(board).transpose(2, 0, 1).copy()
    state = state.astype(np.float32, copy=False)
    legal_mask = ChessEncoder.get_legal_action_mask(
        board, claim_draw=board.claim_draw
    ).astype(np.bool_, copy=False)
    if not legal_mask.any():
        raise ValueError("Cannot sample a move from a position with no legal actions")

    tensor = torch.from_numpy(state).unsqueeze(0).to(device=device)
    logits = policy(tensor)
    if logits.shape != (1, NUM_ACTIONS):
        raise ValueError(
            f"Policy must return logits with shape (1, {NUM_ACTIONS}), "
            f"got {tuple(logits.shape)}"
        )

    # Sampling on CPU permits one private generator to continue across all games
    # without touching PyTorch's process-global RNG state.
    logits_cpu = logits[0].detach().to(device="cpu", dtype=torch.float32)
    mask_tensor = torch.from_numpy(legal_mask)
    masked_logits = logits_cpu.masked_fill(~mask_tensor, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=0)
    if not torch.isfinite(probabilities).all() or probabilities.sum().item() <= 0:
        raise ValueError("Policy produced an invalid distribution over legal actions")
    action = int(torch.multinomial(probabilities, 1, generator=rng).item())
    return state, action, legal_mask.copy()

def _collect_trajectory(
    learner_policy: nn.Module,
    opponent_policy: nn.Module,
    *,
    device: torch.device,
    rng: torch.Generator,
    learner_color: chess.Color,
    max_plies: int,
    claim_draw: bool,
    initial_board: Optional[ChessBoard],
) -> GameTrajectory:
    if max_plies < 0:
        raise ValueError("max_plies must be non-negative")
    learner_color = _validate_learner_color(learner_color)
    if initial_board is not None and not isinstance(initial_board, ChessBoard):
        raise TypeError("initial_board must be a ChessBoard")

    board = initial_board.clone() if initial_board is not None else ChessBoard()
    board.claim_draw = bool(claim_draw)
    starting_ply = len(board.board.move_stack)
    steps: List[TrajectoryStep] = []

    with _models_in_eval(learner_policy, opponent_policy), torch.no_grad():
        for _ in range(max_plies):
            outcome = board.board.outcome(claim_draw=board.claim_draw)
            if outcome is not None:
                break

            is_learner_turn = board.turn == learner_color
            policy = learner_policy if is_learner_turn else opponent_policy
            state, action, legal_mask = _sample_legal_action(
                policy, board, device, rng
            )
            move = ChessEncoder.decode_move(
                action, board, claim_draw=board.claim_draw
            )
            if move is None:
                raise RuntimeError("Sampled legal action could not be decoded")

            if is_learner_turn:
                steps.append(
                    TrajectoryStep(
                        state=state,
                        action=action,
                        legal_mask=legal_mask,
                        current_player=board.current_player,
                        ply_index=len(board.board.move_stack),
                    )
                )
            board.step(move)

        # The final permitted move may itself terminate the game.
        outcome = board.board.outcome(claim_draw=board.claim_draw)

    plies_played = len(board.board.move_stack) - starting_ply
    if outcome is None:
        return GameTrajectory(
            steps=steps,
            learner_color=1 if learner_color == chess.WHITE else -1,
            truncated=True,
            termination_reason="max_plies",
            winner_white=None,
            plies_played=plies_played,
        )

    winner_white = 0 if outcome.winner is None else (1 if outcome.winner else -1)
    for step in steps:
        step.target = float(winner_white * step.current_player)
    return GameTrajectory(
        steps=steps,
        learner_color=1 if learner_color == chess.WHITE else -1,
        truncated=False,
        termination_reason=outcome.termination.name.lower(),
        winner_white=winner_white,
        plies_played=plies_played,
    )

def collect_trajectory(
    learner_policy: nn.Module,
    opponent_policy: nn.Module,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
    learner_color: chess.Color = chess.WHITE,
    max_plies: int = 300,
    claim_draw: bool = True,
    initial_board: Optional[ChessBoard] = None,
) -> GameTrajectory:
    """Collect one game with a private, reproducible sampling stream."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return _collect_trajectory(
        learner_policy,
        opponent_policy,
        device=torch.device(device),
        rng=rng,
        learner_color=learner_color,
        max_plies=max_plies,
        claim_draw=claim_draw,
        initial_board=initial_board,
    )

def collect_trajectory_batch(
    learner_policy: nn.Module,
    opponent_policy: nn.Module,
    num_games: int,
    *,
    device: str | torch.device = "cpu",
    seed: int = 42,
    first_learner_color: chess.Color = chess.WHITE,
    max_plies: int = 300,
    claim_draw: bool = True,
    initial_board: Optional[ChessBoard] = None,
) -> List[GameTrajectory]:
    """Collect games with alternating learner colors and one continuous RNG."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return _collect_trajectory_batch_with_rng(
        learner_policy,
        opponent_policy,
        num_games,
        device=torch.device(device),
        rng=rng,
        first_learner_color=first_learner_color,
        max_plies=max_plies,
        claim_draw=claim_draw,
        initial_board=initial_board,
    )

def _collect_trajectory_batch_with_rng(
    learner_policy: nn.Module,
    opponent_policy: nn.Module,
    num_games: int,
    *,
    device: torch.device,
    rng: torch.Generator,
    first_learner_color: chess.Color,
    max_plies: int,
    claim_draw: bool,
    initial_board: Optional[ChessBoard],
) -> List[GameTrajectory]:
    """Internal batch collector using a caller-owned continuous RNG stream."""
    if num_games < 0:
        raise ValueError("num_games must be non-negative")
    first_learner_color = _validate_learner_color(first_learner_color)
    trajectories = []
    for game_index in range(num_games):
        learner_color = (
            first_learner_color if game_index % 2 == 0 else not first_learner_color
        )
        trajectories.append(
            _collect_trajectory(
                learner_policy,
                opponent_policy,
                device=device,
                rng=rng,
                learner_color=learner_color,
                max_plies=max_plies,
                claim_draw=claim_draw,
                initial_board=initial_board,
            )
        )
    return trajectories


def _collect_trajectory_batch_from_starts_with_rng(
    learner_policy: nn.Module,
    opponent_policy: nn.Module,
    starts: Sequence[GameStart],
    *,
    device: torch.device,
    rng: torch.Generator,
    max_plies: int,
    claim_draw: bool,
) -> List[GameTrajectory]:
    """Collect a caller-scheduled batch while preserving each start history."""
    trajectories = []
    for start in starts:
        if not isinstance(start, GameStart):
            raise TypeError("starts must contain GameStart entries")
        trajectories.append(
            _collect_trajectory(
                learner_policy,
                opponent_policy,
                device=device,
                rng=rng,
                learner_color=start.learner_color,
                max_plies=max_plies,
                claim_draw=claim_draw,
                initial_board=start.board,
            )
        )
    return trajectories

def _snapshot_policy(
    policy: CNNPolicyNetwork,
    *,
    opponent_id: str,
    iteration: int,
    device: torch.device,
) -> OpponentSnapshot:
    """Copy Policy weights into an independent frozen inference model."""
    with torch.random.fork_rng(devices=[]):
        snapshot = CNNPolicyNetwork.from_config(policy.get_config())
    snapshot.load_state_dict(policy.state_dict(), strict=True)
    snapshot.to(device).eval().requires_grad_(False)
    return OpponentSnapshot(opponent_id, iteration, snapshot)

def _collect_trajectory_batch_with_opponent_pool(
    learner_policy: nn.Module,
    opponent_pool: Sequence[OpponentSnapshot],
    num_games: int,
    *,
    device: torch.device,
    move_rng: torch.Generator,
    opponent_rng: np.random.Generator,
    first_learner_color: chess.Color,
    max_plies: int,
    claim_draw: bool,
    initial_board: Optional[ChessBoard],
) -> tuple[List[GameTrajectory], Dict[str, int]]:
    """Collect games while independently sampling one frozen opponent per game."""
    if num_games < 0:
        raise ValueError("num_games must be non-negative")
    if not opponent_pool:
        raise ValueError("opponent_pool must not be empty")
    if not isinstance(opponent_rng, np.random.Generator):
        raise TypeError("opponent_rng must be a numpy.random.Generator")
    identifiers = [snapshot.opponent_id for snapshot in opponent_pool]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("opponent_pool IDs must be unique")
    first_learner_color = _validate_learner_color(first_learner_color)
    trajectories = []
    opponent_games = {identifier: 0 for identifier in identifiers}
    for game_index in range(num_games):
        pool_index = int(opponent_rng.integers(len(opponent_pool)))
        selected = opponent_pool[pool_index]
        opponent_games[selected.opponent_id] += 1
        learner_color = (
            first_learner_color if game_index % 2 == 0 else not first_learner_color
        )
        trajectories.append(
            _collect_trajectory(
                learner_policy,
                selected.model,
                device=device,
                rng=move_rng,
                learner_color=learner_color,
                max_plies=max_plies,
                claim_draw=claim_draw,
                initial_board=initial_board,
            )
        )
    return trajectories, opponent_games


def _collect_trajectory_batch_from_starts_with_opponent_pool(
    learner_policy: nn.Module,
    opponent_pool: Sequence[OpponentSnapshot],
    starts: Sequence[GameStart],
    *,
    device: torch.device,
    move_rng: torch.Generator,
    opponent_rng: np.random.Generator,
    max_plies: int,
    claim_draw: bool,
) -> tuple[List[GameTrajectory], Dict[str, int]]:
    """Collect a scheduled mixed-start batch with one pool draw per game."""
    if not opponent_pool:
        raise ValueError("opponent_pool must not be empty")
    if not isinstance(opponent_rng, np.random.Generator):
        raise TypeError("opponent_rng must be a numpy.random.Generator")
    identifiers = [snapshot.opponent_id for snapshot in opponent_pool]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("opponent_pool IDs must be unique")
    trajectories = []
    opponent_games = {identifier: 0 for identifier in identifiers}
    for start in starts:
        if not isinstance(start, GameStart):
            raise TypeError("starts must contain GameStart entries")
        selected = opponent_pool[int(opponent_rng.integers(len(opponent_pool)))]
        opponent_games[selected.opponent_id] += 1
        trajectories.append(
            _collect_trajectory(
                learner_policy,
                selected.model,
                device=device,
                rng=move_rng,
                learner_color=start.learner_color,
                max_plies=max_plies,
                claim_draw=claim_draw,
                initial_board=start.board,
            )
        )
    return trajectories, opponent_games
