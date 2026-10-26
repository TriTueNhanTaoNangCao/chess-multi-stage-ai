"""
Stage 4: PGN Dataset Pipeline for Supervised Learning (Policy & Value Networks).
Converts PGN games into training samples (state, target action, legal mask, outcome)
with game-level splitting to prevent data leakage and robust error filtering.
"""

from __future__ import annotations
from dataclasses import dataclass
import collections
import hashlib
import io
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import chess
import chess.pgn
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    torch = None
    TorchDataset = object  # type: ignore

from games.chess.encoder import ChessEncoder, NUM_ACTIONS, NUM_CHANNELS


PGN_CLAIM_DRAW = False


@dataclass
class ChessSample:
    """A single training sample extracted before a move is played."""
    state: np.ndarray       # float32, HWC (8, 8, 14)
    target: int             # action index in [0..4671]
    legal_mask: np.ndarray  # bool array of shape (4672,)
    game_id: str            # game identifier
    ply_index: int          # ply number (0-indexed)
    outcome: float          # +1.0 (White won), -1.0 (Black won), 0.0 (draw/unknown)
    outcome_known: bool     # True if outcome is known (1-0, 0-1, 1/2-1/2); False if unknown (*)


def parse_outcome(result_str: str) -> Tuple[float, bool]:
    """
    Parse PGN result string into (outcome, outcome_known).
    - "1-0": (1.0, True)   - White won
    - "0-1": (-1.0, True)  - Black won
    - "1/2-1/2": (0.0, True) - Draw
    - "*", empty, or invalid: (0.0, False) - Unknown / unfinished game
    """
    res = result_str.strip()
    if res == "1-0":
        return 1.0, True
    elif res == "0-1":
        return -1.0, True
    elif res in ("1/2-1/2", "1/2"):
        return 0.0, True
    return 0.0, False


def parse_game(game: chess.pgn.Game, game_id: str) -> Tuple[Optional[List[ChessSample]], Optional[str], str]:
    """
    Parse a single PGN game into samples taken from the state BEFORE each move.

    Returns:
        (samples, error_reason, fingerprint)
        If parsing succeeds, error_reason is None.
        fingerprint is the canonical move sequence hash (variant + root_fen + mainline UCI)
        for deduplication and split grouping.
    """
    if game.errors:
        err = game.errors[0]
        if isinstance(err, chess.IllegalMoveError):
            return None, f"illegal_move: {err}", ""
        return None, f"parse_error: {err}", ""

    # 1. Variant check: only Standard Chess is supported
    variant_raw = game.headers.get("Variant", "Standard").strip()
    if variant_raw.lower() not in ("standard", "chess"):
        return None, f"unsupported_variant: {variant_raw}", ""
    variant_norm = "standard"

    try:
        board = game.board()
    except Exception as e:
        return None, f"invalid_root_board: {e}", ""

    if not board.is_valid():
        return None, "invalid_board_state", ""

    root_fen = board.root().fen(en_passant="fen")
    result_header = game.headers.get("Result", "*")
    outcome, outcome_known = parse_outcome(result_header)

    samples: List[ChessSample] = []
    move_ucis: List[str] = []
    ply_index = 0

    try:
        mainline = list(game.mainline_moves())
    except Exception as e:
        return None, f"corrupted_moves: {e}", ""

    if not mainline:
        return None, "empty_game", ""

    for move in mainline:
        # 1. Validate move legality
        if move not in board.legal_moves:
            return None, f"illegal_move: {move.uci()}", f"{variant_norm}|{root_fen}|{' '.join(move_ucis)}"

        # 2. Encode state BEFORE the move (HWC float32)
        state = ChessEncoder.encode_board(board)

        # 3. Encode target action using current board.turn
        try:
            target = ChessEncoder.encode_move(move, board.turn)
        except Exception as e:
            return None, f"encoding_error: {move.uci()} ({e})", f"{variant_norm}|{root_fen}|{' '.join(move_ucis)}"

        # 4. Validate roundtrip and action boundaries
        if not (0 <= target < NUM_ACTIONS):
            return None, f"target_out_of_bounds: {target}", f"{variant_norm}|{root_fen}|{' '.join(move_ucis)}"

        decoded = ChessEncoder.decode_move(target, board, claim_draw=PGN_CLAIM_DRAW)
        if decoded != move:
            return None, f"decode_mismatch: {move.uci()} != {decoded.uci() if decoded else 'None'}", f"{variant_norm}|{root_fen}|{' '.join(move_ucis)}"

        # 5. Extract boolean legal action mask
        legal_mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        for act_idx in ChessEncoder.legal_action_indices(board, claim_draw=PGN_CLAIM_DRAW):
            legal_mask[act_idx] = True

        if not legal_mask[target]:
            return None, f"target_not_in_legal_mask: {target}", f"{variant_norm}|{root_fen}|{' '.join(move_ucis)}"

        samples.append(ChessSample(
            state=state,
            target=target,
            legal_mask=legal_mask,
            game_id=game_id,
            ply_index=ply_index,
            outcome=outcome,
            outcome_known=outcome_known,
        ))

        move_ucis.append(move.uci())
        board.push(move)
        ply_index += 1

    canonical_moves = " ".join(move_ucis)
    fingerprint_raw = f"{variant_norm}|{root_fen}|{canonical_moves}"
    fingerprint = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()
    return samples, None, fingerprint


def build_dataset(
    pgn_source: Union[str, Path, io.TextIOBase],
    output_dir: Optional[Union[str, Path]] = None,
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
    max_games: Optional[int] = None,
    drop_duplicates: bool = False,
    max_samples: Optional[int] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read PGN, filter invalid games and unsupported variants, group duplicates,
    split deterministically by game, and optionally save to compressed npz files with manifest.json.
    """
    if split_ratios is None:
        split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    # Normalize split ratios
    ratio_sum = sum(split_ratios.values())
    if abs(ratio_sum - 1.0) > 1e-5:
        raise ValueError(f"split_ratios must sum to 1.0, got {ratio_sum}")

    # Prepare stream
    opened_file = None
    if isinstance(pgn_source, (str, Path)):
        p = Path(pgn_source)
        if not p.exists():
            raise FileNotFoundError(f"PGN file not found: {p}")
        stream = open(p, "r", encoding="utf-8", errors="replace")
        opened_file = stream
    else:
        stream = pgn_source

    stats: Dict[str, Any] = {
        "total_pgn_games_read": 0,
        "valid_games": 0,
        "rejected_games": 0,
        "rejection_reasons": collections.defaultdict(int),
        "unique_games": 0,
        "duplicate_games": 0,
        "duplicate_handling": "dropped_completely" if drop_duplicates else "kept_grouped_in_same_split",
        "total_samples_kept": 0,
    }

    # Group valid games by move fingerprint to ensure identical games share the same split
    games_by_fingerprint: Dict[str, List[List[ChessSample]]] = collections.defaultdict(list)

    try:
        game_idx = 0
        while True:
            if max_games is not None and (len(games_by_fingerprint) >= max_games or stats["valid_games"] >= max_games):
                break
            if max_samples is not None and stats["total_samples_kept"] >= max_samples:
                break

            try:
                game = chess.pgn.read_game(stream)
            except Exception as e:
                stats["rejected_games"] += 1
                stats["rejection_reasons"]["pgn_syntax_error"] += 1
                continue

            if game is None:
                break

            stats["total_pgn_games_read"] += 1
            game_id = f"game_{game_idx}"
            game_idx += 1

            samples, err_reason, fingerprint = parse_game(game, game_id)
            if err_reason is not None:
                stats["rejected_games"] += 1
                err_key = err_reason.split(":")[0].strip()
                stats["rejection_reasons"][err_key] += 1
                continue

            if fingerprint in games_by_fingerprint:
                stats["duplicate_games"] += 1
                if drop_duplicates:
                    continue

            games_by_fingerprint[fingerprint].append(samples)
            stats["valid_games"] += 1
            stats["total_samples_kept"] += len(samples)

    finally:
        if opened_file is not None:
            opened_file.close()

    # Deterministic split by unique game fingerprint
    unique_fingerprints = sorted(games_by_fingerprint.keys())
    stats["unique_games"] = len(unique_fingerprints)

    rng = random.Random(seed)
    rng.shuffle(unique_fingerprints)

    n_groups = len(unique_fingerprints)
    train_end = int(round(n_groups * split_ratios.get("train", 0.8)))
    val_end = train_end + int(round(n_groups * split_ratios.get("val", 0.1)))

    split_groups = {
        "train": unique_fingerprints[:train_end],
        "val": unique_fingerprints[train_end:val_end],
        "test": unique_fingerprints[val_end:],
    }

    # Assemble samples per split
    split_samples: Dict[str, List[ChessSample]] = {"train": [], "val": [], "test": []}
    split_game_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split_name, fps in split_groups.items():
        for fp in fps:
            for game_samples in games_by_fingerprint[fp]:
                split_samples[split_name].extend(game_samples)
                split_game_counts[split_name] += 1

    sample_counts = {k: len(v) for k, v in split_samples.items()}

    leakage_note = (
        "Game-level splitting places all games with identical (variant, root_fen, mainline) "
        "into the exact same split to prevent game-sequence leakage. "
        "However, transposition of moves across different games and shared standard opening theory "
        "may still cause identical individual board positions to appear across splits."
    )

    result_summary: Dict[str, Any] = {
        "format_version": 1,
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": NUM_ACTIONS,
        "seed": seed,
        "split_ratios": split_ratios,
        "duplicate_handling": stats["duplicate_handling"],
        "leakage_note": leakage_note,
        "pgn_draw_claim_policy": {
            "claim_draw": PGN_CLAIM_DRAW,
            "description": "Continue through claimable draws while validating the recorded PGN mainline.",
        },
        "stats": {
            "total_pgn_games_read": stats["total_pgn_games_read"],
            "valid_games": stats["valid_games"],
            "rejected_games": stats["rejected_games"],
            "unique_games": stats["unique_games"],
            "duplicate_games": stats["duplicate_games"],
            "duplicate_handling": stats["duplicate_handling"],
            "total_samples_kept": stats["total_samples_kept"],
            "rejection_reasons": dict(stats["rejection_reasons"]),
        },
        "num_games": split_game_counts,
        "num_samples": sample_counts,
    }
    if source_metadata is not None:
        result_summary["source_metadata"] = source_metadata

    # Save to disk if output_dir is specified
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for split_name, samples_list in split_samples.items():
            if not samples_list:
                states = np.empty((0, 8, 8, NUM_CHANNELS), dtype=np.float32)
                targets = np.empty((0,), dtype=np.int32)
                legal_masks = np.empty((0, NUM_ACTIONS), dtype=np.bool_)
                game_ids = np.empty((0,), dtype="<U64")
                ply_indices = np.empty((0,), dtype=np.int32)
                outcomes = np.empty((0,), dtype=np.float32)
                outcome_known = np.empty((0,), dtype=np.bool_)
            else:
                states = np.stack([s.state for s in samples_list], axis=0).astype(np.float32)
                targets = np.array([s.target for s in samples_list], dtype=np.int32)
                legal_masks = np.stack([s.legal_mask for s in samples_list], axis=0).astype(np.bool_)
                game_ids = np.array([s.game_id for s in samples_list], dtype="<U64")
                ply_indices = np.array([s.ply_index for s in samples_list], dtype=np.int32)
                outcomes = np.array([s.outcome for s in samples_list], dtype=np.float32)
                outcome_known = np.array([s.outcome_known for s in samples_list], dtype=np.bool_)

            split_file = out_path / f"{split_name}.npz"
            np.savez_compressed(
                split_file,
                states=states,
                targets=targets,
                legal_masks=legal_masks,
                game_ids=game_ids,
                ply_indices=ply_indices,
                outcomes=outcomes,
                outcome_known=outcome_known,
            )

        manifest_file = out_path / "manifest.json"
        manifest_file.write_text(json.dumps(result_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return result_summary


class ChessDataset(TorchDataset):
    """
    Dataset wrapper compatible with PyTorch DataLoader and NumPy.
    State is stored in HWC (8, 8, 14) and converted to CHW (14, 8, 8) tensor on __getitem__.
    DataLoader then collates multiple items into NCHW (B, 14, 8, 8).
    """

    def __init__(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        legal_masks: np.ndarray,
        game_ids: Optional[np.ndarray] = None,
        ply_indices: Optional[np.ndarray] = None,
        outcomes: Optional[np.ndarray] = None,
        outcome_known: Optional[np.ndarray] = None,
        to_torch: bool = True,
    ):
        self.states = states
        self.targets = targets
        self.legal_masks = legal_masks
        self.game_ids = game_ids
        self.ply_indices = ply_indices
        self.outcomes = outcomes
        self.outcome_known = outcome_known
        self.to_torch = to_torch and (torch is not None)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        state = self.states[idx]          # HWC (8, 8, 14)
        target = int(self.targets[idx])   # action_idx [0..4671]
        mask = self.legal_masks[idx]      # bool (4672,)
        game_id = str(self.game_ids[idx]) if self.game_ids is not None else ""
        ply = int(self.ply_indices[idx]) if self.ply_indices is not None else 0
        outcome = float(self.outcomes[idx]) if self.outcomes is not None else 0.0
        known = bool(self.outcome_known[idx]) if self.outcome_known is not None else False

        if self.to_torch:
            # Convert HWC (8,8,14) to CHW (14,8,8) float32 tensor
            # When batched by DataLoader, items of shape (14,8,8) become (B,14,8,8) NCHW
            state_tensor = torch.from_numpy(state.transpose(2, 0, 1).copy()).float()
            target_tensor = torch.tensor(target, dtype=torch.long)
            # NPY memmaps are opened read-only; copy the slice before exposing
            # it to Torch, which otherwise warns about a non-writable array.
            mask_tensor = torch.from_numpy(np.array(mask, copy=True)).bool()
            outcome_tensor = torch.tensor([outcome], dtype=torch.float32)
            outcome_known_tensor = torch.tensor(known, dtype=torch.bool)
            return {
                "state": state_tensor,
                "target": target_tensor,
                "legal_mask": mask_tensor,
                "outcome": outcome_tensor,
                "outcome_known": outcome_known_tensor,
                "game_id": game_id,
                "ply_index": ply,
            }

        return {
            "state": state,
            "target": target,
            "legal_mask": mask,
            "outcome": outcome,
            "outcome_known": known,
            "game_id": game_id,
            "ply_index": ply,
        }


_DATASET_ARRAY_KEYS = (
    "states",
    "targets",
    "legal_masks",
    "game_ids",
    "ply_indices",
    "outcomes",
    "outcome_known",
)
_REQUIRED_DATASET_ARRAY_KEYS = ("states", "targets", "legal_masks")


def _npy_split_paths(dataset_dir: Path, split: str) -> Optional[Dict[str, Path]]:
    """Resolve a split stored as flat or per-split NPY arrays."""
    layouts = (
        {key: dataset_dir / f"{split}_{key}.npy" for key in _DATASET_ARRAY_KEYS},
        {key: dataset_dir / split / f"{key}.npy" for key in _DATASET_ARRAY_KEYS},
    )
    for paths in layouts:
        if all(paths[key].is_file() for key in _REQUIRED_DATASET_ARRAY_KEYS):
            return paths
    return None


def dataset_split_exists(dataset_dir: Union[str, Path], split: str) -> bool:
    """Return whether a split is available as NPZ or supported NPY arrays."""
    d = Path(dataset_dir)
    return (d / f"{split}.npz").is_file() or _npy_split_paths(d, split) is not None


def load_dataset(dataset_dir: Union[str, Path], split: str = "train", to_torch: bool = True) -> ChessDataset:
    """Load an NPZ split or read-only NPY memmaps and return a ChessDataset.

    NPY splits may use ``<split>_<key>.npy`` in the dataset directory or
    ``<split>/<key>.npy``. The required keys are states, targets and
    legal_masks; the remaining dataset fields are optional.
    """
    d = Path(dataset_dir)
    npz_path = d / f"{split}.npz"

    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {
                key: data[key] if key in data else None
                for key in _DATASET_ARRAY_KEYS
            }
    else:
        npy_paths = _npy_split_paths(d, split)
        if npy_paths is None:
            raise FileNotFoundError(
                f"Split '{split}' not found in {d}: expected {split}.npz, "
                f"{split}_<array>.npy, or {split}/<array>.npy"
            )
        arrays = {
            key: np.load(path, mmap_mode="r", allow_pickle=False) if path.is_file() else None
            for key, path in npy_paths.items()
        }

    lengths = {key: len(arrays[key]) for key in _REQUIRED_DATASET_ARRAY_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset split '{split}' has inconsistent required array lengths: {lengths}")

    return ChessDataset(
        states=arrays["states"],
        targets=arrays["targets"],
        legal_masks=arrays["legal_masks"],
        game_ids=arrays["game_ids"],
        ply_indices=arrays["ply_indices"],
        outcomes=arrays["outcomes"],
        outcome_known=arrays["outcome_known"],
        to_torch=to_torch,
    )
