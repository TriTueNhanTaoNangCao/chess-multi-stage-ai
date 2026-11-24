"""Read ranked Jan--Mar 2024 candidates without splitting or extracting samples.

The caller must finish processing a yielded candidate and call accept_candidate
only if downstream processing accepts it, before requesting the next candidate.
Rejected candidates never reserve a fingerprint, so a later duplicate remains
eligible. No PGN or dataset is written by this module.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterator, Mapping

import chess
import chess.pgn

from scripts.stage4.pgn_candidates_300k import group_of, ranked_offsets


MONTHS = ("2024-01", "2024-02", "2024-03")


@dataclass(frozen=True)
class PGNCandidate:
    month: str
    path: Path
    offset: int
    game: chess.pgn.Game
    moves_uci: tuple[str, ...]
    fingerprint: str


def _standard_mainline(game: chess.pgn.Game) -> tuple[str, ...] | None:
    """Validate the full mainline without creating state/action tensors."""
    if game.errors or group_of(game.headers) is None:
        return None
    try:
        board = game.board()
        if (board.chess960 or not board.is_valid()
                or board.fen(en_passant="fen") != chess.STARTING_FEN):
            return None
        moves = tuple(game.mainline_moves())
        if not moves:
            return None
        for move in moves:
            if not board.is_legal(move):
                return None
            board.push(move)
    except (ValueError, IndexError):
        return None
    return tuple(move.uci() for move in moves)


def iter_candidates(
    pgn_paths: Mapping[str, str | Path],
    seed: int,
    accepted_fingerprints: set[str],
    *,
    month_done: Callable[[str], bool] | None = None,
) -> Iterator[PGNCandidate]:
    """Yield candidates in month order, using ranked_offsets within each month.

Supply exactly the three month keys in MONTHS, each with its actual PGN path.
The same explicit seed is passed to each month's independent ranking. Offsets
are reopened with the text encoding used by ranked_offsets. The shared accepted
set is consulted across all months and is never modified by this iterator.
An optional month_done callback stops reading the rest of a filled month.
"""
    if set(pgn_paths) != set(MONTHS):
        raise ValueError(f"Expected PGN paths for exactly {MONTHS}")
    paths = {month: Path(pgn_paths[month]) for month in MONTHS}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    for month in MONTHS:
        path = paths[month]
        offsets = ranked_offsets(path, seed)
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for offset in offsets:
                if month_done is not None and month_done(month):
                    break
                stream.seek(offset)
                try:
                    game = chess.pgn.read_game(stream)
                except (ValueError, IndexError):
                    continue
                if game is None:
                    continue
                moves_uci = _standard_mainline(game)
                if moves_uci is None:
                    continue
                fingerprint = hashlib.sha256(" ".join(moves_uci).encode("utf-8")).hexdigest()
                if fingerprint in accepted_fingerprints:
                    continue
                yield PGNCandidate(month, path, offset, game, moves_uci, fingerprint)


def accept_candidate(candidate: PGNCandidate, accepted_fingerprints: set[str]) -> bool:
    """Call only AFTER downstream sample acceptance; return False if already used.

If downstream processing rejects a candidate, do not call this function. Share
the same set with iter_candidates so later games/months skip accepted duplicates.
"""
    if candidate.fingerprint in accepted_fingerprints:
        return False
    accepted_fingerprints.add(candidate.fingerprint)
    return True
