"""Header-only candidate filtering and ranking for the Stage 4 300k workflow.

No quota, dataset building, split assignment or position extraction is performed.
Offsets are text-stream seek cookies: reopen with the same UTF-8/error settings
and call stream.seek(offset) before reading a selected game.
"""
from collections import defaultdict
from math import sqrt
from pathlib import Path
import re
from typing import Mapping

import chess.pgn
import numpy as np


def group_of(headers: Mapping[str, str]) -> tuple[str, str] | None:
    """Return (first two ECO characters, format), or None for rejected headers.

Only ordinary standard-start PGNs are candidates; FEN/SetUp games are excluded.
Format comes from explicit EventType/Event labels, not guessed time-control
thresholds. Unlabelled formats are grouped as unknown. No Elo filter is added.
"""
    variant = headers.get("Variant", "Standard").strip().lower()
    if variant not in {"standard", "chess"}:
        return None
    if headers.get("Result", "").strip() not in {"1-0", "0-1", "1/2-1/2"}:
        return None
    if headers.get("FEN", "").strip() or headers.get("SetUp", "").strip() == "1":
        return None

    labels = [headers.get(key, "").strip().lower() for key in ("EventType", "Event")]
    if any(re.search(r"\b(?:ultrabullet|bullet)\b", label) for label in labels):
        return None
    speed = "unknown"
    for label in labels:
        match = re.search(r"\b(blitz|rapid|classical)\b", label)
        if match:
            speed = match.group(1)
            break
    eco = headers.get("ECO", "").strip().upper()
    return (eco[:2] if eco else "??", speed)


def ranked_offsets(pgn_path: str | Path, seed: int) -> list[int]:
    """Rank all accepted offsets using exponential keys with scale=sqrt(group n).

Groups retain their first-appearance order, each group's games retain file
order, and a dedicated NumPy Generator draws one key per accepted game. The
final stable sort preserves original file order when keys are equal. Header
acceptance does not certify that the movetext is legal; that remains a later
pipeline step. No files are written.
"""
    offsets: list[int] = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    with Path(pgn_path).open("r", encoding="utf-8", errors="replace") as stream:
        while True:
            offset = stream.tell()
            headers = chess.pgn.read_headers(stream)
            if headers is None:
                break
            group = group_of(headers)
            if group is not None:
                groups[group].append(len(offsets))
                offsets.append(offset)

    rng = np.random.default_rng(seed)
    keys = np.empty(len(offsets), dtype=np.float64)
    for indices in groups.values():
        n = len(indices)
        keys[indices] = rng.exponential(scale=sqrt(n), size=n)
    order = np.argsort(keys, kind="stable")
    return [offsets[int(index)] for index in order]
