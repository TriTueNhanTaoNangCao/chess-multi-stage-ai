"""Shared atomic IO, hashing, and fingerprints for Stage 5 Value data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _atomic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _game_fingerprint(root_fen: str, moves_uci: Sequence[str]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {"root_fen": root_fen, "moves_uci": list(moves_uci)}
        ).encode("utf-8")
    )


def _state_fingerprint(state: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"chw-float32-14x8x8-v1\0")
    digest.update(np.ascontiguousarray(state, dtype=np.float32).tobytes())
    return digest.hexdigest()
