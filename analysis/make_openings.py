"""Sinh bộ 100 thế mở đầu thứ hai (openings_100_b.json) để kiểm tra độ nhạy (Task T8)."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import random
import sys

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games.chess.board import ChessBoard


SEED = 20261201
SRC_BENCH = ROOT / "scripts" / "benchmark_stage6_vs_stage7.py"
EXISTING_OPENINGS = ROOT / "thực nghiệm" / "chess" / "openings_100.json"
OUT_OPENINGS = ROOT / "thực nghiệm" / "chess" / "openings_100_b.json"


def load_opening_prefixes(src_path: Path) -> list[list[str]]:
    """Đọc biến OPENING_BOOK bằng phân tích AST file nguồn, không import module."""
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OPENING_BOOK":
                    data = ast.literal_eval(node.value)
                    if not isinstance(data, list) or len(data) != 50:
                        raise ValueError("OPENING_BOOK phải có đúng 50 tiền tố")
                    return data
    raise ValueError(f"Không tìm thấy OPENING_BOOK trong {src_path}")


def fen_identity(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def load_existing_identities(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for moves in payload["openings"]:
        board = ChessBoard(claim_draw=True)
        for uci in moves:
            board.step(uci)
        seen.add(fen_identity(board.board))
    return seen


def extend_prefix(prefix: list[str], rng: random.Random) -> tuple[list[str], str] | None:
    board = ChessBoard(claim_draw=True)
    for uci in prefix:
        move = chess.Move.from_uci(uci)
        if not board.board.is_legal(move) or board.is_terminal():
            return None
        board.step(move)

    moves = list(prefix)
    while len(moves) < 8:
        if board.is_terminal():
            return None
        quiet_moves = [
            m
            for m in board.board.legal_moves
            if not board.board.is_capture(m) and not board.board.gives_check(m)
        ]
        if not quiet_moves:
            return None
        chosen = rng.choice(quiet_moves)
        board.step(chosen)
        moves.append(chosen.uci())

    if board.is_terminal() or board.board.turn != chess.WHITE:
        return None
    return moves, fen_identity(board.board)


def verify_openings(openings: list[list[str]], existing_ids: set[str]) -> None:
    if len(openings) != 100:
        raise ValueError(f"Số thế mở đầu phải bằng 100, nhận được {len(openings)}")
    seen: set[str] = set()
    for idx, moves in enumerate(openings):
        if len(moves) != 8:
            raise ValueError(f"Thế {idx} có độ dài {len(moves)} != 8")
        board = ChessBoard(claim_draw=True)
        for uci in moves:
            move = chess.Move.from_uci(uci)
            if not board.board.is_legal(move):
                raise ValueError(f"Thế {idx} có nước không hợp lệ: {uci}")
            board.step(move)
        if board.is_terminal():
            raise ValueError(f"Thế {idx} đã kết thúc")
        if board.board.turn != chess.WHITE:
            raise ValueError(f"Thế {idx} không phải Trắng đi tiếp")
        ident = fen_identity(board.board)
        if ident in seen or ident in existing_ids:
            raise ValueError(f"Thế {idx} bị trùng FEN: {ident}")
        seen.add(ident)


def main() -> None:
    prefixes = load_opening_prefixes(SRC_BENCH)
    existing_ids = load_existing_identities(EXISTING_OPENINGS)
    rng = random.Random(SEED)

    openings: list[list[str]] = []
    seen_new: set[str] = set()

    # Mỗi tiền tố 2 thế
    for prefix in prefixes:
        count = 0
        attempts = 0
        while count < 2 and attempts < 200:
            attempts += 1
            res = extend_prefix(prefix, rng)
            if res is None:
                continue
            moves, ident = res
            if ident in existing_ids or ident in seen_new:
                continue
            seen_new.add(ident)
            openings.append(moves)
            count += 1

    # Sinh bù nếu chưa đủ 100
    idx = 0
    attempts = 0
    while len(openings) < 100 and attempts < 5000:
        attempts += 1
        prefix = prefixes[idx % len(prefixes)]
        idx += 1
        res = extend_prefix(prefix, rng)
        if res is None:
            continue
        moves, ident = res
        if ident in existing_ids or ident in seen_new:
            continue
        seen_new.add(ident)
        openings.append(moves)

    verify_openings(openings, existing_ids)

    payload = {
        "description": "100 alternative fixed legal 8-ply starts generated from OPENING_BOOK with quiet non-checking extensions (seed 20261201), disjoint from openings_100.json.",
        "generator_seed": SEED,
        "openings": openings,
    }
    OUT_OPENINGS.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Đã tạo và kiểm tra hợp lệ {len(openings)} thế mở đầu tại {OUT_OPENINGS}")


if __name__ == "__main__":
    main()
