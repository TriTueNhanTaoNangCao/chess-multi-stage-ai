"""Durable local match history, replay and exports. No Streamlit dependency."""
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

import chess
import chess.pgn
from arena.evaluator import restore_initial_board


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_import_record(record):
    """Reject malformed display/statistics data before it reaches persistent history."""
    if not isinstance(record, dict):
        raise ValueError("Mỗi bản ghi phải là một object")
    for field in ("white", "black", "initial_fen", "final_fen", "result"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{field} phải là chuỗi không rỗng")
    for field in ("id", "started_at", "run_id", "mode", "reason", "error", "root_fen", "import_note"):
        if field in record and not isinstance(record[field], str):
            raise ValueError(f"{field} phải là chuỗi")
    for field in ("moves", "initial_moves"):
        value = record.get(field, [] if field == "initial_moves" else None)
        if not isinstance(value, list) or not all(isinstance(move, str) for move in value):
            raise ValueError(f"{field} phải là danh sách nước đi")
    if "claim_draw" in record and not isinstance(record["claim_draw"], bool):
        raise ValueError("claim_draw phải là boolean")
    for side in ("white", "black"):
        seconds, calls = record.get(f"{side}_seconds"), record.get(f"{side}_calls")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Thời gian phải là số hữu hạn không âm")
        if type(calls) is not int or calls < 0:
            raise ValueError("Số lượt phải là số nguyên không âm")
    configs = record.get("configs", {})
    if not isinstance(configs, dict):
        raise ValueError("configs phải là object")
    for side in ("white", "black"):
        config = configs.get(side, {})
        if not isinstance(config, dict):
            raise ValueError(f"configs.{side} phải là object")
        for field in ("id", "label"):
            if field in config and (not isinstance(config[field], str) or not config[field]):
                raise ValueError(f"configs.{side}.{field} phải là chuỗi không rỗng")
    telemetry = record.get("telemetry", [])
    if not isinstance(telemetry, list):
        raise ValueError("telemetry phải là danh sách")
    for entry in telemetry:
        if not isinstance(entry, dict) or type(entry.get("ply")) is not int or entry["ply"] < 1:
            raise ValueError("Thống kê nước đi cần ply là số nguyên dương")
        seconds = entry.get("seconds")
        if entry.get("color") not in ("white", "black"):
            raise ValueError("Màu quân trong thống kê không hợp lệ")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Thời gian từng nước phải là số hữu hạn không âm")
        for field in ("san", "uci", "error"):
            if field in entry and not isinstance(entry[field], str):
                raise ValueError(f"telemetry.{field} phải là chuỗi")
    # Also catch non-finite numbers in optional metadata before any INSERT.
    json.dumps(record, allow_nan=False)


def replay(record, ply=None):
    board = restore_initial_board(record)
    moves = record["moves"] if ply is None else record["moves"][:ply]
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if not board.board.is_legal(move):
            raise ValueError(f"Lịch sử chứa nước sai luật: {uci}")
        board.board.push(move)
    return board


def to_pgn(record):
    game = chess.pgn.Game()
    initial = restore_initial_board(record)
    game.setup(initial.board)
    game.headers.update(White=record["white"], Black=record["black"],
                        Result=record["result"], Event=record.get("mode", "Chess AI Arena"),
                        Date=record.get("started_at", "????-??-??")[:10].replace("-", "."))
    game.headers["Termination"] = record.get("reason") or "unterminated"
    node = game
    for uci in record["moves"]:
        move = chess.Move.from_uci(uci)
        if not initial.board.is_legal(move):
            raise ValueError(f"Lịch sử chứa nước sai luật: {uci}")
        initial.board.push(move)
        node = node.add_variation(move)
    return str(game)


def game_rows(records):
    rows = []
    for r in records:
        winner = r["white"] if r["result"] == "1-0" else r["black"] if r["result"] == "0-1" else ""
        rows.append({"id": r["id"], "started_at": r.get("started_at", ""),
                     "white": r["white"], "black": r["black"], "result": r["result"],
                     "winner": winner, "reason": r.get("reason", ""), "plies": len(r["moves"]),
                     "mode": r.get("mode", ""), "run_id": r.get("run_id", ""),
                     "white_seconds": r["white_seconds"], "black_seconds": r["black_seconds"]})
    return rows


def csv_text(rows):
    if not rows:
        return ""
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + output.getvalue()


def standings(records):
    """Only completed results count; separate configurations even for one algorithm."""
    from arena.tournament import update_elo
    totals = {}
    for record in sorted(records, key=lambda r: (r.get("started_at", ""), r["id"])):
        if record["result"] not in ("1-0", "0-1", "1/2-1/2"):
            continue
        players = []
        for side in ("white", "black"):
            config = record.get("configs", {}).get(side, {})
            key = config.get("id", record[side])
            label = config.get("label", record[side])
            row = totals.setdefault(key, dict(agent=label, variant=key, games=0, wins=0,
                                             draws=0, losses=0, seconds=0.0, calls=0, elo=1200.0))
            players.append(row)
        score = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}[record["result"]]
        if players[0] is not players[1]:
            players[0]["elo"], players[1]["elo"] = update_elo(players[0]["elo"], players[1]["elo"], score)
        for side, row, value in zip(("white", "black"), players, (score, 1-score)):
            row["games"] += 1
            row["wins" if value == 1 else "losses" if value == 0 else "draws"] += 1
            row["seconds"] += record[f"{side}_seconds"]
            row["calls"] += record[f"{side}_calls"]
    return sorted([dict(agent=r["agent"], variant=r["variant"], games=r["games"],
                        wins=r["wins"], draws=r["draws"], losses=r["losses"],
                        win_rate=round(r["wins"]/r["games"]*100, 1),
                        ms_per_move=round(1000*r["seconds"]/r["calls"], 2) if r["calls"] else 0,
                        elo=round(r["elo"], 1)) for r in totals.values()], key=lambda r: -r["elo"])


class HistoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL, run_id TEXT NOT NULL,
                payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS games_run ON games(run_id)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, record):
        with self.connect() as db:
            db.execute("""INSERT INTO games VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload""",
                       (record["id"], record["started_at"], record.get("run_id", ""),
                        json.dumps(record, ensure_ascii=False, allow_nan=False)))

    def list_games(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM games ORDER BY started_at DESC, rowid DESC")]

    def import_tournament(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("File phải chứa một object với danh sách records")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("File phải chứa danh sách records")
        prepared = []
        for index, record in enumerate(records):
            _validate_import_record(record)
            final = replay(record)
            if final.fen() != record["final_fen"]:
                raise ValueError(f"Ván {index+1}: FEN cuối không khớp")
            if record["result"] not in ("*", "1-0", "0-1", "1/2-1/2"):
                raise ValueError("Kết quả không hợp lệ")
            fingerprint = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            imported = {**record, "id": record.get("id") or f"import-{fingerprint}-{index}",
                             "started_at": record.get("started_at", utc_now()),
                             "mode": record.get("mode", "import"), "run_id": record.get("run_id", "import"),
                             "import_note": "Thời điểm nhập; file cũ có thể không có ngày thi đấu",
                             "source_config": payload.get("config", {})}
            prepared.append((imported["id"], imported["started_at"], imported["run_id"],
                             json.dumps(imported, ensure_ascii=False, allow_nan=False)))
        # One transaction: a failed batch leaves no partial history. SQLite also
        # handles duplicate IDs within the batch and concurrent imports correctly.
        with self.connect() as db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO games VALUES (?, ?, ?, ?)", prepared)
            return db.total_changes - before


def new_record_id():
    return uuid4().hex
