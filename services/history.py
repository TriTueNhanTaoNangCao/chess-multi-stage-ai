"""Durable local match history, replay and exports. No Streamlit dependency."""
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import chess
import chess.pgn
from arena.evaluator import restore_initial_board


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("File phải chứa danh sách records")
        prepared = []
        for index, record in enumerate(records):
            for side in ("white", "black"):
                if not isinstance(record[side], str):
                    raise ValueError("Tên người chơi phải là chuỗi")
                if record[f"{side}_seconds"] < 0 or record[f"{side}_calls"] < 0:
                    raise ValueError("Thời gian và số lượt phải không âm")
            final = replay(record)
            if final.fen() != record["final_fen"]:
                raise ValueError(f"Ván {index+1}: FEN cuối không khớp")
            if record["result"] not in ("*", "1-0", "0-1", "1/2-1/2"):
                raise ValueError("Kết quả không hợp lệ")
            fingerprint = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            prepared.append({**record, "id": record.get("id") or f"import-{fingerprint}-{index}",
                             "started_at": record.get("started_at", utc_now()),
                             "mode": record.get("mode", "import"), "run_id": record.get("run_id", "import"),
                             "import_note": "Thời điểm nhập; file cũ có thể không có ngày thi đấu",
                             "source_config": payload.get("config", {})})
        existing = {r["id"] for r in self.list_games()}
        for record in prepared:
            if record["id"] not in existing:
                self.save(record)
        return sum(r["id"] not in existing for r in prepared)


def new_record_id():
    return uuid4().hex
