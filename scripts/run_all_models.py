"""Resumable, color-paired benchmark. Run --help; paths are project-relative."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.evaluator import restore_initial_board
from arena.session import GameSession, measure_move
from arena.tournament import update_elo
from games.chess.board import ChessBoard
from services.history import HistoryStore, replay, to_pgn, utc_now


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def source_hash():
    paths = [p for folder in ("agents", "arena", "games", "models", "services", "training", "scripts")
             for p in (ROOT / folder).rglob("*.py")]
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)})


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def load_config(path, selected=None, smoke=False):
    config = json.loads(project_path(path).read_text(encoding="utf-8-sig"))
    required = {"schema_version", "seed", "games_per_pair", "max_plies", "claim_draw", "device",
                "torch_threads", "move_timeout_seconds", "startup_timeout_seconds", "openings_file",
                "output_dir", "agents"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError(f"Configuration keys must be exactly {sorted(required)}; schema_version=1")
    for name in ("games_per_pair", "max_plies", "torch_threads", "move_timeout_seconds", "startup_timeout_seconds"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0 or config["games_per_pair"] % 2:
        raise ValueError("seed must be nonnegative; games_per_pair must be even")
    if not isinstance(config["claim_draw"], bool) or config["device"] not in ("cpu", "cuda"):
        raise ValueError("claim_draw must be boolean; device must be cpu or cuda")
    from agents import AGENT_REGISTRY
    agents = config["agents"]
    if not isinstance(agents, list) or len(agents) < 2:
        raise ValueError("At least two agent configurations required")
    if not all(isinstance(a, dict) and isinstance(a.get("id"), str) for a in agents):
        raise ValueError("Each agent must be an object with a string ID")
    ids = [a["id"] for a in agents]
    if any(not isinstance(x, str) or not x or not x.replace("_", "").isalnum() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Agent IDs must be unique alphanumeric/underscore strings")
    for agent in agents:
        if set(agent) != {"id", "key", "kwargs"} or agent["key"] not in AGENT_REGISTRY or not isinstance(agent["kwargs"], dict):
            raise ValueError(f"Invalid agent: {agent}")
        if {"seed", "device", "name", "player_id"} & set(agent["kwargs"]):
            raise ValueError("seed/device/name/player_id are controlled by the runner")
        needed = {"supervised": ("checkpoint",), "rl_policy": ("checkpoint",),
                  "alphago": ("checkpoint_policy", "checkpoint_value"), "alphazero": ("checkpoint",)}
        if any(not agent["kwargs"].get(k) for k in needed.get(agent["key"], ())):
            raise ValueError(f"{agent['id']}: explicit trained checkpoint paths required")
    if selected:
        if set(selected) - set(ids):
            raise ValueError(f"Unknown agent IDs: {set(selected) - set(ids)}")
        config["agents"] = [a for a in agents if a["id"] in selected]
        if len(config["agents"]) < 2:
            raise ValueError("Select at least two agents")
    openings = json.loads(project_path(config["openings_file"]).read_text(encoding="utf-8"))["openings"]
    if smoke:
        config["games_per_pair"], config["max_plies"] = 2, min(config["max_plies"], 8)
    count = config["games_per_pair"] // 2
    if len(openings) < count:
        raise ValueError(f"Need at least {count} openings")
    openings = openings[:count]
    positions = set()
    for opening in openings:
        board = opening_board(opening, config["claim_draw"])
        if board.is_terminal() or not board.turn or len(opening) % 2:
            raise ValueError("Each opening must be nonterminal with White to move and even plies")
        identity = " ".join(board.fen().split()[:4])
        if identity in positions:
            raise ValueError("Duplicate opening positions")
        positions.add(identity)
    config["openings"] = openings
    config["mode"] = "smoke" if smoke else "official"
    return config


def opening_board(moves, claim_draw):
    board = ChessBoard(claim_draw=claim_draw)
    for move in moves:
        board.step(move)
    return board


def schedule(config):
    games = []
    for a, b in combinations(config["agents"], 2):
        pair = f"{a['id']}__{b['id']}"
        for index, opening in enumerate(config["openings"]):
            # Seeds belong to agent + opening, independent of opponent or color.
            seeds = {x["id"]: int(digest([config["seed"], x["id"], index])[:8], 16) % 2147483647 for x in (a, b)}
            for leg, (white, black) in enumerate(((a, b), (b, a))):
                games.append(dict(id=f"{pair}__o{index:03d}__l{leg}", pair=pair, opening_index=index,
                                  opening=opening, leg=leg, white=white, black=black, seeds=seeds))
    return games


def provenance(config):
    hashes, missing = {}, []
    for agent in config["agents"]:
        for key, value in agent["kwargs"].items():
            if key.startswith("checkpoint"):
                path = project_path(value)
                if not path.is_file():
                    missing.append(f"{agent['id']}.{key}: {path}")
                else:
                    hashes[f"{agent['id']}.{key}"] = file_hash(path)
    env = {name: version(name) for name in ("chess", "numpy", "torch")}
    env["python"], env["platform"] = platform.python_version(), platform.platform()
    env["cpu"] = platform.processor() or None
    try:
        import psutil
        env["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        env["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        env["ram_gb"] = round(psutil.virtual_memory().total / (2**30), 1)
    except ImportError:
        env["cpu_cores_physical"] = None
        env["cpu_cores_logical"] = None
        env["ram_gb"] = None
    try:
        import torch
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["cuda"] = torch.version.cuda
    except Exception:
        env["gpu"] = None
        env["cuda"] = None
    frozen = dict(config=config, checkpoint_sha256=hashes, source_sha256=source_hash(), environment=env)
    frozen["fingerprint"] = digest(frozen)
    return frozen, missing


def build_agent(spec, seed, device, claim_draw=True):
    from agents import AGENT_REGISTRY
    kwargs = dict(spec["kwargs"])
    for key in list(kwargs):
        if key.startswith("checkpoint"):
            kwargs[key] = str(project_path(kwargs[key]))
    if spec["key"] in ("supervised", "rl_policy", "alphago", "alphazero"):
        import torch
        expected = {"checkpoint_policy": "CNNPolicyNetwork", "checkpoint_value": "ValueNetwork",
                    "checkpoint": "DualHeadResNet" if spec["key"] == "alphazero" else "CNNPolicyNetwork"}
        for key, value in kwargs.items():
            if key.startswith("checkpoint"):
                payload = torch.load(value, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict):
                    raise ValueError(f"{spec['id']}: expected a metadata/state_dict checkpoint")
                required = {"model_class": expected[key], "encoder_version": "hwc14-white-v1",
                            "claim_draw": claim_draw}
                if expected[key] != "ValueNetwork":
                    required.update(action_version="az73-rankflip-v1", num_actions=4672)
                for field, want in required.items():
                    if payload.get(field) != want:
                        raise ValueError(f"{spec['id']}: incompatible {field} in {value}")
                if expected[key] in ("ValueNetwork", "DualHeadResNet") and payload.get("value_perspective") != "side_to_move":
                    raise ValueError(f"{spec['id']}: incompatible value perspective")
                if "state_dict" not in payload:
                    raise ValueError(f"{spec['id']}: missing state_dict")
                del payload
        kwargs["device"] = device
    if spec["key"] not in ("minimax", "alphabeta"):
        kwargs["seed"] = seed
    agent = AGENT_REGISTRY[spec["key"]](**kwargs)
    agent.name = spec["id"]
    return agent


def _worker(connection, config, job, check_only=False):
    try:
        import torch
        torch.set_num_threads(config["torch_threads"])
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if check_only:
            agent = build_agent(job, config["seed"], config["device"], config["claim_draw"])
            agent.reset()
            connection.send(("checked", job["id"]))
            return
        agents = [build_agent(job[side], job["seeds"][job[side]["id"]], config["device"], config["claim_draw"])
                  for side in ("white", "black")]
        game = GameSession(agents[0].name, agents[1].name,
                           board=opening_board(job["opening"], config["claim_draw"]), max_plies=config["max_plies"])
        game.reset_agents(agents)
        connection.send(("snapshot", game.record()))
        while not game.finished:
            game.accept(*measure_move(agents[game.index], game.state))
            connection.send(("snapshot", game.record()))
        connection.send(("done", game.record()))
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def timeout_record(record, seconds):
    record = deepcopy(record)
    board = replay(record)
    white = board.turn
    side = "white" if white else "black"
    error = f"Move exceeded {seconds}s wall-clock budget; worker terminated"
    record.update(result="0-1" if white else "1-0", reason="forfeit", error=error)
    record[side + "_seconds"] += seconds
    record[side + "_calls"] += 1
    record["telemetry"].append(dict(ply=len(record["moves"]) + 1, color=side, seconds=seconds, error=error))
    record["timeout"] = True
    return record


def isolated_job(config, job, check_only=False):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_worker, args=(sender, config, job, check_only))
    worker.start()
    sender.close()
    snapshot = None
    try:
        while True:
            budget = config["startup_timeout_seconds"] if snapshot is None else config["move_timeout_seconds"]
            if not receiver.poll(budget):
                if snapshot is None or snapshot["reason"]:
                    raise RuntimeError("Worker startup/finalization timeout; no completed game saved")
                return timeout_record(snapshot, budget)
            try:
                kind, payload = receiver.recv()
            except EOFError as exc:
                raise RuntimeError(f"Worker exited unexpectedly (exit code {worker.exitcode}); rerun to retry game") from exc
            if kind == "error":
                raise RuntimeError(payload)
            if kind in ("checked", "done"):
                return payload
            snapshot = payload
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=10)
        if worker.is_alive():
            worker.kill()
            worker.join()
        receiver.close()


def preflight(config):
    frozen, missing = provenance(config)
    if missing:
        raise ValueError("Missing chess checkpoints (no random fallback allowed):\n  " + "\n  ".join(missing))
    for spec in config["agents"]:
        print(f"Checking {spec['id']}...", flush=True)
        isolated_job(config, spec, check_only=True)
    return frozen


@contextmanager
def run_lock(output):
    with (output / ".runner.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another runner is using this output directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def bind_manifest(output, frozen):
    path = output / "manifest.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored["fingerprint"] != frozen["fingerprint"]:
            raise ValueError("Resume refused: config, openings, checkpoint, code or environment changed. Use a new output directory.")
    else:
        if (output / "history.sqlite3").exists():
            raise ValueError("Database exists without manifest; choose a new output directory")
        atomic_json(path, frozen)


def validate_records(records, config, fingerprint, require_complete=False):
    expected = {j["id"]: j for j in schedule(config)}
    seen = set()
    for record in records:
        key = record["id"]
        if key in seen or key not in expected or record.get("run_id") != fingerprint:
            raise ValueError(f"Unknown/duplicate/mismatched game ID: {key}")
        seen.add(key)
        job = expected[key]
        if record["result"] not in ("1-0", "0-1", "1/2-1/2") or not record["reason"]:
            raise ValueError(f"Unfinished record: {key}")
        for side in ("white", "black"):
            if record[side] != job[side]["id"]:
                raise ValueError(f"Wrong player/color: {key}")
        if record["initial_moves"] != job["opening"] or record["claim_draw"] != config["claim_draw"]:
            raise ValueError(f"Wrong opening/draw policy: {key}")
        if restore_initial_board(record).fen() != opening_board(job["opening"], config["claim_draw"]).fen():
            raise ValueError(f"Wrong starting position: {key}")
        if record.get("seeds") != job["seeds"] or len(record["moves"]) > config["max_plies"]:
            raise ValueError(f"Wrong seeds/ply count: {key}")
        telemetry = record["telemetry"]
        if [entry["uci"] for entry in telemetry if "uci" in entry] != record["moves"]:
            raise ValueError(f"Move telemetry mismatch: {key}")
        for side in ("white", "black"):
            entries = [entry for entry in telemetry if entry["color"] == side]
            if record[side + "_calls"] != len(entries) or abs(record[side + "_seconds"] - sum(e["seconds"] for e in entries)) > 1e-6:
                raise ValueError(f"Timing/call telemetry mismatch: {key}")
        if replay(record).fen() != record["final_fen"]:
            raise ValueError(f"Replay mismatch: {key}")
        board = replay(record)
        outcome = board.outcome()
        if record["reason"] == "forfeit":
            if not record.get("error") or outcome is not None or record["result"] != ("0-1" if board.turn else "1-0"):
                raise ValueError(f"Invalid forfeit: {key}")
        elif record["reason"] == "max_plies_adjudication":
            if record["result"] != "1/2-1/2" or len(record["moves"]) != config["max_plies"] or outcome is not None:
                raise ValueError(f"Invalid ply cap: {key}")
        elif record["reason"] not in ("reset_forfeit", "double_forfeit"):
            if outcome is None or outcome.result() != record["result"] or outcome.termination.name.lower() != record["reason"]:
                raise ValueError(f"Outcome mismatch: {key}")
    if require_complete and len(seen) != len(expected):
        raise ValueError(f"Incomplete benchmark: {len(seen)}/{len(expected)} games")
    return len(expected)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def statistics(records, agents):
    totals = {}
    for name in agents:
        totals[name] = dict(agent=name, games=0, wins=0, draws=0, losses=0, white_games=0, white_wins=0,
                            white_draws=0, white_losses=0, black_games=0, black_wins=0, black_draws=0,
                            black_losses=0, seconds=0.0, calls=0, elo=1200.0, timeout_games=0, error_games=0)
    for record in records:
        score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[record["result"]]
        white, black = totals[record["white"]], totals[record["black"]]
        white["elo"], black["elo"] = update_elo(white["elo"], black["elo"], score)
        for side, row, value in (("white", white, score), ("black", black, 1-score)):
            outcome = "wins" if value == 1 else "losses" if value == 0 else "draws"
            row["games"] += 1
            row[outcome] += 1
            row[side + "_games"] += 1
            row[side + "_" + outcome] += 1
            row["seconds"] += record[side + "_seconds"]
            row["calls"] += record[side + "_calls"]
            row["timeout_games"] += bool(record.get("timeout"))
            row["error_games"] += bool(record.get("error"))
    for row in totals.values():
        for prefix in ("", "white_", "black_"):
            games = row[prefix + "games"]
            for metric, numerator in (("win_pct", row[prefix + "wins"]), ("draw_pct", row[prefix + "draws"]),
                                      ("score_pct", row[prefix + "wins"] + row[prefix + "draws"] / 2)):
                row[prefix + metric] = round(100 * numerator / games, 4) if games else None
        seconds = row.pop("seconds")
        row["ms_per_move"] = round(1000 * seconds / row["calls"], 3) if row["calls"] else None
        row.pop("calls")
        row["elo"] = round(row["elo"], 2)
    return sorted(totals.values(), key=lambda r: (-r["elo"], r["agent"]))


def export_report(output, frozen, store, full=True):
    config = frozen["config"]
    jobs = schedule(config)
    indexed = {r["id"]: r for r in store.list_games()}
    if set(indexed) - {j["id"] for j in jobs}:
        raise ValueError("Database contains games outside the frozen schedule")
    records = [indexed[j["id"]] for j in jobs if j["id"] in indexed]
    agents = [a["id"] for a in config["agents"]]
    rows = statistics(records, agents)
    matchups = []
    for a, b in combinations(agents, 2):
        subset = [r for r in records if {r["white"], r["black"]} == {a, b}]
        for row in statistics(subset, (a, b)):
            matchups.append(dict(pair=f"{a}__{b}", **row))
    errors = sum(bool(r.get("error")) for r in records)
    summary = dict(completed_games=len(records), expected_games=len(jobs),
                   complete=len(records) == len(jobs), clean=errors == 0,
                   mode=config["mode"], error_games=errors,
                   timeout_games=sum(bool(r.get("timeout")) for r in records),
                   percent_unit="0..100", paired_games_are_not_independent=True,
                   checkpoint_sha256=frozen["checkpoint_sha256"], fingerprint=frozen["fingerprint"])
    atomic_json(output / "progress.json", summary)
    if full:
        atomic_json(output / "tournament_results.json", dict(config=config, summary=summary, leaderboard=rows,
                                                            matchups=matchups, records=records))
    write_csv(output / "leaderboard.csv", rows)
    write_csv(output / "matchups.csv", matchups)
    lines = ["# All-model chess benchmark", "", f"Mode: {config['mode']}; completed {len(records)}/{len(jobs)} games.",
             f"Error games: {errors}; timeout games: {summary['timeout_games']}.",
             "Percentages use 0..100. Elo is local, schedule-dependent. Paired games are not independent.", "",
             "| Agent | W / D / L | Win % | Score % | White win % | Black win % | Elo | ms/move |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['agent']} | {row['wins']}/{row['draws']}/{row['losses']} | {row['win_pct']} | {row['score_pct']} | {row['white_win_pct']} | {row['black_win_pct']} | {row['elo']} | {row['ms_per_move']} |")
    (output / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if full:
        temp = output / "games.pgn.tmp"
        with temp.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(to_pgn(record) + "\n\n")
        temp.replace(output / "games.pgn")
    return summary


def run(config, output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        frozen = preflight(config)
        bind_manifest(output, frozen)
        store = HistoryStore(output / "history.sqlite3")
        records = store.list_games()
        validate_records(records, config, frozen["fingerprint"])
        completed = {r["id"] for r in records}
        jobs = schedule(config)
        count = 0
        try:
            for job in jobs:
                if job["id"] in completed:
                    continue
                if limit is not None and count >= limit:
                    break
                started = utc_now()
                record = isolated_job(config, job)
                record.update(id=job["id"], run_id=frozen["fingerprint"], started_at=started, ended_at=utc_now(),
                              mode="benchmark", pair=job["pair"], opening_index=job["opening_index"],
                              leg=job["leg"], seeds=job["seeds"], max_plies=config["max_plies"],
                              configs={side: dict(label=job[side]["id"], **job[side])
                                       for side in ("white", "black")})
                validate_records([record], config, frozen["fingerprint"])
                store.save(record)
                completed.add(job["id"])
                count += 1
                message = f"[{len(completed)}/{len(jobs)}] {job['id']}: {record['result']} ({record['reason']})"
                print(message, flush=True)
                with (output / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"{utc_now()} {message}\n")
                atomic_json(output / "progress.json", dict(completed_games=len(completed), expected_games=len(jobs),
                                                            last_game=job["id"], updated_at=utc_now()))
                if count % 100 == 0:
                    export_report(output, frozen, store, full=False)
        except BaseException as exc:
            with (output / "run.log").open("a", encoding="utf-8") as stream:
                stream.write(f"{utc_now()} INTERRUPTED {type(exc).__name__}: {exc}\n")
            raise
        finally:
            export_report(output, frozen, store)
        return len(completed) == len(jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "smoke", "run", "report", "verify"))
    parser.add_argument("--config", default="thực nghiệm/chess/experiment_cfg.json")
    parser.add_argument("--agents", nargs="+", help="Subset of configuration IDs; use a separate output directory")
    parser.add_argument("--output", help="Project-relative or absolute output directory")
    parser.add_argument("--limit", type=int, help="Maximum NEW games this invocation; rerun to continue")
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.command in ("report", "verify"):
            if not args.output:
                raise ValueError("report/verify require --output")
            output = project_path(args.output)
            frozen = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            with run_lock(output):
                store = HistoryStore(output / "history.sqlite3")
                validate_records(store.list_games(), frozen["config"], frozen["fingerprint"], args.command == "verify")
                summary = export_report(output, frozen, store)
                print(json.dumps(summary, indent=2))
                if args.command == "verify" and not summary["clean"]:
                    raise ValueError("All games completed but bot errors/timeouts exist; inspect before accepting benchmark")
            return 0
        config = load_config(args.config, args.agents, args.command == "smoke")
        jobs = schedule(config)
        output = project_path(args.output or config["output_dir"] + ("_smoke" if args.command == "smoke" else ""))
        print(f"{len(config['agents'])} agents; {len(config['agents'])*(len(config['agents'])-1)//2} pairs; {len(jobs)} games; mode={config['mode']}")
        print(f"Output: {output}")
        if args.command == "plan":
            _, missing = provenance(config)
            print("Missing checkpoints:\n" + "\n".join(missing) if missing else "Checkpoint paths exist; use check to validate models")
            return 0
        if args.command == "check":
            frozen = preflight(config)
            print("Preflight OK. Fingerprint:", frozen["fingerprint"])
            return 0
        done = run(config, output, args.limit)
        print("COMPLETE; run verify for final acceptance." if done else "PAUSED; rerun the same command to resume.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed games are saved; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    stdout: Any = sys.stdout
    if hasattr(stdout, "reconfigure"):
        stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    raise SystemExit(main())
