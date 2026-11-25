"""Balanced reduced benchmark with a wall-clock deadline; preserves original runs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import multiprocessing as mp
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import run_all_models as core


class BudgetExpired(Exception):
    """Experiment deadline, not a bot loss."""


def config_for_budget(path, games):
    if games < 2 or games % 2:
        raise ValueError("games-per-pair must be even and >=2")
    config = core.load_config(path)
    if games > config["games_per_pair"]:
        raise ValueError("Reduced game count must not exceed original schedule")
    config["games_per_pair"] = games
    config["openings"] = config["openings"][:games // 2]
    config["mode"] = "reduced_time_budget"
    return config


def frozen_manifest(config):
    frozen, missing = core.provenance(config)
    frozen.pop("fingerprint")
    frozen["budget_runner_sha256"] = core.file_hash(Path(__file__))
    frozen["fingerprint"] = core.digest(frozen)
    return frozen, missing


def read_records(folder):
    path = folder / "history.sqlite3"
    if not path.exists():
        raise ValueError(f"Missing database: {path}")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=15) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT payload FROM games")]


def choose_reuse(folder, target):
    """Only import comparable local records, always as complete color pairs."""
    source = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for key in ("source_sha256", "environment"):
        if source[key] != target[key]:
            raise ValueError(f"Cannot reuse {folder}: {key} differs. Keep it as a separate report.")
    old, new = source["config"], target["config"]
    if old.get("move_time_limit_seconds") != new.get("move_time_limit_seconds"):
        raise ValueError("Cannot reuse a different per-move time limit")
    for key in ("seed", "max_plies", "claim_draw", "device", "torch_threads", "move_timeout_seconds"):
        if old[key] != new[key]:
            raise ValueError(f"Cannot reuse {folder}: {key} differs")
    if old["openings"][:len(new["openings"])] != new["openings"]:
        raise ValueError("Cannot reuse different openings")
    specs = {a["id"]: a for a in new["agents"]}
    for agent in old["agents"]:
        if agent["id"] not in specs or agent != specs[agent["id"]]:
            raise ValueError(f"Cannot reuse different agent configuration: {agent['id']}")
    for key, value in source["checkpoint_sha256"].items():
        if target["checkpoint_sha256"].get(key) != value:
            raise ValueError(f"Cannot reuse different checkpoint: {key}")
    desired = {job["id"] for job in core.schedule(new)}
    selected = [r for r in read_records(folder) if r["id"] in desired]
    core.validate_records(selected, old, source["fingerprint"])
    index = {r["id"]: r for r in selected}
    result = []
    jobs = core.schedule(new)
    for first, second in zip(jobs[::2], jobs[1::2]):
        if first["id"] in index and second["id"] in index:
            for job in (first, second):
                record = deepcopy(index[job["id"]])
                record["origin"] = dict(folder=str(folder), run_id=source["fingerprint"], game_id=record["id"])
                record["run_id"] = target["fingerprint"]
                result.append(record)
    return result


def save_pair(store, records):
    if len(records) != 2:
        raise ValueError("Must save both color legs together")
    with store.connect() as db:
        db.executemany("INSERT INTO games VALUES (?, ?, ?, ?)",
                       [(r["id"], r["started_at"], r["run_id"], json.dumps(r, ensure_ascii=False, allow_nan=False))
                        for r in records])


def balanced(records, config, fingerprint):
    core.validate_records(records, config, fingerprint)
    ids = {r["id"] for r in records}
    jobs = core.schedule(config)
    for a, b in zip(jobs[::2], jobs[1::2]):
        if (a["id"] in ids) != (b["id"] in ids):
            raise ValueError(f"Unbalanced color pair: {a['pair']}, opening {a['opening_index']}")


def run_game(config, job, deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired()
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=core._worker, args=(sender, config, job))
    process.start()
    sender.close()
    snapshot = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExpired()
            per_step = core.worker_timeout(config, snapshot is None)
            if not receiver.poll(min(per_step, remaining)):
                if remaining <= per_step or time.monotonic() >= deadline:
                    raise BudgetExpired()
                if snapshot is None or snapshot["reason"]:
                    raise RuntimeError("Worker startup/finalization timeout")
                return core.timeout_record(snapshot, per_step)
            try:
                kind, payload = receiver.recv()
            except EOFError as exc:
                raise RuntimeError("Worker crashed; no unfinished color pair saved") from exc
            if kind == "error":
                raise RuntimeError(payload)
            if kind == "done":
                return payload
            snapshot = payload
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()


def collect_new_pair(config, jobs, fingerprint, deadline):
    records = []
    for job in jobs:
        started = core.utc_now()
        record = run_game(config, job, deadline)
        record.update(id=job["id"], run_id=fingerprint, started_at=started, ended_at=core.utc_now(),
                      mode="reduced_time_budget", pair=job["pair"], opening_index=job["opening_index"],
                      leg=job["leg"], seeds=job["seeds"], max_plies=config["max_plies"],
                      configs={side: dict(label=job[side]["id"], **job[side]) for side in ("white", "black")})
        core.validate_records([record], config, fingerprint)
        records.append(record)
        print(f"  leg {job['leg'] + 1}/2: {job['id']} -> {record['result']} ({record['reason']})", flush=True)
    return records


def report(output, frozen, store):
    records = store.list_games()
    balanced(records, frozen["config"], frozen["fingerprint"])
    summary = core.export_report(output, frozen, store)
    counts = {}
    for r in records:
        counts[r["pair"]] = counts.get(r["pair"], 0) + 1
    pairs = len(frozen["config"]["agents"]) * (len(frozen["config"]["agents"]) - 1) // 2
    summary.update(pairs_with_results=len(counts), expected_pairs=pairs,
                   min_games_per_covered_pair=min(counts.values(), default=0),
                   reused_games=sum("origin" in r for r in records), color_balanced=True,
                   acceptance="COMPLETE_REDUCED" if summary["complete"] else "PARTIAL_REDUCED_NOT_COMPLETE")
    core.atomic_json(output / "progress.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report", "verify"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("experiment_with_checkpoints.json")))
    parser.add_argument("--games-per-pair", type=int, default=10)
    parser.add_argument("--hours", type=float, default=7)
    parser.add_argument("--move-time-limit", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--output", default="results/reduced_8h")
    parser.add_argument("--reuse", nargs="*", default=None)
    args = parser.parse_args(argv)
    output = core.project_path(args.output)
    if args.command in ("report", "verify"):
        frozen = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        with core.run_lock(output):
            store = core.HistoryStore(output / "history.sqlite3")
            summary = report(output, frozen, store)
            print(json.dumps(summary, indent=2))
            if args.command == "verify" and not (summary["complete"] and summary["clean"]):
                raise ValueError("Reduced run incomplete or contains errors; see coverage in progress.json")
        return 0
    if not 0 < args.hours <= 24:
        raise ValueError("hours must be in (0,24]")
    deadline = time.monotonic() + args.hours * 3600
    config = config_for_budget(args.config, args.games_per_pair)
    if args.move_time_limit is not None:
        from agents.search_limits import validate_seconds
        validate_seconds(args.move_time_limit)
        config["move_time_limit_seconds"] = args.move_time_limit
    frozen, missing = frozen_manifest(config)
    reusable = []
    sources = args.reuse if args.reuse is not None else ([] if config.get("move_time_limit_seconds") else ["results/classic_200"])
    for path in sources:
        reusable.extend(choose_reuse(core.project_path(path), frozen))
    if len({r["id"] for r in reusable}) != len(reusable):
        raise ValueError("Reuse sources overlap; choose nonoverlapping sources")
    jobs = core.schedule(config)
    print(f"Reduced target: {len(config['agents'])} bots, {len(jobs)} games, {args.games_per_pair} games/pair; reuse eligible: {len(reusable)}")
    print(f"Output: {output}; deadline: {args.hours} hours from this invocation; no guarantee of full coverage")
    if missing:
        raise ValueError("Missing checkpoints:\n" + "\n".join(missing))
    if args.command == "plan":
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with core.run_lock(output):
        core.bind_manifest(output, frozen)
        store = core.HistoryStore(output / "history.sqlite3")
        existing = store.list_games()
        balanced(existing, config, frozen["fingerprint"])
        seen = {r["id"] for r in existing}
        for i in range(0, len(reusable), 2):
            pair = reusable[i:i+2]
            if pair[0]["id"] not in seen:
                save_pair(store, pair)
                seen.update(r["id"] for r in pair)
        groups = [jobs[i:i+2] for i in range(0, len(jobs), 2)]
        # Each opening round covers all pairs before moving to the next opening.
        groups.sort(key=lambda group: (group[0]["opening_index"],
                    0 if (group[0]["white"]["key"] in ("supervised", "rl_policy", "alphago", "alphazero")) !=
                         (group[0]["black"]["key"] in ("supervised", "rl_policy", "alphago", "alphazero")) else 1,
                    group[0]["pair"]))
        try:
            for pair in groups:
                if pair[0]["id"] in seen:
                    continue
                print(f"[{len(seen)}/{len(jobs)}] next color pair: {pair[0]['pair']}, opening {pair[0]['opening_index']}", flush=True)
                records = collect_new_pair(config, pair, frozen["fingerprint"], deadline)
                save_pair(store, records)
                seen.update(r["id"] for r in records)
                with (output / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"{core.utc_now()} saved {len(seen)}/{len(jobs)}; {pair[0]['pair']}\n")
                core.atomic_json(output / "progress.json", dict(completed_games=len(seen), expected_games=len(jobs),
                                  reused_games=sum(r["id"] in seen for r in reusable), updated_at=core.utc_now(),
                                  mode="reduced_time_budget", color_balanced=True))
        except BudgetExpired:
            print("TIME BUDGET ENDED. Unfinished color pair excluded, not counted as a loss.", flush=True)
        finally:
            summary = report(output, frozen, store)
            print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped. Completed color pairs saved. Rerun with same settings to resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
