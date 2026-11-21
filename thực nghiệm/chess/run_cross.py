"""Deadline-limited classic/cross matches, reusing compatible completed games."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_budget as budget

core = budget.core
NEURAL = {"supervised", "rl_policy", "alphago", "alphazero"}
CLASSIC_CROSS = "classic_internal_and_cross"


def cross_jobs(config):
    if config.get("pair_scope") == CLASSIC_CROSS:
        return [job for job in core.schedule(config)
                if not (job["white"]["key"] in NEURAL and job["black"]["key"] in NEURAL)]
    return [job for job in core.schedule(config)
            if (job["white"]["key"] in NEURAL) != (job["black"]["key"] in NEURAL)]


def manifest(config):
    config.setdefault("pair_scope", "classic_vs_neural_only")
    frozen, missing = budget.frozen_manifest(config)
    frozen.pop("fingerprint")
    frozen["cross_runner_sha256"] = core.file_hash(Path(__file__))
    frozen["fingerprint"] = core.digest(frozen)
    return frozen, missing


def validate(records, frozen, complete=False):
    config = frozen["config"]
    jobs = cross_jobs(config)
    allowed = {job["id"] for job in jobs}
    if any(r["id"] not in allowed for r in records):
        raise ValueError("Non-cross/out-of-scope game found in selected output")
    budget.balanced(records, config, frozen["fingerprint"])
    if complete and len(records) != len(jobs):
        raise ValueError(f"Incomplete selected benchmark: {len(records)}/{len(jobs)} games")


def reuse_records(sources, frozen, jobs):
    """Merge overlapping local sources without double-counting or accepting conflicts."""
    allowed = {j["id"] for j in jobs}
    indexed = {}
    for path in sources:
        for record in budget.choose_reuse(core.project_path(path), frozen):
            if record["id"] not in allowed:
                continue
            previous = indexed.get(record["id"])
            if previous is not None:
                def comparable(r):
                    return {k: v for k, v in r.items() if k not in ("run_id", "origin")}
                if comparable(previous) != comparable(record):
                    raise ValueError(f"Conflicting reuse records: {record['id']}")
            else:
                indexed[record["id"]] = record
    return [indexed[j["id"]] for j in jobs if j["id"] in indexed]


def report(output, frozen, store):
    config = frozen["config"]
    jobs = cross_jobs(config)
    indexed = {r["id"]: r for r in store.list_games()}
    records = [indexed[j["id"]] for j in jobs if j["id"] in indexed]
    validate(list(indexed.values()), frozen)
    names = [a["id"] for a in config["agents"]]
    rows = core.statistics(records, names)
    pairs = list(dict.fromkeys(j["pair"] for j in jobs))
    matchups = []
    for pair in pairs:
        participants = pair.split("__")
        subset = [r for r in records if r["pair"] == pair]
        matchups.extend(dict(pair=pair, **row) for row in core.statistics(subset, participants))
    counts = Counter(r["pair"] for r in records)
    errors = sum(bool(r.get("error")) for r in records)
    classic_games = sum(r["white"] not in NEURAL and r["black"] not in NEURAL for r in records)
    summary = dict(mode=config["mode"], pair_scope=config["pair_scope"],
                   completed_games=len(records), expected_games=len(jobs),
                   classic_internal_games=classic_games, cross_games=len(records) - classic_games,
                   pairs_with_results=len(counts), expected_pairs=len(pairs),
                   min_games_per_covered_pair=min(counts.values(), default=0),
                   complete=len(records) == len(jobs), clean=errors == 0, color_balanced=True,
                   reused_games=sum("origin" in r for r in records),
                   error_games=errors, timeout_games=sum(bool(r.get("timeout")) for r in records),
                   fingerprint=frozen["fingerprint"], percent_unit="0..100",
                   checkpoint_sha256=frozen["checkpoint_sha256"], paired_games_are_not_independent=True,
                   acceptance="COMPLETE_REDUCED_SELECTED" if len(records) == len(jobs) else "PARTIAL_REDUCED_SELECTED_NOT_COMPLETE")
    core.atomic_json(output / "progress.json", summary)
    core.atomic_json(output / "tournament_results.json", dict(config=config, summary=summary,
                     leaderboard=rows, matchups=matchups, records=records))
    core.write_csv(output / "leaderboard.csv", rows)
    core.write_csv(output / "matchups.csv", matchups)
    combined = config["pair_scope"] == CLASSIC_CROSS
    title = "Classic internal + classic vs neural" if combined else "Classic vs neural only"
    lines = [f"# {title}: reduced benchmark", "",
             f"Completed {len(records)}/{len(jobs)} games; covered {len(counts)}/{len(pairs)} pairs.",
             f"Classic internal: {classic_games}; cross-group: {len(records) - classic_games}; reused: {summary['reused_games']}.",
             "Neural-vs-neural games excluded. Elo belongs only to this selected schedule.",
             "Percentages use 0..100; small sample, paired games are not independent.", "",
             "| Agent | W / D / L | Win % | Score % | White win % | Black win % | ms/move |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['agent']} | {r['wins']}/{r['draws']}/{r['losses']} | {r['win_pct']} | {r['score_pct']} | {r['white_win_pct']} | {r['black_win_pct']} | {r['ms_per_move']} |")
    (output / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp = output / "games.pgn.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(core.to_pgn(r) + "\n\n")
    temp.replace(output / "games.pgn")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report", "verify"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("experiment_with_checkpoints.json")))
    parser.add_argument("--games-per-pair", type=int, default=10)
    parser.add_argument("--hours", type=float, default=9)
    parser.add_argument("--scope", choices=("cross", "classic-cross"), default="classic-cross")
    parser.add_argument("--output", default="results/remaining_classic_cross_9h_10")
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
                raise ValueError("Selected benchmark incomplete or contains errors; inspect progress.json")
        return 0
    if not 0 < args.hours <= 24:
        raise ValueError("hours must be in (0,24]")
    deadline = time.monotonic() + args.hours * 3600
    config = budget.config_for_budget(args.config, args.games_per_pair)
    config["pair_scope"] = CLASSIC_CROSS if args.scope == "classic-cross" else "classic_vs_neural_only"
    config["mode"] = "reduced_classic_and_cross" if args.scope == "classic-cross" else "reduced_cross_only"
    jobs = cross_jobs(config)
    if not jobs:
        raise ValueError("Need at least one classic and one neural agent")
    frozen, missing = manifest(config)
    if missing:
        raise ValueError("Missing checkpoints:\n" + "\n".join(missing))
    sources = args.reuse
    if sources is None:
        sources = [str(p) for p in (core.project_path("results/classic_200"), core.project_path("results/reduced_9h_10"))
                   if (p / "manifest.json").exists()]
    reusable = reuse_records(sources, frozen, jobs)
    internal = sum(j["white"]["key"] not in NEURAL and j["black"]["key"] not in NEURAL for j in jobs)
    print(f"Scope={args.scope}: {len(set(j['pair'] for j in jobs))} pairs, {len(jobs)} target games, {args.games_per_pair // 2} per color")
    print(f"Classic internal target={internal}; cross target={len(jobs) - internal}; reusable={len(reusable)}; new games needed={len(jobs) - len(reusable)}")
    print(f"Output: {output}; deadline: {args.hours} hours. No guarantee of finishing all games.")
    if args.command == "plan":
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with core.run_lock(output):
        core.bind_manifest(output, frozen)
        store = core.HistoryStore(output / "history.sqlite3")
        existing = store.list_games()
        validate(existing, frozen)
        seen = {r["id"] for r in existing}
        for index in range(0, len(reusable), 2):
            pair = reusable[index:index+2]
            if pair[0]["id"] not in seen:
                budget.save_pair(store, pair)
                seen.update(r["id"] for r in pair)
        groups = [jobs[i:i+2] for i in range(0, len(jobs), 2)]
        # Cover all pairs before adding another opening; finish cheap classic legs first.
        groups.sort(key=lambda pair: (pair[0]["opening_index"],
                    pair[0]["white"]["key"] in NEURAL or pair[0]["black"]["key"] in NEURAL,
                    pair[0]["pair"]))
        try:
            for pair in groups:
                if pair[0]["id"] in seen:
                    continue
                print(f"[{len(seen)}/{len(jobs)}] {pair[0]['pair']}, opening {pair[0]['opening_index']}", flush=True)
                records = budget.collect_new_pair(config, pair, frozen["fingerprint"], deadline)
                budget.save_pair(store, records)
                seen.update(r["id"] for r in records)
                with (output / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"{core.utc_now()} saved {len(seen)}/{len(jobs)} {pair[0]['pair']}\n")
                core.atomic_json(output / "progress.json", dict(completed_games=len(seen), expected_games=len(jobs),
                                 pair_scope=config["pair_scope"], color_balanced=True, updated_at=core.utc_now()))
        except budget.BudgetExpired:
            print("TIME BUDGET ENDED. Incomplete color pair excluded, not counted as a loss.")
        finally:
            print(json.dumps(report(output, frozen, store), indent=2))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped. Completed color pairs saved; rerun same command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
