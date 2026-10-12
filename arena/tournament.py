"""Round-robin logistic Elo; executable directly or with python -m."""
import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import AGENT_REGISTRY, build_agent
from agents.stage2_tree_search import MinimaxAgent
from arena.evaluator import play_game

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k=32):
    if not 0 <= score_a <= 1 or k <= 0:
        raise ValueError("Invalid Elo parameters")
    delta = k * (score_a - expected_score(rating_a, rating_b))
    return rating_a + delta, rating_b - delta

def build_default_tournament(depth=2, seed=42):
    return [build_agent(key, depth, seed+i) for i, key in enumerate(AGENT_REGISTRY)]

def run_tournament(agents, *, games=2, max_plies=300, output_dir="results", k=32, metadata=None):
    if len(agents) < 2 or games < 2 or games % 2 or max_plies < 1 or k <= 0:
        raise ValueError("Use >=2 agents, an even games >=2, max_plies >=1, K >0")
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError("Agent names must be unique")
    stats = {name: dict(agent=name, elo=1200.0, games=0, wins=0, draws=0,
                        losses=0, seconds=0.0, calls=0) for name in names}
    records = []
    for a, b in combinations(agents, 2):
        for i in range(games):
            white, black = (a, b) if i % 2 == 0 else (b, a)
            record = play_game(white, black, max_plies=max_plies)
            records.append(record.to_dict())
            score = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}[record.result]
            ws, bs = stats[white.name], stats[black.name]
            ws["elo"], bs["elo"] = update_elo(ws["elo"], bs["elo"], score, k)
            for row, value, seconds, calls in (
                (ws, score, record.white_seconds, record.white_calls),
                (bs, 1-score, record.black_seconds, record.black_calls)):
                row["games"] += 1
                row["wins" if value == 1 else "losses" if value == 0 else "draws"] += 1
                row["seconds"] += seconds
                row["calls"] += calls
    rows = []
    for row in stats.values():
        rows.append({key: row[key] for key in ("agent", "elo", "games", "wins", "draws", "losses")} |
                    {"win_rate": row["wins"]/row["games"], "draw_rate": row["draws"]/row["games"],
                     "score_rate": (row["wins"]+row["draws"]/2)/row["games"],
                     "ms_per_move": row["seconds"]*1000/row["calls"] if row["calls"] else 0})
    rows.sort(key=lambda row: (-row["elo"], row["agent"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"leaderboard.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Local Elo leaderboard", "",
             f"Games/pair: {games}; max plies: {max_plies}; initial Elo: 1200; logistic K: {k}.",
             "Ply-limit draws are adjudications. These are local estimates, not FIDE ratings.", "",
             "| Agent | Elo | W / D / L | Win % | Draw % | ms/move |",
             "| :--- | ---: | :---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f'| {row["agent"]} | {row["elo"]:.1f} | {row["wins"]} / {row["draws"]} / {row["losses"]} | {row["win_rate"]:.1%} | {row["draw_rate"]:.1%} | {row["ms_per_move"]:.2f} |')
    (output/"leaderboard.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    payload = {"config": {"games_per_pair": games, "max_plies": max_plies, "initial_elo": 1200,
                           "k": k, "claim_draw": True, **(metadata or {})},
               "leaderboard": rows, "records": records}
    (output/"tournament_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

def build_cli_agents(specs, depth, seed):
    """KEY or KEY:DEPTH; explicit depth only applies to search agents."""
    agents = []
    config = []
    if depth < 1:
        raise ValueError("--depth must be >=1")
    for i, spec in enumerate(specs):
        parts = spec.split(":")
        key = parts[0]
        if key not in AGENT_REGISTRY or len(parts) > 2:
            raise ValueError(f"Unknown agent specification: {spec}")
        local_depth = depth
        if len(parts) == 2:
            if not issubclass(AGENT_REGISTRY[key], MinimaxAgent):
                raise ValueError(f"{key} does not use search depth")
            try:
                local_depth = int(parts[1])
            except ValueError:
                raise ValueError(f"Depth must be a positive integer: {spec}") from None
            if local_depth < 1:
                raise ValueError(f"Depth must be >=1: {spec}")
        agent = build_agent(key, local_depth, seed+i)
        agents.append(agent)
        config.append({"key": key, "name": agent.name,
                       "depth": local_depth if isinstance(agent, MinimaxAgent) else None,
                       "seed": seed+i if not isinstance(agent, MinimaxAgent) else None})
    return agents, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2, help="Even games per pair, alternating colors")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", nargs="+", default=list(AGENT_REGISTRY),
                        metavar="KEY[:DEPTH]", help="Example: alphabeta:2 alphabeta:4")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)
    try:
        if args.games < 2 or args.games % 2:
            raise ValueError("--games must be an even integer >=2")
        if args.max_plies < 1:
            raise ValueError("--max-plies must be >=1")
        agents, config = build_cli_agents(args.agents, args.depth, args.seed)
        if len(agents) < 2 or len({agent.name for agent in agents}) != len(agents):
            raise ValueError("Use >=2 unique agents; compare depths with alphabeta:2 alphabeta:4")
    except ValueError as exc:
        parser.error(str(exc))
    result = run_tournament(agents, games=args.games, max_plies=args.max_plies,
                            output_dir=args.output_dir, metadata={"seed": args.seed, "depth": args.depth,
                                                                 "agents": config})
    for row in result["leaderboard"]:
        print(f'{row["agent"]:24} Elo {row["elo"]:.1f}  W/D/L {row["wins"]}/{row["draws"]}/{row["losses"]}')
    return result

if __name__ == "__main__":
    main()
