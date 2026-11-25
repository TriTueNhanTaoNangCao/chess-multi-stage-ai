"""Phân tích thống kê kết quả đối kháng Arena (Task T4).

Tính tỷ lệ thắng/hòa/thua, score, khoảng tin cậy Wilson 95%, Bootstrap theo cặp
thế mở đầu, phân tích Trắng/Đen, phân bố lý do kết thúc và telemetry từng agent.
Không tính Elo.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.history import HistoryStore


WILSON_Z_95 = 1.959964


def wilson_interval(successes: float, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Tính khoảng tin cậy Wilson cho tỷ lệ điểm successes / n, kẹp trong [0, 1]."""
    if n <= 0:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, successes / n))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(max(0.0, p * (1.0 - p) / n + z2 / (4.0 * n * n)))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def game_points_for(record: dict[str, Any], agent_id: str) -> float:
    """Điểm của agent_id trong một ván (1.0 thắng, 0.5 hòa, 0.0 thua)."""
    res = record["result"]
    if res == "1/2-1/2":
        return 0.5
    if record["white"] == agent_id:
        return 1.0 if res == "1-0" else 0.0
    if record["black"] == agent_id:
        return 1.0 if res == "0-1" else 0.0
    raise ValueError(f"Agent {agent_id!r} không tham gia ván {record.get('id')}")


def bootstrap_pair_ci(
    records: list[dict[str, Any]],
    agent_a: str,
    n_boot: int = 10000,
    seed: int = 20261124,
) -> tuple[float, float]:
    """Bootstrap có hoàn lại theo đơn vị opening_index (gồm cả leg 0 và leg 1)."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        grouped[int(rec["opening_index"])].append(game_points_for(rec, agent_a))
    if not grouped:
        return (0.0, 0.0)

    units = np.array(
        [[sum(pts), float(len(pts))] for _, pts in sorted(grouped.items())],
        dtype=np.float64,
    )
    k = units.shape[0]
    raw_score = float(units[:, 0].sum() / units[:, 1].sum())
    if n_boot <= 0 or k == 0:
        return (raw_score, raw_score)

    rng = np.random.default_rng(seed)
    pts_col = units[:, 0]
    cnt_col = units[:, 1]
    chunk_size = max(1, min(n_boot, 2_000_000 // max(1, k)))
    boot_scores = np.empty(n_boot, dtype=np.float64)
    pos = 0
    while pos < n_boot:
        take = min(chunk_size, n_boot - pos)
        idx = rng.integers(0, k, size=(take, k))
        boot_scores[pos : pos + take] = pts_col[idx].sum(axis=1) / cnt_col[idx].sum(axis=1)
        pos += take

    low, high = np.percentile(boot_scores, [2.5, 97.5])
    return (max(0.0, float(low)), min(1.0, float(high)))


def _agent_checkpoint_hashes(manifest: dict[str, Any], agent_id: str) -> dict[str, str]:
    prefix = f"{agent_id}."
    return {
        k: v
        for k, v in manifest.get("checkpoint_sha256", {}).items()
        if k.startswith(prefix)
    }


def load_validated_inputs(
    input_dirs: list[Path],
    *,
    allow_mixed: bool = False,
    first_openings: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Đọc và kiểm tra tính nhất quán giữa nhiều thư mục kết quả."""
    if first_openings is not None and first_openings < 1:
        raise ValueError("--first-openings phải >= 1")

    manifests: list[dict[str, Any]] = []
    all_games: list[dict[str, Any]] = []
    agent_order: list[str] = []
    seen_agents: dict[str, dict[str, Any]] = {}
    seen_agent_hashes: dict[str, dict[str, str]] = {}
    seen_pairs: dict[frozenset[str], Path] = {}
    baseline_rule: tuple[Any, ...] | None = None

    for raw_dir in input_dirs:
        folder = Path(raw_dir)
        manifest_path = folder / "manifest.json"
        db_path = folder / "history.sqlite3"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Thiếu manifest.json trong {folder}")
        if not db_path.is_file():
            raise FileNotFoundError(f"Thiếu history.sqlite3 trong {folder}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = manifest["config"]
        manifests.append({"dir": str(folder), "manifest": manifest})

        rule = (cfg["max_plies"], cfg["claim_draw"], cfg["seed"], cfg["games_per_pair"])
        if not allow_mixed:
            if baseline_rule is None:
                baseline_rule = rule
            elif rule != baseline_rule:
                raise ValueError(
                    f"Cấu hình không khớp giữa các thư mục tại {folder}: "
                    f"(max_plies, claim_draw, seed, games_per_pair)={rule} != {baseline_rule}. "
                    f"Dùng --allow-mixed để bỏ qua kiểm tra này."
                )

        for agent in cfg["agents"]:
            aid = agent["id"]
            hashes = _agent_checkpoint_hashes(manifest, aid)
            if aid not in seen_agents:
                seen_agents[aid] = agent
                seen_agent_hashes[aid] = hashes
                agent_order.append(aid)
            elif not allow_mixed:
                if agent != seen_agents[aid] or hashes != seen_agent_hashes[aid]:
                    raise ValueError(
                        f"Agent {aid!r} có cấu hình hoặc checkpoint SHA256 khác nhau ở {folder}. "
                        f"Dùng --allow-mixed để bỏ qua kiểm tra này."
                    )

        games = HistoryStore(db_path).list_games()
        if first_openings is not None:
            games = [g for g in games if int(g["opening_index"]) < first_openings]

        dir_pairs = {frozenset((g["white"], g["black"])) for g in games}
        for pkey in dir_pairs:
            if pkey in seen_pairs:
                raise ValueError(
                    f"Cặp {sorted(pkey)} xuất hiện ở cả {seen_pairs[pkey]} và {folder}"
                )
            seen_pairs[pkey] = folder

        all_games.extend(games)

    if not all_games:
        raise ValueError("Không có ván đấu nào để thống kê.")

    return all_games, manifests, agent_order


def extract_agent_telemetry(records: list[dict[str, Any]], agent_id: str) -> dict[str, Any]:
    """Trích xuất chỉ số latency, nodes_visited, cutoffs, simulation_budget của agent_id."""
    ms_list: list[float] = []
    nodes_list: list[float] = []
    cutoffs_list: list[float] = []
    budgets: list[int] = []

    for rec in records:
        if rec["white"] == agent_id:
            target_color = "white"
        elif rec["black"] == agent_id:
            target_color = "black"
        else:
            continue

        for entry in rec.get("telemetry", []):
            if entry.get("color") != target_color or entry.get("error"):
                continue
            ms_list.append(float(entry["seconds"]) * 1000.0)
            if "nodes_visited" in entry and entry["nodes_visited"] is not None:
                nodes_list.append(float(entry["nodes_visited"]))
            if "cutoffs" in entry and entry["cutoffs"] is not None:
                cutoffs_list.append(float(entry["cutoffs"]))
            if "simulation_budget" in entry and entry["simulation_budget"] is not None:
                budgets.append(int(entry["simulation_budget"]))

    if ms_list:
        arr = np.array(ms_list, dtype=np.float64)
        ms_mean = round(float(arr.mean()), 3)
        ms_median = round(float(np.median(arr)), 3)
        ms_p95 = round(float(np.percentile(arr, 95, method="linear")), 3)
    else:
        ms_mean = ms_median = ms_p95 = None

    nodes_mean = round(float(np.mean(nodes_list)), 3) if nodes_list else None
    cutoffs_mean = round(float(np.mean(cutoffs_list)), 3) if cutoffs_list else None
    budget_mode = Counter(budgets).most_common(1)[0][0] if budgets else None

    return {
        "moves": len(ms_list),
        "ms_mean": ms_mean,
        "ms_median": ms_median,
        "ms_p95": ms_p95,
        "nodes_visited_mean": nodes_mean,
        "cutoffs_mean": cutoffs_mean,
        "simulation_budget": budget_mode,
    }


def compute_wdl(records: list[dict[str, Any]], agent_id: str) -> dict[str, Any]:
    """Tính W/D/L, score, tỷ lệ theo màu Trắng/Đen của agent_id trên tập records."""
    n = len(records)
    wins = draws = losses = 0
    w_games = w_wins = w_draws = w_losses = 0
    b_games = b_wins = b_draws = b_losses = 0

    for rec in records:
        pts = game_points_for(rec, agent_id)
        is_white = rec["white"] == agent_id
        if pts == 1.0:
            wins += 1
            if is_white:
                w_wins += 1
            else:
                b_wins += 1
        elif pts == 0.5:
            draws += 1
            if is_white:
                w_draws += 1
            else:
                b_draws += 1
        else:
            losses += 1
            if is_white:
                w_losses += 1
            else:
                b_losses += 1
        if is_white:
            w_games += 1
        else:
            b_games += 1

    successes = wins + 0.5 * draws
    score = successes / n if n else 0.0
    w_score = (w_wins + 0.5 * w_draws) / w_games if w_games else None
    b_score = (b_wins + 0.5 * b_draws) / b_games if b_games else None
    return {
        "N": n,
        "W": wins,
        "D": draws,
        "L": losses,
        "score": score,
        "white_games": w_games,
        "white_W": w_wins,
        "white_D": w_draws,
        "white_L": w_losses,
        "white_score": w_score,
        "black_games": b_games,
        "black_W": b_wins,
        "black_D": b_draws,
        "black_L": b_losses,
        "black_score": b_score,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def analyze(
    input_dirs: list[Path],
    output_dir: Path,
    *,
    n_boot: int = 10000,
    bootstrap_seed: int = 20261124,
    allow_mixed: bool = False,
) -> dict[str, Any]:
    all_games, manifests, agent_order = load_validated_inputs(
        input_dirs, allow_mixed=allow_mixed
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nhóm ván theo pair (a__b)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_games:
        pair_id = rec.get("pair") or f"{rec['white']}__{rec['black']}"
        by_pair[pair_id].append(rec)

    pair_rows: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    all_reasons = Counter()

    for pair_id in sorted(by_pair):
        recs = by_pair[pair_id]
        agent_a, agent_b = pair_id.split("__", 1)
        wdl = compute_wdl(recs, agent_a)
        w_lo, w_hi = wilson_interval(wdl["W"] + 0.5 * wdl["D"], wdl["N"])
        b_lo, b_hi = bootstrap_pair_ci(recs, agent_a, n_boot=n_boot, seed=bootstrap_seed)

        tel_a = extract_agent_telemetry(recs, agent_a)
        tel_b = extract_agent_telemetry(recs, agent_b)
        reasons = Counter(rec.get("reason") or "unknown" for rec in recs)
        all_reasons.update(reasons)
        mean_plies = round(sum(len(r["moves"]) for r in recs) / len(recs), 2)
        timeouts = sum(bool(r.get("timeout")) for r in recs)
        errors = sum(bool(r.get("error")) for r in recs)

        row = {
            "pair": pair_id,
            "agent_a": agent_a,
            "agent_b": agent_b,
            "N": wdl["N"],
            "W": wdl["W"],
            "D": wdl["D"],
            "L": wdl["L"],
            "score": round(wdl["score"], 6),
            "wilson_low": round(w_lo, 6),
            "wilson_high": round(w_hi, 6),
            "bootstrap_low": round(b_lo, 6),
            "bootstrap_high": round(b_hi, 6),
            "white_score": round(wdl["white_score"], 6) if wdl["white_score"] is not None else None,
            "black_score": round(wdl["black_score"], 6) if wdl["black_score"] is not None else None,
            "mean_plies": mean_plies,
            "a_ms_mean": tel_a["ms_mean"],
            "a_ms_p95": tel_a["ms_p95"],
            "b_ms_mean": tel_b["ms_mean"],
            "b_ms_p95": tel_b["ms_p95"],
            "timeout_games": timeouts,
            "error_games": errors,
            "reasons": "; ".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
        }
        pair_rows.append(row)
        pair_lookup[(agent_a, agent_b)] = {
            "score": wdl["score"],
            "w_lo": w_lo,
            "w_hi": w_hi,
        }
        pair_lookup[(agent_b, agent_a)] = {
            "score": 1.0 - wdl["score"],
            "w_lo": 1.0 - w_hi,
            "w_hi": 1.0 - w_lo,
        }

    # Tổng hợp theo từng agent
    involved_agents = [
        aid for aid in agent_order if any(r["white"] == aid or r["black"] == aid for r in all_games)
    ]
    for r in all_games:
        for s in (r["white"], r["black"]):
            if s not in involved_agents:
                involved_agents.append(s)

    agent_rows: list[dict[str, Any]] = []
    for aid in involved_agents:
        recs = [r for r in all_games if r["white"] == aid or r["black"] == aid]
        wdl = compute_wdl(recs, aid)
        w_lo, w_hi = wilson_interval(wdl["W"] + 0.5 * wdl["D"], wdl["N"])
        tel = extract_agent_telemetry(recs, aid)
        agent_rows.append(
            {
                "agent": aid,
                "N": wdl["N"],
                "W": wdl["W"],
                "D": wdl["D"],
                "L": wdl["L"],
                "score": round(wdl["score"], 6),
                "wilson_low": round(w_lo, 6),
                "wilson_high": round(w_hi, 6),
                "white_score": round(wdl["white_score"], 6) if wdl["white_score"] is not None else None,
                "black_score": round(wdl["black_score"], 6) if wdl["black_score"] is not None else None,
                "mean_plies": round(sum(len(r["moves"]) for r in recs) / len(recs), 2),
                "moves": tel["moves"],
                "ms_mean": tel["ms_mean"],
                "ms_median": tel["ms_median"],
                "ms_p95": tel["ms_p95"],
                "nodes_visited_mean": tel["nodes_visited_mean"],
                "cutoffs_mean": tel["cutoffs_mean"],
                "simulation_budget": tel["simulation_budget"],
                "timeout_games": sum(bool(r.get("timeout")) for r in recs),
                "error_games": sum(bool(r.get("error")) for r in recs),
            }
        )

    agent_rows.sort(key=lambda x: (-x["score"], x["agent"]))
    write_csv(output_dir / "pairs.csv", pair_rows)
    write_csv(output_dir / "agents.csv", agent_rows)

    # Ma trận đối đầu agent x agent
    matrix_csv_rows: list[dict[str, Any]] = []
    for row_id in involved_agents:
        m_row: dict[str, Any] = {"agent": row_id}
        for col_id in involved_agents:
            if row_id == col_id:
                m_row[col_id] = "-"
            elif (row_id, col_id) in pair_lookup:
                m_row[col_id] = round(pair_lookup[(row_id, col_id)]["score"], 4)
            else:
                m_row[col_id] = ""
        matrix_csv_rows.append(m_row)
    write_csv(output_dir / "matrix_score.csv", matrix_csv_rows, fieldnames=["agent", *involved_agents])

    md_lines = [
        "# Ma trận Score đối đầu (Wilson 95% CI)",
        "",
        "Ô ở hàng `i`, cột `j` là `score = (W + 0.5*D) / N` của `i` khi đấu với `j`, kèm `[Wilson low, Wilson high]`.",
        "",
        "| Agent | " + " | ".join(involved_agents) + " |",
        "| :--- | " + " | ".join(":---:" for _ in involved_agents) + " |",
    ]
    for row_id in involved_agents:
        cells = []
        for col_id in involved_agents:
            if row_id == col_id:
                cells.append("-")
            elif (row_id, col_id) in pair_lookup:
                cell = pair_lookup[(row_id, col_id)]
                cells.append(f"{cell['score']:.4f} [{cell['w_lo']:.3f}, {cell['w_hi']:.3f}]")
            else:
                cells.append("-")
        md_lines.append(f"| `{row_id}` | " + " | ".join(cells) + " |")
    (output_dir / "matrix_score.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Summary markdown
    fingerprints = [m["manifest"].get("fingerprint", "") for m in manifests]
    summary_lines = [
        "# Tổng hợp Thống kê Arena",
        "",
        "> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.",
        "",
        f"- **Tổng số ván:** {len(all_games)}",
        f"- **Tổng số cặp:** {len(pair_rows)}",
        f"- **Bootstrap:** {n_boot} lần lấy mẫu có hoàn lại theo `opening_index` (seed `{bootstrap_seed}`)",
        "- **Thư mục nguồn & Fingerprint:**",
    ]
    for item in manifests:
        m = item["manifest"]
        cfg = m["config"]
        summary_lines.append(
            f"  - `{item['dir']}` | fingerprint: `{m.get('fingerprint')}` | "
            f"seed={cfg.get('seed')}, max_plies={cfg.get('max_plies')}, "
            f"claim_draw={cfg.get('claim_draw')}, device={cfg.get('device')}"
        )

    env0 = manifests[0]["manifest"].get("environment", {})
    summary_lines.extend(
        [
            "",
            "## Cấu hình môi trường (manifest đầu)",
            f"- Python `{env0.get('python')}`, chess `{env0.get('chess')}`, torch `{env0.get('torch')}`, numpy `{env0.get('numpy')}`",
            f"- CPU: `{env0.get('cpu')}` ({env0.get('cpu_cores_physical')} physical / {env0.get('cpu_cores_logical')} logical), RAM: `{env0.get('ram_gb')} GB`",
            f"- GPU: `{env0.get('gpu')}`, CUDA: `{env0.get('cuda')}`",
            "",
            "## Kết quả theo cặp",
            "",
            "| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |",
            "| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |",
        ]
    )
    for r in pair_rows:
        w_s = f"{r['white_score']:.4f}" if r["white_score"] is not None else "-"
        b_s = f"{r['black_score']:.4f}" if r["black_score"] is not None else "-"
        summary_lines.append(
            f"| `{r['pair']}` | {r['N']} | {r['W']}-{r['D']}-{r['L']} | {r['score']:.4f} | "
            f"[{r['wilson_low']:.4f}, {r['wilson_high']:.4f}] | "
            f"[{r['bootstrap_low']:.4f}, {r['bootstrap_high']:.4f}] | "
            f"{w_s} / {b_s} | {r['mean_plies']} |"
        )

    summary_lines.extend(
        [
            "",
            "## Hiệu năng từng Agent (Telemetry)",
            "",
            "| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for r in agent_rows:
        summary_lines.append(
            f"| `{r['agent']}` | {r['N']} | {r['score']:.4f} | "
            f"{r['ms_mean'] if r['ms_mean'] is not None else '-'} | "
            f"{r['ms_median'] if r['ms_median'] is not None else '-'} | "
            f"{r['ms_p95'] if r['ms_p95'] is not None else '-'} | "
            f"{r['nodes_visited_mean'] if r['nodes_visited_mean'] is not None else '-'} | "
            f"{r['cutoffs_mean'] if r['cutoffs_mean'] is not None else '-'} | "
            f"{r['simulation_budget'] if r['simulation_budget'] is not None else '-'} | "
            f"{r['timeout_games']} / {r['error_games']} |"
        )

    summary_lines.extend(
        [
            "",
            "## Phân bố lý do kết thúc",
            "",
            "| Lý do | Số ván | Tỷ lệ |",
            "| :--- | ---: | ---: |",
        ]
    )
    total_g = len(all_games)
    for reason, cnt in all_reasons.most_common():
        summary_lines.append(f"| `{reason}` | {cnt} | {100.0 * cnt / total_g:.2f}% |")

    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"games": len(all_games), "pairs": len(pair_rows), "fingerprints": fingerprints}


def main() -> None:
    parser = argparse.ArgumentParser(description="Thống kê kết quả Arena (Wilson, Bootstrap, p95).")
    parser.add_argument("--input", nargs="+", required=True, help="Danh sách thư mục kết quả")
    parser.add_argument("--output", required=True, help="Thư mục xuất thống kê")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Số lần lấy mẫu bootstrap")
    parser.add_argument("--bootstrap-seed", type=int, default=20261124, help="Seed cho bootstrap")
    parser.add_argument("--allow-mixed", action="store_true", help="Cho phép gộp thư mục khác tham số")
    args = parser.parse_args()

    res = analyze(
        [Path(p) for p in args.input],
        Path(args.output),
        n_boot=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        allow_mixed=args.allow_mixed,
    )
    print(
        f"Đã xuất 5 file vào {args.output}: {res['pairs']} cặp, {res['games']} ván."
    )


if __name__ == "__main__":
    main()
