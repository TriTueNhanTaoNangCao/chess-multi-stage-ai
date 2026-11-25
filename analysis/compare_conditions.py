"""So sánh nhiều điều kiện thí nghiệm (Ablation, Seed, Budget, Openings) - Task T5."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.arena_stats import (
    bootstrap_pair_ci,
    compute_wdl,
    extract_agent_telemetry,
    wilson_interval,
)
from services.history import HistoryStore


def parse_cond(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Tham số --cond phải có dạng nhãn=thư_mục, nhận được: {spec!r}")
    label, path_str = spec.split("=", 1)
    label = label.strip()
    if not label or not path_str.strip():
        raise ValueError(f"Nhãn hoặc đường dẫn rỗng trong --cond: {spec!r}")
    return label, Path(path_str.strip())


def evaluate_condition(
    label: str,
    folder: Path,
    *,
    n_boot: int,
    bootstrap_seed: int,
    first_openings: int | None = None,
) -> dict[str, Any]:
    manifest_path = folder / "manifest.json"
    db_path = folder / "history.sqlite3"
    if not manifest_path.is_file() or not db_path.is_file():
        raise FileNotFoundError(f"Thiếu manifest.json hoặc history.sqlite3 trong {folder}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = manifest["config"]["seed"]
    games = HistoryStore(db_path).list_games()
    if first_openings is not None:
        if first_openings < 1:
            raise ValueError("--first-openings phải >= 1")
        games = [g for g in games if int(g["opening_index"]) < first_openings]

    if not games:
        raise ValueError(f"Không có ván đấu nào trong {folder} (first_openings={first_openings})")

    pairs = {g.get("pair") or f"{g['white']}__{g['black']}" for g in games}
    if len(pairs) != 1:
        raise ValueError(
            f"Điều kiện {label!r} ({folder}) phải chứa đúng 1 cặp, tìm thấy {len(pairs)}: {sorted(pairs)}"
        )
    pair_id = next(iter(pairs))
    agent_a, agent_b = pair_id.split("__", 1)

    wdl = compute_wdl(games, agent_a)
    w_lo, w_hi = wilson_interval(wdl["W"] + 0.5 * wdl["D"], wdl["N"])
    b_lo, b_hi = bootstrap_pair_ci(games, agent_a, n_boot=n_boot, seed=bootstrap_seed)
    tel_a = extract_agent_telemetry(games, agent_a)
    tel_b = extract_agent_telemetry(games, agent_b)

    return {
        "label": label,
        "pair": pair_id,
        "seed": seed,
        "N": wdl["N"],
        "W": wdl["W"],
        "D": wdl["D"],
        "L": wdl["L"],
        "score": round(wdl["score"], 6),
        "wilson_low": round(w_lo, 6),
        "wilson_high": round(w_hi, 6),
        "bootstrap_low": round(b_lo, 6),
        "bootstrap_high": round(b_hi, 6),
        "a_ms_mean": tel_a["ms_mean"],
        "b_ms_mean": tel_b["ms_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh nhiều điều kiện thí nghiệm.")
    parser.add_argument("--cond", action="append", required=True, help="Cặp nhãn=thư_mục")
    parser.add_argument("--output", required=True, help="Thư mục xuất conditions.csv và conditions.md")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261124)
    parser.add_argument(
        "--first-openings",
        type=int,
        default=None,
        help="Chỉ tính N thế mở đầu đầu tiên của mỗi thư mục",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for raw in args.cond:
        label, folder = parse_cond(raw)
        rows.append(
            evaluate_condition(
                label,
                folder,
                n_boot=args.bootstrap,
                bootstrap_seed=args.bootstrap_seed,
                first_openings=args.first_openings,
            )
        )

    csv_path = out_dir / "conditions.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})

    md_lines = [
        "# So sánh các điều kiện thực nghiệm",
        "",
        f"- Bootstrap: `{args.bootstrap}` lần (seed `{args.bootstrap_seed}`)"
        + (f"\n- Giới hạn: `{args.first_openings}` thế mở đầu đầu tiên" if args.first_openings else ""),
        "",
        "| Nhãn | Cặp | Seed | N | W-D-L | Score | Wilson 95% | Bootstrap 95% | ms TB (A) | ms TB (B) |",
        "| :--- | :--- | ---: | ---: | :---: | ---: | :---: | :---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['label']}` | `{r['pair']}` | {r['seed']} | {r['N']} | "
            f"{r['W']}-{r['D']}-{r['L']} | {r['score']:.4f} | "
            f"[{r['wilson_low']:.4f}, {r['wilson_high']:.4f}] | "
            f"[{r['bootstrap_low']:.4f}, {r['bootstrap_high']:.4f}] | "
            f"{r['a_ms_mean'] if r['a_ms_mean'] is not None else '-'} | "
            f"{r['b_ms_mean'] if r['b_ms_mean'] is not None else '-'} |"
        )
    (out_dir / "conditions.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(rows)} điều kiện vào {out_dir}")


if __name__ == "__main__":
    main()
