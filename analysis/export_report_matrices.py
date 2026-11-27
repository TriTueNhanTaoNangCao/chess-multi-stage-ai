"""Xuất 3 nhóm ma trận bổ trợ và biểu đồ heatmap cho báo cáo từ dữ liệu thật trong repo."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "report_matrices"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SHORT_NAMES = {
    "minimax": "Minimax",
    "alphabeta": "AlphaBeta",
    "alphago": "AlphaGo",
    "rl_policy": "RL Policy",
    "supervised": "SL Policy",
    "think2": "Think2",
    "think3": "Think3",
    "alphazero_v2": "AlphaZero v2",
    "mcts_pure": "Pure MCTS",
    "think1": "Think1",
    "random": "Random",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_pair_lookup(pairs_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for r in pairs_rows:
        a, b = r["agent_a"], r["agent_b"]
        n = int(r["N"])
        w, d, l = int(r["W"]), int(r["D"]), int(r["L"])
        score = float(r["score"])
        w_lo, w_hi = float(r["wilson_low"]), float(r["wilson_high"])
        ws = float(r["white_score"])
        bs = float(r["black_score"])
        lookup[(a, b)] = {
            "N": n, "W": w, "D": d, "L": l, "score": score,
            "w_lo": w_lo, "w_hi": w_hi, "white_score": ws, "black_score": bs,
        }
        lookup[(b, a)] = {
            "N": n, "W": l, "D": d, "L": w, "score": 1.0 - score,
            "w_lo": 1.0 - w_hi, "w_hi": 1.0 - w_lo,
            "white_score": 1.0 - bs, "black_score": 1.0 - ws,
        }
    return lookup


def plot_score_heatmap(
    agents_order: list[str],
    lookup: dict[tuple[str, str], dict],
    title: str,
    out_path: Path,
    *,
    show_ci: bool = False,
) -> None:
    n = len(agents_order)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, a in enumerate(agents_order):
        for j, b in enumerate(agents_order):
            if i != j and (a, b) in lookup:
                mat[i, j] = lookup[(a, b)]["score"]

    fig_size = (11.5, 9.2) if n > 7 else (8.8, 7.0)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#E9ECEF")

    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label("Score = (W + 0,5 × D) / N", fontsize=10)

    labels = [SHORT_NAMES.get(a, a) for a in agents_order]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=9.5 if n > 7 else 10)
    ax.set_yticklabels(labels, fontsize=9.5 if n > 7 else 10)
    ax.set_xlabel("Đối thủ (cột j)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Tác tử đánh giá (hàng i)", fontsize=10.5, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)

    for i, a in enumerate(agents_order):
        for j, b in enumerate(agents_order):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#495057", fontsize=10)
                continue
            cell = lookup.get((a, b))
            if not cell:
                continue
            sc = cell["score"]
            txt_color = "white" if (sc < 0.16 or sc > 0.86) else "#111827"
            if show_ci:
                txt = f"{sc:.3f}\n[{cell['w_lo']:.2f}, {cell['w_hi']:.2f}]"
                fsize = 7.5
            else:
                txt = f"{sc:.3f}\n({cell['W']}-{cell['D']}-{cell['L']})"
                fsize = 6.8 if n > 7 else 8.2
            ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=fsize, fontweight="medium")

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_wdl_and_color_matrices(agents_7: list[str], lookup_7: dict[tuple[str, str], dict]) -> None:
    # 1. Ma trận W-D-L 7x7
    wdl_csv = []
    wdl_md = [
        "# Ma trận Thắng – Hòa – Thua (W–D–L) của 7 tác tử lõi (Chế độ A, N = 200 ván/cặp)",
        "",
        "Ô ở hàng `i`, cột `j` ghi `W–D–L` của tác tử hàng `i` khi gặp tác tử cột `j`.",
        "",
        "| Agent | " + " | ".join(SHORT_NAMES[a] for a in agents_7) + " |",
        "| :--- | " + " | ".join(":---:" for _ in agents_7) + " |",
    ]
    for a in agents_7:
        row_csv = {"agent": a}
        cells = []
        for b in agents_7:
            if a == b:
                row_csv[b] = "-"
                cells.append("-")
            else:
                c = lookup_7[(a, b)]
                val = f"{c['W']}-{c['D']}-{c['L']}"
                row_csv[b] = val
                cells.append(val)
        wdl_csv.append(row_csv)
        wdl_md.append(f"| `{a}` | " + " | ".join(cells) + " |")

    with (OUT_DIR / "matrix_7x7_wdl.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", *agents_7])
        w.writeheader()
        w.writerows(wdl_csv)
    (OUT_DIR / "matrix_7x7_wdl.md").write_text("\n".join(wdl_md) + "\n", encoding="utf-8")

    # 2. Ma trận Chênh lệch Trắng - Đen (7x7)
    color_csv = []
    color_md = [
        "# Ma trận Ảnh hưởng Thứ tự Đi trước (Trắng vs Đen) — 7 tác tử lõi (Chế độ A)",
        "",
        "Mỗi ô hàng `i`, cột `j` ghi `Trắng / Đen (Δ = Trắng − Đen)` của `i` trước `j` (mỗi màu 100 ván).",
        "",
        "| Agent | " + " | ".join(SHORT_NAMES[a] for a in agents_7) + " |",
        "| :--- | " + " | ".join(":---:" for _ in agents_7) + " |",
    ]
    n = len(agents_7)
    diff_mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, a in enumerate(agents_7):
        row_csv = {"agent": a}
        cells = []
        for j, b in enumerate(agents_7):
            if a == b:
                row_csv[b] = "-"
                cells.append("-")
            else:
                c = lookup_7[(a, b)]
                ws, bs = c["white_score"], c["black_score"]
                delta = ws - bs
                diff_mat[i, j] = delta
                row_csv[b] = f"{ws:.3f}/{bs:.3f} ({delta:+.3f})"
                cells.append(f"{ws:.3f} / {bs:.3f} ({delta:+.3f})")
        color_csv.append(row_csv)
        color_md.append(f"| `{a}` | " + " | ".join(cells) + " |")

    with (OUT_DIR / "matrix_7x7_color_diff.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", *agents_7])
        w.writeheader()
        w.writerows(color_csv)
    (OUT_DIR / "matrix_7x7_color_diff.md").write_text("\n".join(color_md) + "\n", encoding="utf-8")

    # Heatmap chênh lệch Trắng - Đen
    fig, ax = plt.subplots(figsize=(8.6, 6.8), dpi=300)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#E9ECEF")
    im = ax.imshow(diff_mat, cmap=cmap, vmin=-0.25, vmax=0.25, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Δ = Score(Trắng) − Score(Đen)", fontsize=10)
    labels = [SHORT_NAMES[a] for a in agents_7]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Đối thủ (cột j)", fontsize=10.5)
    ax.set_ylabel("Tác tử hàng i", fontsize=10.5)
    ax.set_title("Chênh lệch lợi thế đi trước (White Score − Black Score, 7×7)", fontsize=11.5, fontweight="bold", pad=12)
    for i, a in enumerate(agents_7):
        for j, b in enumerate(agents_7):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#495057", fontsize=10)
                continue
            c = lookup_7[(a, b)]
            ws, bs = c["white_score"], c["black_score"]
            delta = ws - bs
            ax.text(j, i, f"{delta:+.3f}\n({ws:.2f}/{bs:.2f})", ha="center", va="center", color="#111827", fontsize=8)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matrix_7x7_color_diff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_conditions_summary() -> None:
    rows = []

    def add_pair_csv(group: str, label: str, pair_path: Path, invert: bool = False) -> None:
        r = read_csv(pair_path)[0]
        w, d, l = int(r["W"]), int(r["D"]), int(r["L"])
        sc = float(r["score"])
        w_lo, w_hi = float(r["wilson_low"]), float(r["wilson_high"])
        b_lo, b_hi = float(r["bootstrap_low"]), float(r["bootstrap_high"])
        a_ms, b_ms = float(r["a_ms_mean"]), float(r["b_ms_mean"])
        pair_name = r["pair"]
        if invert:
            w, l = l, w
            sc = 1.0 - sc
            w_lo, w_hi = 1.0 - w_hi, 1.0 - w_lo
            b_lo, b_hi = 1.0 - b_hi, 1.0 - b_lo
            a_ms, b_ms = b_ms, a_ms
            a, b = pair_name.split("__", 1)
            pair_name = f"{b}__{a}"
        rows.append({
            "group": group, "label": label, "pair": pair_name, "N": int(r["N"]),
            "W": w, "D": d, "L": l, "score": f"{sc:.4f}",
            "wilson_95": f"[{w_lo:.4f}, {w_hi:.4f}]",
            "bootstrap_95": f"[{b_lo:.4f}, {b_hi:.4f}]",
            "ms_a": f"{a_ms:.1f}", "ms_b": f"{b_ms:.1f}",
        })

    def add_cond_csv(group: str, cond_path: Path) -> None:
        for r in read_csv(cond_path):
            rows.append({
                "group": group, "label": r["label"], "pair": r["pair"], "N": int(r["N"]),
                "W": int(r["W"]), "D": int(r["D"]), "L": int(r["L"]),
                "score": f"{float(r['score']):.4f}",
                "wilson_95": f"[{float(r['wilson_low']):.4f}, {float(r['wilson_high']):.4f}]",
                "bootstrap_95": f"[{float(r['bootstrap_low']):.4f}, {float(r['bootstrap_high']):.4f}]",
                "ms_a": f"{float(r['a_ms_mean']):.1f}", "ms_b": f"{float(r['b_ms_mean']):.1f}",
            })

    add_cond_csv("Ngân sách (A vs B)", ROOT / "results" / "stats_mode_A_vs_B" / "conditions.csv")
    add_cond_csv("Độ nhạy Seed (50 thế đầu)", ROOT / "results" / "stats_seed_42_vs_123" / "conditions.csv")
    add_cond_csv("Độ nhạy Khai cuộc", ROOT / "results" / "stats_bonus_openings" / "conditions.csv")
    # AB1: rl_policy__alphago -> đảo góc nhìn AlphaGo để đồng nhất với các điều kiện AlphaGo
    add_pair_csv("AB1 (Policy+Value+MCTS)", "AB1: AlphaGo vs RL Policy", ROOT / "results" / "stats_ab1" / "pairs.csv", invert=True)
    add_cond_csv("AB2 (λ Value vs Rollout)", ROOT / "results" / "stats_ab2" / "conditions.csv")
    add_pair_csv("AB4 (Cắt tỉa Alpha-Beta)", "AB4: Minimax vs AlphaBeta", ROOT / "results" / "stats_ab4" / "pairs.csv", invert=False)
    for r in read_csv(ROOT / "results" / "stats_bonus_timed_2s" / "pairs.csv"):
        rows.append({
            "group": "Ngân sách Timed 2s (Tesla T4)", "label": f"Timed 2s: {r['pair']}",
            "pair": r["pair"], "N": int(r["N"]),
            "W": int(r["W"]), "D": int(r["D"]), "L": int(r["L"]),
            "score": f"{float(r['score']):.4f}",
            "wilson_95": f"[{float(r['wilson_low']):.4f}, {float(r['wilson_high']):.4f}]",
            "bootstrap_95": f"[{float(r['bootstrap_low']):.4f}, {float(r['bootstrap_high']):.4f}]",
            "ms_a": f"{float(r['a_ms_mean']):.1f}", "ms_b": f"{float(r['b_ms_mean']):.1f}",
        })

    with (OUT_DIR / "matrix_conditions_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_lines = [
        "# Ma trận Tổng hợp các Chế độ Ngân sách, Độ nhạy & Ablation",
        "",
        "| Nhóm | Điều kiện | Cặp (A__B) | N | W–D–L (A) | Score (A) | Wilson 95% | Bootstrap 95% | ms TB (A / B) |",
        "| :--- | :--- | :--- | ---: | :---: | ---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['group']} | `{r['label']}` | `{r['pair']}` | {r['N']} | "
            f"{r['W']}-{r['D']}-{r['L']} | **{r['score']}** | {r['wilson_95']} | {r['bootstrap_95']} | "
            f"{r['ms_a']} / {r['ms_b']} |"
        )
    (OUT_DIR / "matrix_conditions_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agents_11 = [r["agent"] for r in read_csv(ROOT / "results" / "stats_final_A_11x11" / "agents.csv")]
    lookup_11 = build_pair_lookup(read_csv(ROOT / "results" / "stats_final_A_11x11" / "pairs.csv"))
    plot_score_heatmap(
        agents_11,
        lookup_11,
        "Ma trận Score đối đầu 11×11 sắp xếp theo thứ hạng (Chế độ A, 11.000 ván)",
        OUT_DIR / "matrix_11x11_heatmap.png",
        show_ci=False,
    )

    agents_7 = [r["agent"] for r in read_csv(ROOT / "results" / "stats_final_A_7x7" / "agents.csv")]
    lookup_7 = build_pair_lookup(read_csv(ROOT / "results" / "stats_final_A_7x7" / "pairs.csv"))
    plot_score_heatmap(
        agents_7,
        lookup_7,
        "Ma trận Score & Wilson 95% của 7 tác tử lõi (Chế độ A, 4.200 ván)",
        OUT_DIR / "matrix_7x7_heatmap.png",
        show_ci=True,
    )

    export_wdl_and_color_matrices(agents_7, lookup_7)
    export_conditions_summary()
    print(f"Đã xuất xong toàn bộ ma trận bổ trợ tại {OUT_DIR}")


if __name__ == "__main__":
    main()
