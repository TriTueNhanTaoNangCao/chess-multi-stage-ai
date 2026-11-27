"""Tái tạo và sửa các hình minh họa + biểu đồ thực nghiệm từ dữ liệu thật trong repo."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "report_figures"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#E8F1FA",
    ec: str = "#2C3E50",
    header: str | None = None,
    fontsize: float = 9.0,
    header_fc: str = "#DCE8F5",
    lw: float = 1.4,
    radius: float = 0.015,
) -> None:
    if header:
        hh = min(0.32 * h, 0.075)
        body = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(body)
        hdr = patches.Rectangle((x, y + h - hh), w, hh, facecolor=header_fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - hh / 2, header, ha="center", va="center", fontsize=fontsize + 0.5, fontweight="bold", color="#111827")
        ax.text(x + w / 2, y + (h - hh) / 2, text, ha="center", va="center", fontsize=fontsize - 0.3, color="#1F2937", linespacing=1.35)
    else:
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#111827", linespacing=1.35)


def draw_arrow(ax, x1: float, y1: float, x2: float, y2: float, *, dashed: bool = False, label: str = "") -> None:
    ls = "--" if dashed else "-"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.4, linestyle=ls, mutation_scale=12),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, label, ha="center", va="bottom", fontsize=8, color="#334155",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def make_fig_3_4_classes() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(
        ax, 0.03, 0.60, 0.31, 0.31,
        "get_legal_actions()\nstep(move)\nis_terminal()\ncheck_winner()\nclone()",
        header="BaseBoard «abstract»", fc="#F3F4F6", header_fc="#E5E7EB", radius=0.01, fontsize=9.2,
    )
    draw_box(
        ax, 0.03, 0.14, 0.31, 0.36,
        "board: chess.Board\nclaim_draw: bool\nreset(), pop(), fen()\noutcome(), san(move)\nget_result_relative()\ncurrent_player: ±1",
        header="ChessBoard", fc="#EAF2FB", header_fc="#D5E6F7", radius=0.01, fontsize=9.0,
    )
    draw_arrow(ax, 0.185, 0.50, 0.185, 0.60)

    draw_box(
        ax, 0.40, 0.72, 0.57, 0.21,
        "name, player_id\nact(board: ChessBoard) → chess.Move\nreset()",
        header="BaseAgent «abstract»", fc="#F3F4F6", header_fc="#E5E7EB", radius=0.01, fontsize=9.2,
    )
    # Nhánh kế thừa
    ax.plot([0.685, 0.685], [0.72, 0.675], color="#334155", lw=1.4)
    ax.plot([0.535, 0.835], [0.675, 0.675], color="#334155", lw=1.4)
    draw_arrow(ax, 0.535, 0.675, 0.535, 0.64)
    draw_arrow(ax, 0.835, 0.675, 0.835, 0.64)

    draw_box(
        ax, 0.40, 0.40, 0.27, 0.24,
        "Random: lấy mẫu đều\nThink1: chiếu hết 1 nước\nThink2: an toàn + MVV-LVA\nThink3: PST 1 ply",
        header="Random, Think1–3", fc="#FEF5E7", header_fc="#FDE7C6", radius=0.01, fontsize=8.6,
    )
    draw_box(
        ax, 0.70, 0.40, 0.27, 0.24,
        "Negamax theo độ sâu\nAlphaBeta: α-β, MVV-LVA,\nquiescence (q_depth=6)\nnodes_visited, cutoffs",
        header="Minimax, AlphaBeta", fc="#FEF5E7", header_fc="#FDE7C6", radius=0.01, fontsize=8.6,
    )
    draw_box(
        ax, 0.40, 0.14, 0.27, 0.23,
        "simulations, c\nmax_rollout_depth\nselect / expand /\nrollout / backprop",
        header="PureMCTSAgent", fc="#FEF5E7", header_fc="#FDE7C6", radius=0.01, fontsize=8.6,
    )
    draw_box(
        ax, 0.70, 0.14, 0.27, 0.23,
        "AlphaGo: policy + value\nAlphaZero: DualHeadResNet\nAlphaGoMCTS,\nAlphaZeroMCTS (PUCT)",
        header="AlphaGo, AlphaZero", fc="#FEF5E7", header_fc="#FDE7C6", radius=0.01, fontsize=8.5,
    )
    draw_box(
        ax, 0.40, 0.03, 0.57, 0.08,
        "SupervisedAgent, RLPolicyAgent (argmax có mặt nạ hợp lệ)",
        fc="#E6F4EA", radius=0.01, fontsize=9.0,
    )
    ax.text(
        0.03, 0.06,
        "Tất cả tác tử ở cột phải đều kế thừa BaseAgent\n(chỉ vẽ nét nối cho hàng đầu để tránh rối hình).",
        ha="left", va="center", fontsize=8.5, color="#4B5563",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_4_classes_fixed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_3_5_sequence() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = [(0.13, "Arena\nplay_game"), (0.37, "ChessBoard\n(state)"), (0.62, "Agent\n(bên đi)"), (0.87, "GameResult\n(bản ghi)")]
    for cx, lbl in cols:
        draw_box(ax, cx - 0.10, 0.86, 0.20, 0.10, lbl, fc="#E4EFFA", fontsize=9.5, radius=0.012)
        ax.plot([cx, cx], [0.06, 0.86], color="#94A3B8", ls="--", lw=1.3)

    draw_arrow(ax, 0.13, 0.80, 0.62, 0.80, label="reset()  (lỗi → reset_forfeit)")
    ax.text(0.03, 0.74, "Lặp đến khi kết thúc hoặc đủ max_plies (400 ply):", fontsize=8.8, color="#374151", style="italic")
    draw_arrow(ax, 0.13, 0.67, 0.37, 0.67, label="turn, is_terminal()")
    draw_arrow(ax, 0.13, 0.58, 0.37, 0.58, label="clone()  (bản sao giữ lịch sử)")
    draw_arrow(ax, 0.13, 0.49, 0.62, 0.49, label="act(clone)")
    draw_arrow(ax, 0.62, 0.40, 0.13, 0.40, dashed=True, label="chess.Move")
    draw_arrow(ax, 0.13, 0.31, 0.37, 0.31, label="is_legal(move)? sai/ngoại lệ → forfeit")
    draw_arrow(ax, 0.13, 0.22, 0.37, 0.22, label="step(move)")
    draw_arrow(ax, 0.13, 0.13, 0.87, 0.13, label="ghi UCI, seconds, telemetry, reason")
    ax.text(0.50, 0.04, 'Vượt max_plies (400) → "max_plies_adjudication" (hòa, ghi riêng)', ha="center", fontsize=8.8, color="#475569")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_5_sequence_fixed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_3_10_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.03, 0.70, 0.28, 0.22, "Lichess Elite PGN\n(01–03/2024, 300.000 ván)\nlọc trùng theo SHA256", fc="#F3F4F6", fontsize=9.0)
    draw_box(ax, 0.36, 0.70, 0.27, 0.22, "Dataset Stage 4\nchia theo ván (90% / 10%)\n~9,53 triệu mẫu", fc="#E8F1FA", fontsize=9.0)
    draw_box(ax, 0.68, 0.70, 0.29, 0.22, "Stage 4: CNN chính sách\n(Cross-Entropy + mask)\nbest_lichess_300k.pt", fc="#E6F4EA", fontsize=9.0)

    draw_box(ax, 0.03, 0.40, 0.28, 0.22, "Dữ liệu Value (3 pha)\nStage 4 + B16 tự chơi\n24.576 ván, chia 80/10/10", fc="#F3F4F6", fontsize=8.8)
    draw_box(ax, 0.36, 0.40, 0.27, 0.22, "Stage 5: ValueNetwork\n(MSE, tanh, epoch 2)\nval_mse = 0,2429", fc="#E6F4EA", fontsize=9.0)
    draw_box(ax, 0.68, 0.40, 0.29, 0.22, "Stage 5: REINFORCE RL\nB16 (400 iter) → Paper Policy\n(80 iter, baseline Value)", fc="#E6F4EA", fontsize=8.7)

    draw_box(
        ax, 0.03, 0.08, 0.60, 0.24,
        "Stage 6: AlphaGo-style MCTS\nprior = Paper Policy (Stage 5) · V_leaf = (1−λ)·Value + λ·rollout (15 ply)\n(hỗ trợ fallback chính sách Stage 4)",
        fc="#FEF3DF", fontsize=9.0,
    )
    draw_box(
        ax, 0.68, 0.08, 0.29, 0.24,
        "Stage 7: AlphaZero v2\nDualHeadResNet (6 khối dư)\nPUCT + tự chơi từ 0 (300 vòng)",
        fc="#FEF3DF", fontsize=8.8,
    )

    draw_arrow(ax, 0.31, 0.81, 0.36, 0.81)
    draw_arrow(ax, 0.63, 0.81, 0.68, 0.81)
    draw_arrow(ax, 0.825, 0.70, 0.825, 0.62)
    draw_arrow(ax, 0.31, 0.51, 0.36, 0.51)
    draw_arrow(ax, 0.495, 0.40, 0.495, 0.32)
    draw_arrow(ax, 0.76, 0.40, 0.60, 0.32)
    # Mũi tên nét đứt đi qua hành lang trống giữa ValueNetwork (0.63) và REINFORCE (0.68)
    ax.plot([0.72, 0.655, 0.655], [0.70, 0.65, 0.37], color="#334155", lw=1.4, ls="--")
    draw_arrow(ax, 0.655, 0.37, 0.56, 0.32, dashed=True)

    ax.text(0.03, 0.02, "Nét liền: luồng dữ liệu / trọng số chính thức của bản 300k; Nét đứt: nhánh khởi tạo/tùy chọn.", fontsize=8.3, color="#4B5563")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_10_pipeline_updated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_3_11_tourney() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.03, 0.68, 0.27, 0.23, "11 tác tử (Cấp 1–8)\nRandom, Think1–3, Minimax,\nAlphaBeta, MCTS, SL, RL,\nAlphaGo, AlphaZero v2", fc="#F3F4F6", fontsize=8.7)
    draw_box(ax, 0.35, 0.68, 0.29, 0.23, "Round-robin 55 cặp\n100 thế khai cuộc × 2 màu\n= 200 ván/cặp (11.000 ván)\ngiới hạn 400 ply, seed 42", fc="#E8F1FA", fontsize=8.7)
    draw_box(ax, 0.69, 0.68, 0.28, 0.23, "Độ nhạy & Tái lập\nLặp seed 123 (100 ván)\nTập thế openings_100_b\nAblation AB1, AB2, AB4", fc="#E8F1FA", fontsize=8.7)

    draw_box(ax, 0.03, 0.37, 0.44, 0.21, "Chế độ ngân sách tính toán cố định:\n• Chế độ A: 30 sims / depth 2 (tắt quiescence)\n• Chế độ Timed 2s: ≤ 2 giây/nước (Kaggle T4)", fc="#FEF3DF", fontsize=8.9)
    draw_box(ax, 0.53, 0.37, 0.44, 0.21, "Chế độ ngân sách mở rộng:\n• Chế độ B: 100 sims / depth 2 + quiescence (q=6)\n  (công bố đầy đủ tham số & checkpoint SHA256)", fc="#FEF3DF", fontsize=8.9)

    draw_box(ax, 0.03, 0.07, 0.94, 0.21, "Độ đo tổng hợp: W/D/L, Score = (W + 0,5·D)/N, khoảng tin cậy Wilson 95% & Bootstrap 95% (theo thế khai cuộc),\nđộ trễ TB / trung vị / p95, số nút / mô phỏng, đo RAM (RSS) & VRAM độc lập → Ma trận đối đầu 11×11 & 7×7", fc="#E6F4EA", fontsize=8.9)

    draw_arrow(ax, 0.30, 0.795, 0.35, 0.795)
    draw_arrow(ax, 0.64, 0.795, 0.69, 0.795)
    draw_arrow(ax, 0.46, 0.68, 0.30, 0.58)
    draw_arrow(ax, 0.54, 0.68, 0.70, 0.58)
    draw_arrow(ax, 0.25, 0.37, 0.25, 0.28)
    draw_arrow(ax, 0.75, 0.37, 0.75, 0.28)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_3_11_tourney_updated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_4_1_and_4_2() -> None:
    log_path = ROOT / "checkpoints" / "stage7_v2" / "training_log.csv"
    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    iters = np.array([int(r["iteration"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    ploss = np.array([float(r["policy_loss"]) for r in rows])
    vloss = np.array([float(r["value_loss"]) for r in rows])
    plies = np.array([float(r["avg_plies"]) for r in rows])
    lr = np.array([float(r["lr"]) for r in rows])

    # Tìm điểm giảm lr
    lr_drops = [iters[i] for i in range(1, len(lr)) if lr[i] < lr[i - 1]]

    # Hình 4.1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
    ax1.plot(iters, loss, color="#111827", lw=1.8, label="Tổng")
    ax1.plot(iters, ploss, color="#374151", lw=1.4, ls="--", label="Chính sách")
    ax1.plot(iters, vloss, color="#6B7280", lw=1.4, ls=":", label="Giá trị")
    for x in lr_drops:
        ax1.axvline(x, color="#9CA3AF", lw=1.0)
        ax1.text(x + 3, ax1.get_ylim()[1] * 0.88, "lr/10", fontsize=8.5, color="#4B5563")
    ax1.set_xlabel("Vòng huấn luyện")
    ax1.set_ylabel("Hàm mất mát")
    ax1.grid(alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, loc="center", fontsize=9)

    ax2.plot(iters, plies, color="#111827", lw=1.3)
    ax2.set_xlabel("Vòng huấn luyện")
    ax2.set_ylabel("Độ dài TB ván tự chơi (ply)")
    ax2.grid(alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_1_az2_loss_length.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Hình 4.2
    games = np.array([float(r["games"]) for r in rows])
    decisive = (np.array([float(r["white_wins"]) for r in rows]) + np.array([float(r["black_wins"]) for r in rows])) / games
    draws = np.array([float(r["draws"]) for r in rows]) / games
    capped = np.array([float(r["capped"]) for r in rows]) / games

    def moving_avg(arr: np.ndarray, win: int = 9) -> np.ndarray:
        half = win // 2
        out = np.empty_like(arr)
        for i in range(len(arr)):
            out[i] = arr[max(0, i - half) : min(len(arr), i + half + 1)].mean()
        return out

    ev_it, ev_w, ev_d, ev_c = [], [], [], []
    for r in rows:
        if r["eval_win"]:
            tot = float(r["eval_win"]) + float(r["eval_draw"]) + float(r["eval_loss"]) + float(r["eval_capped"])
            ev_it.append(int(r["iteration"]))
            ev_w.append(float(r["eval_win"]) / tot)
            ev_d.append(float(r["eval_draw"]) / tot)
            ev_c.append(float(r["eval_capped"]) / tot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
    ax1.plot(iters, moving_avg(decisive), color="#111827", lw=1.7, label="Có kết quả thắng/thua")
    ax1.plot(iters, moving_avg(draws), color="#374151", lw=1.4, ls="--", label="Hòa (lặp thế, 50 nước)")
    ax1.plot(iters, moving_avg(capped), color="#6B7280", lw=1.4, ls=":", label="Chạm giới hạn 200 ply")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Vòng huấn luyện")
    ax1.set_ylabel("Tỷ lệ ván tự chơi (TB trượt 9 vòng)")
    ax1.grid(alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax2.plot(ev_it, ev_w, "-o", color="#111827", ms=3.5, lw=1.3, label="Thắng")
    ax2.plot(ev_it, ev_d, "--s", color="#374151", ms=3.5, lw=1.2, label="Hòa")
    ax2.plot(ev_it, ev_c, ":^", color="#6B7280", ms=3.8, lw=1.2, label="Chưa kết thúc")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Vòng huấn luyện")
    ax2.set_ylabel("Tỷ lệ trên 16 ván trước Random")
    ax2.grid(alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_2_az2_outcomes_eval.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig_4_3_and_4_4() -> None:
    def read_one(p: Path):
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    mode_ab = read_one(ROOT / "results" / "stats_mode_A_vs_B" / "conditions.csv")
    seeds = read_one(ROOT / "results" / "stats_seed_42_vs_123" / "conditions.csv")
    ab2 = read_one(ROOT / "results" / "stats_ab2" / "conditions.csv")
    op_b = read_one(ROOT / "results" / "stats_bonus_openings" / "conditions.csv")[1]
    ab1_raw = read_one(ROOT / "results" / "stats_ab1" / "pairs.csv")[0]
    ab1_sc = 1.0 - float(ab1_raw["score"])
    ab1_wlo, ab1_whi = 1.0 - float(ab1_raw["wilson_high"]), 1.0 - float(ab1_raw["wilson_low"])
    ab1_blo, ab1_bhi = 1.0 - float(ab1_raw["bootstrap_high"]), 1.0 - float(ab1_raw["bootstrap_low"])

    entries = [
        ("Chế độ A: 30 mô phỏng, seed 42 (200 ván)", float(mode_ab[0]["score"]), float(mode_ab[0]["wilson_low"]), float(mode_ab[0]["wilson_high"]), float(mode_ab[0]["bootstrap_low"]), float(mode_ab[0]["bootstrap_high"])),
        ("Chế độ B: 100 mô phỏng, seed 42 (200 ván)", float(mode_ab[1]["score"]), float(mode_ab[1]["wilson_low"]), float(mode_ab[1]["wilson_high"]), float(mode_ab[1]["bootstrap_low"]), float(mode_ab[1]["bootstrap_high"])),
        ("Chế độ A, seed 42, 100 ván đầu", float(seeds[0]["score"]), float(seeds[0]["wilson_low"]), float(seeds[0]["wilson_high"]), float(seeds[0]["bootstrap_low"]), float(seeds[0]["bootstrap_high"])),
        ("Chế độ A, seed 123 (100 ván)", float(seeds[1]["score"]), float(seeds[1]["wilson_low"]), float(seeds[1]["wilson_high"]), float(seeds[1]["bootstrap_low"]), float(seeds[1]["bootstrap_high"])),
        ("Chế độ A, Openings B (100 ván)", float(op_b["score"]), float(op_b["wilson_low"]), float(op_b["wilson_high"]), float(op_b["bootstrap_low"]), float(op_b["bootstrap_high"])),
        ("AB2 λ = 0 (chỉ mạng giá trị, 100 ván)", float(ab2[0]["score"]), float(ab2[0]["wilson_low"]), float(ab2[0]["wilson_high"]), float(ab2[0]["bootstrap_low"]), float(ab2[0]["bootstrap_high"])),
        ("AB2 λ = 1 (chỉ rollout, 100 ván)", float(ab2[2]["score"]), float(ab2[2]["wilson_low"]), float(ab2[2]["wilson_high"]), float(ab2[2]["bootstrap_low"]), float(ab2[2]["bootstrap_high"])),
        ("AB1: AlphaGo trước RL policy-only (100 ván)", ab1_sc, ab1_wlo, ab1_whi, ab1_blo, ab1_bhi),
    ]

    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=300)
    y_pos = np.arange(len(entries))[::-1]
    ax.axvline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    for y, (_, sc, wlo, whi, blo, bhi) in zip(y_pos, entries):
        ax.plot([wlo, whi], [y, y], color="#111827", lw=1.6)
        ax.plot([blo, bhi], [y, y], color="#9CA3AF", lw=6.0, alpha=0.85, zorder=2)
        ax.scatter([sc], [y], color="#111827", s=40, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in entries], fontsize=9.2)
    ax.set_xlim(0.45, 1.01)
    ax.set_xlabel("Score của AlphaGo (đường mảnh: Wilson 95%; đoạn dày: Bootstrap 95%)", fontsize=9.8)
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_3_alphago_conditions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Hình 4.4 từ stats_ab4/agents.csv
    ab4_agents = {r["agent"]: r for r in read_one(ROOT / "results" / "stats_ab4" / "agents.csv")}
    mm, ab = ab4_agents["minimax"], ab4_agents["alphabeta"]
    names = ["Minimax", "Alpha-Beta"]
    nodes = [float(mm["nodes_visited_mean"]), float(ab["nodes_visited_mean"])]
    ms = [float(mm["ms_mean"]), float(ab["ms_mean"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=300)
    b1 = ax1.bar(names, nodes, width=0.45, color=["#9CA3AF", "#374151"])
    ax1.bar_label(b1, labels=[f"{v:.0f}" for v in nodes], padding=3, fontsize=9.5)
    ax1.set_ylim(0, max(nodes) * 1.15)
    ax1.set_ylabel("Số nút duyệt TB mỗi nước")
    ax1.grid(axis="y", alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)

    b2 = ax2.bar(names, ms, width=0.45, color=["#9CA3AF", "#374151"])
    ax2.bar_label(b2, labels=[f"{v:.1f}" for v in ms], padding=3, fontsize=9.5)
    ax2.set_ylim(0, max(ms) * 1.15)
    ax2.set_ylabel("Thời gian TB mỗi nước (ms)")
    ax2.grid(axis="y", alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_4_4_ab4_minimax_alphabeta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_fig_3_4_classes()
    make_fig_3_5_sequence()
    make_fig_3_10_pipeline()
    make_fig_3_11_tourney()
    make_fig_4_1_and_4_2()
    make_fig_4_3_and_4_4()
    print(f"Đã tạo xong bộ hình báo cáo tại {OUT_DIR}")


if __name__ == "__main__":
    main()
