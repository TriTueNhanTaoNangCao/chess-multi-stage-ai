#!/usr/bin/env python
"""
So sánh trực tiếp hai checkpoint AlphaZero (ví dụ v1 với v2, hoặc bản vòng 10 với bản cuối).

Dùng đúng giao thức của scripts/benchmark_stage6_vs_stage7.py: kho khai cuộc 50 thế, mỗi cặp
ván đổi màu, không nhiễu, chọn nước thăm nhiều nhất, cùng số mô phỏng cho hai bên, Wilson 95%.

Ví dụ (đo v2 so với v1 với 100 ván, 30 mô phỏng):
    python scripts/benchmark_az_checkpoints.py --a checkpoints/stage7_v2/best.pt \
        --b checkpoints/stage7/best.pt --games 100 --simulations 30
Điểm (score) tính cho checkpoint A: (thắng + 0,5 hòa) / số ván.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from agents import build_agent  # noqa: E402
from arena.evaluator import play_game  # noqa: E402
from benchmark_stage6_vs_stage7 import compute_sha256, compute_wilson_ci, create_opening_board  # noqa: E402


def make_agent(path: str, name: str, sims: int, seed: int, device: str):
    agent = build_agent("alphazero", depth=1, seed=seed, device=device, checkpoint=path)
    agent.simulations = sims
    agent.mcts.simulations = sims
    agent.add_noise = False
    agent.name = name
    return agent


def main() -> None:
    ap = argparse.ArgumentParser(description="Đối đầu hai checkpoint AlphaZero")
    ap.add_argument("--a", required=True, help="Checkpoint A (ví dụ bản mới)")
    ap.add_argument("--b", required=True, help="Checkpoint B (ví dụ bản cũ)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--simulations", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/az_checkpoints/result.json")
    args = ap.parse_args()
    if args.games < 2 or args.games % 2:
        raise SystemExit("--games phải là số chẵn để cân bằng màu")
    torch.set_num_threads(args.threads)

    wins = draws = losses = 0
    t0 = time.time()
    for i in range(args.games):
        a = make_agent(args.a, "A", args.simulations, args.seed + i * 37, args.device)
        b = make_agent(args.b, "B", args.simulations, args.seed + i * 37 + 13, args.device)
        board = create_opening_board(i // 2)
        a_white = i % 2 == 0
        rec = play_game(a if a_white else b, b if a_white else a, max_plies=args.max_plies, board=board)
        score_white = 0.5 if rec.result == "1/2-1/2" else (1.0 if rec.result == "1-0" else 0.0)
        score_a = score_white if a_white else 1.0 - score_white
        wins += score_a == 1.0
        draws += score_a == 0.5
        losses += score_a == 0.0
        print(f"[{i + 1:3d}/{args.games}] A {'Trắng' if a_white else 'Đen '} -> {rec.result} | "
              f"A: {wins}T {draws}H {losses}B | {time.time() - t0:.0f}s", flush=True)

    total = args.games
    score = (wins + 0.5 * draws) / total
    lo, hi = compute_wilson_ci(wins + 0.5 * draws, total)
    result = {
        "a": args.a, "b": args.b,
        "sha256_a": compute_sha256(args.a), "sha256_b": compute_sha256(args.b),
        "games": total, "simulations": args.simulations, "max_plies": args.max_plies,
        "a_wins": int(wins), "draws": int(draws), "a_losses": int(losses),
        "a_score": score, "wilson95": [lo, hi], "seconds": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nA: {wins} thắng, {draws} hòa, {losses} thua | score {score:.4f} | Wilson 95%: [{lo:.4f}, {hi:.4f}]")
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
