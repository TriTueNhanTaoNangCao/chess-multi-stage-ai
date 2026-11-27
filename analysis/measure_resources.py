"""Đo tài nguyên RAM, VRAM và độ trễ nước đi trên tiến trình spawn độc lập (Task T6)."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_all_models import build_agent, load_config, opening_board, project_path


def _rss_and_peak_mb(proc: psutil.Process) -> tuple[float, float]:
    mem = proc.memory_info()
    rss_mb = mem.rss / (1024.0 ** 2)
    peak_wset = getattr(mem, "peak_wset", None)
    peak_mb = max(rss_mb, peak_wset / (1024.0 ** 2)) if peak_wset is not None else rss_mb
    return rss_mb, peak_mb


def _measure_worker(
    conn: Any,
    spec: dict[str, Any],
    seed: int,
    device: str,
    claim_draw: bool,
    torch_threads: int,
    openings: list[list[str]],
    moves_per_opening: int,
) -> None:
    try:
        import torch

        torch.set_num_threads(torch_threads)
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Đã chọn --device cuda nhưng CUDA không khả dụng")
            torch.cuda.reset_peak_memory_stats()

        proc = psutil.Process(os.getpid())
        rss_before_mb, rss_peak_mb = _rss_and_peak_mb(proc)

        agent = build_agent(spec, seed, device, claim_draw)
        agent.reset()
        rss_after_load_mb, peak_now = _rss_and_peak_mb(proc)
        rss_peak_mb = max(rss_peak_mb, peak_now)

        ms_samples: list[float] = []
        for opening in openings:
            board = opening_board(opening, claim_draw)
            agent.reset()
            for _ in range(moves_per_opening):
                if board.is_terminal():
                    break
                snap = board.clone()
                started = perf_counter()
                move = agent.act(snap)
                elapsed_ms = (perf_counter() - started) * 1000.0
                ms_samples.append(elapsed_ms)
                board.step(move)
                _, peak_now = _rss_and_peak_mb(proc)
                rss_peak_mb = max(rss_peak_mb, peak_now)

        if device == "cuda":
            vram_alloc_peak_mb: float | str = round(torch.cuda.max_memory_allocated() / (1024.0 ** 2), 3)
            vram_reserved_peak_mb: float | str = round(torch.cuda.max_memory_reserved() / (1024.0 ** 2), 3)
        else:
            vram_alloc_peak_mb = "null"
            vram_reserved_peak_mb = "null"

        arr = np.array(ms_samples, dtype=np.float64)
        row = {
            "agent": spec["id"],
            "device": device,
            "moves": len(ms_samples),
            "rss_before_mb": round(rss_before_mb, 3),
            "rss_after_load_mb": round(rss_after_load_mb, 3),
            "rss_peak_mb": round(rss_peak_mb, 3),
            "vram_alloc_peak_mb": vram_alloc_peak_mb,
            "vram_reserved_peak_mb": vram_reserved_peak_mb,
            "ms_mean": round(float(arr.mean()), 3) if len(arr) else 0.0,
            "ms_p95": round(float(np.percentile(arr, 95, method="linear")), 3) if len(arr) else 0.0,
        }
        conn.send(("ok", row))
    except Exception as exc:
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run_single_agent(
    spec: dict[str, Any],
    seed: int,
    device: str,
    claim_draw: bool,
    torch_threads: int,
    openings: list[list[str]],
    moves_per_opening: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_measure_worker,
        args=(send_conn, spec, seed, device, claim_draw, torch_threads, openings, moves_per_opening),
    )
    proc.start()
    send_conn.close()
    try:
        kind, payload = recv_conn.recv()
        if kind != "ok":
            raise RuntimeError(f"Lỗi đo tài nguyên agent {spec['id']}: {payload}")
        return payload
    finally:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        recv_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo RAM/VRAM và latency cho từng agent.")
    parser.add_argument("--config", required=True, help="Đường dẫn file config JSON")
    parser.add_argument("--agents", nargs="*", default=None, help="Danh sách agent ID (mặc định tất cả)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Thiết bị đo")
    parser.add_argument("--moves", type=int, default=20, help="Số nước mỗi thế mở đầu")
    parser.add_argument("--openings", type=int, default=5, help="Số thế mở đầu đầu tiên")
    parser.add_argument("--output", required=True, help="Đường dẫn file CSV đầu ra")
    args = parser.parse_args()

    if args.moves < 1 or args.openings < 1:
        raise ValueError("--moves và --openings phải >= 1")

    cfg = load_config(args.config, selected=args.agents or None)
    device = args.device or cfg["device"]
    raw_openings = json.loads(project_path(cfg["openings_file"]).read_text(encoding="utf-8"))["openings"]
    openings = raw_openings[: args.openings]
    if len(openings) < args.openings:
        raise ValueError(f"File khai cuộc chỉ có {len(raw_openings)} thế, cần {args.openings}")

    rows: list[dict[str, Any]] = []
    for spec in cfg["agents"]:
        print(f"Đang đo {spec['id']} trên {device}...", flush=True)
        row = run_single_agent(
            spec,
            cfg["seed"],
            device,
            cfg["claim_draw"],
            cfg["torch_threads"],
            openings,
            args.moves,
        )
        rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent",
        "device",
        "moves",
        "rss_before_mb",
        "rss_after_load_mb",
        "rss_peak_mb",
        "vram_alloc_peak_mb",
        "vram_reserved_peak_mb",
        "ms_mean",
        "ms_p95",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {len(rows)} dòng vào {out_path}")


if __name__ == "__main__":
    main()
