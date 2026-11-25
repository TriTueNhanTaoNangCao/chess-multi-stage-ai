"""Phân chia các cặp đấu thành N luồng chạy PowerShell theo thuật toán tham lam (Task T7)."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_all_models import load_config


DEFAULT_WEIGHTS = {
    "random": 1,
    "minimax": 2,
    "alphabeta": 2,
    "supervised": 2,
    "rl_policy": 2,
    "mcts_pure": 6,
    "alphago": 8,
    "alphazero": 8,
    "alphazero_v2": 8,
}


def resolve_weight(agent_spec: dict, overrides: dict[str, int]) -> int:
    aid = agent_spec["id"]
    key = agent_spec["key"]
    if aid in overrides:
        return overrides[aid]
    if key in overrides:
        return overrides[key]
    if aid.startswith("think") or key.startswith("think"):
        return 1
    if aid in DEFAULT_WEIGHTS:
        return DEFAULT_WEIGHTS[aid]
    if key in DEFAULT_WEIGHTS:
        return DEFAULT_WEIGHTS[key]
    return 4


def parse_weights(items: list[str] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--weights phải có dạng id=số, nhận được: {item!r}")
        k, v = item.split("=", 1)
        result[k.strip()] = int(v.strip())
    return result


def normalize_pair(pair_str: str) -> frozenset[str]:
    if "__" not in pair_str:
        raise ValueError(f"Tên cặp loại trừ phải có dạng A__B, nhận được: {pair_str!r}")
    a, b = pair_str.split("__", 1)
    return frozenset((a, b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lập lịch phân chia các cặp đấu cho N luồng.")
    parser.add_argument("--config", required=True, help="Đường dẫn file config JSON")
    parser.add_argument("--agents", nargs="*", default=None, help="Danh sách agent ID")
    parser.add_argument("--root", required=True, help="Thư mục gốc lưu kết quả từng cặp")
    parser.add_argument("--split", type=int, default=1, help="Số luồng chạy song song (mặc định 1)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Các cặp A__B cần bỏ qua")
    parser.add_argument("--weights", nargs="*", default=[], help="Trọng số tùy chỉnh dạng id=trọng_số")
    args = parser.parse_args()

    if args.split < 1:
        raise ValueError("--split phải >= 1")

    cfg = load_config(args.config, selected=args.agents or None)
    overrides = parse_weights(args.weights)
    excluded = {normalize_pair(p) for p in args.exclude}

    pairs = []
    for a, b in combinations(cfg["agents"], 2):
        if frozenset((a["id"], b["id"])) in excluded:
            continue
        cost = resolve_weight(a, overrides) + resolve_weight(b, overrides)
        pairs.append((cost, a["id"], b["id"]))

    # Tham lam: cặp nặng nhất vào luồng đang có tổng chi phí nhỏ nhất
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    streams: list[dict] = [{"cost": 0, "pairs": []} for _ in range(args.split)]
    for cost, a_id, b_id in pairs:
        target = min(streams, key=lambda s: s["cost"])
        target["pairs"].append((cost, a_id, b_id))
        target["cost"] += cost

    root_path = Path(args.root)
    root_path.mkdir(parents=True, exist_ok=True)
    cfg_rel = str(Path(args.config).as_posix())

    print(f"Tổng số cặp: {len(pairs)} | Số luồng: {args.split}")
    print("| Luồng | Số cặp | Tổng chi phí | File script |")
    print("|---:|---:|---:|---|")
    for idx, stream in enumerate(streams, start=1):
        ps1_path = root_path / f"run_machine_{idx}.ps1"
        lines = [
            f"# Luồng {idx}: {len(stream['pairs'])} cặp, tổng chi phí = {stream['cost']}",
            '$ErrorActionPreference = "Stop"',
            "",
        ]
        for cost, a_id, b_id in stream["pairs"]:
            out_dir = f"{root_path.as_posix()}/{a_id}__{b_id}"
            lines.append(f"# {a_id}__{b_id} (chi phí {cost})")
            lines.append(
                f'python scripts/run_all_models.py run --config "{cfg_rel}" --agents {a_id} {b_id} --output "{out_dir}"'
            )
            lines.append('if ($LASTEXITCODE -ne 0) { throw "run thất bại" }')
            lines.append(
                f'python scripts/run_all_models.py verify --output "{out_dir}"'
            )
            lines.append('if ($LASTEXITCODE -ne 0) { throw "verify thất bại" }')
            lines.append("")
        ps1_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"| {idx} | {len(stream['pairs'])} | {stream['cost']} | {ps1_path.as_posix()} |")


if __name__ == "__main__":
    main()
