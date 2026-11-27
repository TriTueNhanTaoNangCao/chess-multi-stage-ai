"""Tạo experiment_cfg_final_A.json, experiment_cfg_final_B.json, experiment_cfg_ablation.json.

Chạy từ gốc repo:  python analysis/make_configs.py
Nguồn: thực nghiệm/chess/experiment_cfg_az_v2.json (không bị sửa).
Chỉ đổi giá trị, giữ đúng bộ khóa mà scripts/run_all_models.py yêu cầu.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "thực nghiệm" / "chess" / "experiment_cfg_az_v2.json"

PATHS = {
    "supervised": "checkpoints/stage4/best_lichess.pt",
    "rl_policy": "checkpoints/stage5/paper_frozen_2073/last.pt",
    "alphago_policy": "checkpoints/stage6/policy.pt",
    "alphago_value": "checkpoints/stage6/value.pt",
    "alphazero": "checkpoints/stage7/best.pt",
    "alphazero_v2": "checkpoints/stage7_v2/best.pt",
}


def by_id(cfg):
    return {a["id"]: a for a in cfg["agents"]}


def apply_paths(agents):
    agents["supervised"]["kwargs"]["checkpoint"] = PATHS["supervised"]
    agents["rl_policy"]["kwargs"]["checkpoint"] = PATHS["rl_policy"]
    agents["alphago"]["kwargs"]["checkpoint_policy"] = PATHS["alphago_policy"]
    agents["alphago"]["kwargs"]["checkpoint_value"] = PATHS["alphago_value"]
    agents["alphazero"]["kwargs"]["checkpoint"] = PATHS["alphazero"]
    agents["alphazero_v2"]["kwargs"]["checkpoint"] = PATHS["alphazero_v2"]


def set_budget(agents, sims, quiescence):
    for key in ("mcts_pure", "alphago", "alphazero", "alphazero_v2"):
        agents[key]["kwargs"]["simulations"] = sims
    kw = {"depth": 2}
    if quiescence:
        kw.update(q_depth=6, use_quiescence=True)
    else:
        kw.update(use_quiescence=False)
    agents["alphabeta"]["kwargs"] = kw


def common(cfg, args, output_dir):
    cfg["max_plies"] = 400
    cfg["claim_draw"] = True
    cfg["seed"] = args.seed
    cfg["games_per_pair"] = args.games
    cfg["device"] = args.device
    cfg["torch_threads"] = args.threads
    cfg["output_dir"] = output_dir


def make_mode(base, args, mode):
    cfg = copy.deepcopy(base)
    agents = by_id(cfg)
    apply_paths(agents)
    if mode == "A":
        set_budget(agents, 30, quiescence=False)
    else:
        set_budget(agents, 100, quiescence=True)
    common(cfg, args, f"results/final_{mode}_arena_s{args.seed}")
    return cfg


def agent(base_agents, src_id, new_id, **override):
    item = copy.deepcopy(base_agents[src_id])
    item["id"] = new_id
    item["kwargs"].update(override)
    return item


def make_ablation(base, args):
    cfg = make_mode(base, args, "A")
    cfg["games_per_pair"] = args.ablation_games
    a = by_id(cfg)
    agents = [
        a["minimax"], a["alphabeta"],
        agent(a, "alphabeta", "alphabeta_d1", depth=1),
        agent(a, "alphabeta", "alphabeta_d3", depth=3),
        a["mcts_pure"], a["rl_policy"],
        agent(a, "alphago", "alphago_l0", rollout_weight=0.0),
        a["alphago"],
        agent(a, "alphago", "alphago_l1", rollout_weight=1.0),
        agent(a, "alphazero_v2", "az2_s10", simulations=10),
        a["alphazero_v2"],
        agent(a, "alphazero_v2", "az2_s100", simulations=100),
        agent(a, "alphazero_v2", "az2_s200", simulations=200),
    ]
    cfg["agents"] = agents
    cfg["output_dir"] = f"results/ablation_s{args.seed}"
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=200, help="ván mỗi cặp ở chế độ A và B, số chẵn")
    parser.add_argument("--ablation-games", type=int, default=100, help="ván mỗi cặp ở file ablation, số chẵn")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-dir", default=str(SRC.parent))
    args = parser.parse_args()
    base = json.loads(SRC.read_text(encoding="utf-8-sig"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "experiment_cfg_final_A.json": make_mode(base, args, "A"),
        "experiment_cfg_final_B.json": make_mode(base, args, "B"),
        "experiment_cfg_ablation.json": make_ablation(base, args),
    }
    for name, cfg in files.items():
        (out / name).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Đã tạo", out / name, f"({len(cfg['agents'])} agent)")


if __name__ == "__main__":
    main()
