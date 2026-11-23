#!/usr/bin/env python
"""
Stage 7 (v2): huấn luyện AlphaZero sát bài báo hơn bản v1 (training/stage7_train_alphazero.py).

Điểm khác so với v1
- Tự chơi nhiều ván song song, suy luận gom theo lô (GPU nếu có), nhiều tiến trình (--workers).
- Ván tự chơi dài hơn (mặc định 200 ply thay vì 80) và dùng luật hòa đúng (lặp thế 3 lần, 50 nước).
- Ván bị cắt vì hết giới hạn ply KHÔNG có nhãn kết quả thật: bỏ qua Value của ván đó
  (value_mask = False) thay vì gán z = 0 như v1. Policy của ván đó vẫn được học.
- Mục tiêu Policy là phân bố số lượt thăm với tau = 1 (tau chỉ dùng để chọn nước đi:
  tau = 1 trong 30 ply đầu, sau đó chọn nước thăm nhiều nhất).
- Giữ cây con của nước đã đi; nhiễu Dirichlet (alpha 0,3, epsilon 0,25) ở nút gốc.
- SGD momentum 0,9, weight decay 1e-4 (L2), giảm learning rate theo bậc (như bài báo).
- Bộ đệm replay lớn hơn (mặc định 100.000 mẫu), lưu thưa nên nhẹ bộ nhớ.
- Có thể dừng giữa chừng và chạy tiếp (--resume); lưu replay buffer định kỳ.
- Checkpoint cùng định dạng v1, nạp được bằng AlphaZeroAgent và benchmark.
  Thư mục mặc định checkpoints/stage7_v2 (KHÔNG ghi đè checkpoints/stage7 của v1).

Chạy thử nhanh (vài phút, kiểm tra cài đặt):
    python training/stage7_train_alphazero_v2.py --smoke
Đo tốc độ trước khi chạy lâu:
    python training/stage7_train_alphazero_v2.py --bench --workers 3 --simulations 100
Chạy thật:
    python training/stage7_train_alphazero_v2.py --iterations 300 --workers 3 --simulations 100
Chạy tiếp sau khi bị dừng:
    python training/stage7_train_alphazero_v2.py --resume (giữ nguyên các tham số cũ)
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from agents.stage7_alphazero.batched_mcts import SearchConfig, play_games  # noqa: E402
from games.chess.encoder import NUM_ACTIONS  # noqa: E402
from models.dual_head_resnet import DualHeadResNet  # noqa: E402

LOG_FIELDS = [
    "iteration", "games", "avg_plies", "white_wins", "black_wins", "draws", "capped",
    "samples_added", "buffer", "value_mask_frac", "loss", "policy_loss", "value_loss", "lr",
    "selfplay_seconds", "train_seconds", "sims_per_sec", "elapsed_seconds",
    "eval_win", "eval_draw", "eval_loss", "eval_capped", "adjudicated",
]


# --------------------------------------------------------------------------------------
# Replay buffer (lưu thưa: chỉ giữ chỉ số và xác suất của các nước hợp lệ)
# --------------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.data: List[tuple] = []
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add_records(self, records: List[Dict[str, Any]]) -> int:
        added = 0
        for rec in records:
            t = len(rec["z"])
            for i in range(t):
                item = (rec["states"][i], rec["idxs"][i], rec["probs"][i], float(rec["z"][i]), bool(rec["vmask"][i]))
                if len(self.data) < self.capacity:
                    self.data.append(item)
                else:
                    self.data[self.pos] = item
                    self.pos = (self.pos + 1) % self.capacity
                added += 1
        return added

    def sample(self, batch_size: int, device):
        n = len(self.data)
        ids = self.rng.integers(0, n, size=batch_size)
        items = [self.data[i] for i in ids]
        states = np.stack([it[0] for it in items]).astype(np.float32)
        lens = np.fromiter((len(it[1]) for it in items), dtype=np.int64, count=batch_size)
        rows = np.repeat(np.arange(batch_size), lens)
        cols = np.concatenate([it[1] for it in items]).astype(np.int64)
        vals = np.concatenate([it[2] for it in items])
        pi = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
        pi[rows, cols] = vals
        z = np.array([it[3] for it in items], dtype=np.float32)
        m = np.array([it[4] for it in items], dtype=np.float32)
        return (
            torch.from_numpy(states).to(device),
            torch.from_numpy(pi).to(device),
            torch.from_numpy(z).to(device),
            torch.from_numpy(m).to(device),
        )

    def save(self, path: Path) -> None:
        if not self.data:
            return
        lens = np.array([len(it[1]) for it in self.data], dtype=np.int64)
        tmp = Path(str(path) + ".tmp.npz")
        np.savez(
            tmp,
            states=np.stack([it[0] for it in self.data]),
            idx=np.concatenate([it[1] for it in self.data]),
            prob=np.concatenate([it[2] for it in self.data]),
            lens=lens,
            z=np.array([it[3] for it in self.data], dtype=np.float32),
            m=np.array([it[4] for it in self.data], dtype=bool),
            pos=np.array([self.pos]),
        )
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        d = np.load(path)
        states, idx, prob = d["states"], d["idx"], d["prob"]
        lens, z, m = d["lens"], d["z"], d["m"]
        offs = np.concatenate([[0], np.cumsum(lens)])
        self.data = [
            (states[i], idx[offs[i]:offs[i + 1]], prob[offs[i]:offs[i + 1]], float(z[i]), bool(m[i]))
            for i in range(len(lens))
        ]
        self.pos = int(d["pos"][0]) % max(1, self.capacity)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity:]
            self.pos = 0


# --------------------------------------------------------------------------------------
# Hàm mất mát và bước huấn luyện
# --------------------------------------------------------------------------------------
def compute_loss(logits, values, pi, z, mask):
    """L = (z - v)^2 (chỉ mẫu có nhãn) - pi^T log p. L2 do weight decay của bộ tối ưu đảm nhận."""
    log_p = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi * log_p).sum(dim=-1).mean()
    v = values.reshape(-1)
    denom = mask.sum().clamp(min=1.0)
    value_loss = (((v - z) ** 2) * mask).sum() / denom
    return policy_loss + value_loss, policy_loss, value_loss


def train_steps(model, optimizer, buffer: ReplayBuffer, steps: int, batch_size: int, device, clip: float):
    model.train()
    tot = pol = val = 0.0
    for _ in range(steps):
        x, pi, z, m = buffer.sample(batch_size, device)
        logits, values = model(x)
        loss, p_loss, v_loss = compute_loss(logits, values, pi, z, m)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        tot += float(loss.item())
        pol += float(p_loss.item())
        val += float(v_loss.item())
    model.eval()
    steps = max(1, steps)
    return {"loss": tot / steps, "policy_loss": pol / steps, "value_loss": val / steps}


def lr_for_iteration(base_lr: float, it: int, total: int, milestones: List[float]) -> float:
    passed = sum(1 for f in milestones if it > int(f * total))
    return base_lr * (0.1 ** passed)


# --------------------------------------------------------------------------------------
# Tiến trình phụ: tự chơi / đánh giá
# --------------------------------------------------------------------------------------
_W: Dict[str, Any] = {}


def _worker_init(device_str: str, threads: int) -> None:
    torch.set_num_threads(threads)
    _W["device"] = torch.device(device_str)
    _W["model"] = None
    _W["mcfg"] = None


def _worker_task(task):
    weights, model_cfg, cfg_dict, n_games, seed, eval_random = task
    if _W.get("model") is None or _W.get("mcfg") != model_cfg:
        model = DualHeadResNet.from_config(model_cfg).to(_W["device"])
        _W["model"], _W["mcfg"] = model, dict(model_cfg)
    model = _W["model"]
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    return play_games(model, _W["device"], n_games, SearchConfig(**cfg_dict), seed, eval_random=eval_random)


def state_dict_numpy(model) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def generate(pool, model, device, n_games: int, cfg: SearchConfig, seed: int, workers: int, eval_random: bool = False):
    """Sinh n_games ván (tự chơi hoặc đánh giá). Trả về (records, stats)."""
    cfg_dict = dict(cfg.__dict__) if hasattr(cfg, "__dict__") else dict(cfg.__dataclass_fields__)
    if pool is None:
        return play_games(model, device, n_games, cfg, seed, eval_random=eval_random)
    weights = state_dict_numpy(model)
    base, extra = divmod(n_games, workers)
    tasks = []
    for w in range(workers):
        n = base + (1 if w < extra else 0)
        if n > 0:
            tasks.append((weights, model.get_config(), cfg_dict, n, seed * 1000 + w, eval_random))
    results = pool.map(_worker_task, tasks, chunksize=1)
    records: List[Dict[str, Any]] = []
    stats = {"sims": 0, "nn_evals": 0, "nn_batches": 0, "nn_seconds": 0.0, "games": 0}
    for recs, st in results:
        records.extend(recs)
        for k in stats:
            stats[k] += st.get(k, 0)
    return records, stats


# --------------------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------------------
def _jsonable(args: argparse.Namespace) -> Dict[str, Any]:
    out = {}
    for k, v in vars(args).items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)
    return out


def _atomic_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def make_payload(model, args, step: int, metrics: Dict[str, float]) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "stage": 7,
        "az_version": 2,
        "model_class": "DualHeadResNet",
        "model_config": model.get_config(),
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": NUM_ACTIONS,
        "value_perspective": "side_to_move",
        "claim_draw": True,
        "seed": args.seed,
        "step": step,
        "training_config": _jsonable(args),
        "metrics": metrics,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


# --------------------------------------------------------------------------------------
# Tham số
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero v2 (Stage 7): tự chơi song song, SGD theo bài báo")
    p.add_argument("--checkpoint-dir", default="checkpoints/stage7_v2")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--games-per-iter", type=int, default=96, help="Số ván tự chơi mỗi vòng")
    p.add_argument("--simulations", type=int, default=100, help="Số mô phỏng MCTS mỗi nước khi tự chơi")
    p.add_argument("--parallel-games", type=int, default=32, help="Số ván chạy đồng thời trong mỗi tiến trình")
    p.add_argument("--workers", type=int, default=3, help="Số tiến trình tự chơi (0 = chạy trong tiến trình chính)")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--temp-plies", type=int, default=30)
    p.add_argument("--adjudicate-material", type=float, default=0.0,
                   help="(không có trong bài báo) >0: ván hết giới hạn ply được chấm theo vật chất nếu chênh >= ngưỡng "
                        "(đơn vị tốt, gợi ý 3). 0 = tắt, ván bị cắt không có nhãn Value")
    p.add_argument("--adjudicate-draws", action="store_true",
                   help="(không có trong bài báo) cần --adjudicate-material > 0: ván hòa do lặp thế hoặc luật 50 nước "
                        "cũng được chấm theo vật chất. Có thể bật khi --resume vì chỉ đổi cách gán nhãn")
    p.add_argument("--random-opening-plies", type=int, default=0,
                   help="(không có trong bài báo) mỗi ván tự chơi bắt đầu bằng 0..N nước ngẫu nhiên (gợi ý 6). 0 = tắt")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--blocks", type=int, default=6, help="Số khối residual")
    p.add_argument("--channels", type=int, default=128, help="Số kênh của thân mạng")
    p.add_argument("--buffer-size", type=int, default=100000, help="Sức chứa replay buffer (số mẫu)")
    p.add_argument("--min-buffer", type=int, default=4000, help="Số mẫu tối thiểu trước khi bắt đầu học")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-steps", type=int, default=150, help="Số bước cập nhật mỗi vòng")
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-milestones", default="0.6,0.85", help="Các mốc (tỉ lệ số vòng) giảm lr 10 lần")
    p.add_argument("--clip-grad", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=5, help="Đánh giá với đối thủ ngẫu nhiên mỗi N vòng (0 = tắt)")
    p.add_argument("--eval-games", type=int, default=16)
    p.add_argument("--eval-sims", type=int, default=50)
    p.add_argument("--eval-max-plies", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=10, help="Lưu bản chụp trọng số mỗi N vòng (0 = tắt)")
    p.add_argument("--save-buffer-every", type=int, default=5)
    p.add_argument("--max-hours", type=float, default=0.0, help="Tự dừng sau số giờ này (0 = không giới hạn)")
    p.add_argument("--device", default="auto", help="auto, cpu hoặc cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bench", action="store_true", help="Chỉ đo tốc độ tự chơi rồi thoát")
    p.add_argument("--smoke", action="store_true", help="Chạy thử rất nhỏ để kiểm tra cài đặt")
    args = p.parse_args(argv)
    if args.smoke:
        args.iterations = 2
        args.games_per_iter = 4
        args.simulations = 8
        args.parallel_games = 4
        args.workers = 0
        args.blocks = 2
        args.channels = 32
        args.train_steps = 5
        args.batch_size = 32
        args.min_buffer = 64
        args.eval_every = 1
        args.eval_games = 2
        args.eval_sims = 8
        args.max_plies = 60
        args.eval_max_plies = 60
        args.buffer_size = 5000
        args.checkpoint_dir = "checkpoints/stage7_v2_smoke"
        args.save_buffer_every = 1
        args.snapshot_every = 1
    return args


def build_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def main(argv: Optional[List[str]] = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args(argv)
    if args.smoke and Path(args.checkpoint_dir).exists():
        import shutil
        shutil.rmtree(args.checkpoint_dir)
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.manual_seed(args.seed)
    cpu_total = os.cpu_count() or 4
    torch.set_num_threads(max(1, cpu_total if args.workers == 0 else 2))

    ckpt_dir = Path(args.checkpoint_dir)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    buffer_path = ckpt_dir / "replay.npz"
    log_path = ckpt_dir / "training_log.csv"

    if latest_path.exists() and not args.resume and not args.bench:
        raise SystemExit(
            f"Đã có {latest_path}. Dùng --resume để chạy tiếp, hoặc đổi --checkpoint-dir để chạy mới "
            "(chương trình không ghi đè kết quả cũ)."
        )

    start_it = 0
    payload = None
    if args.resume and latest_path.exists():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        model = DualHeadResNet.from_config(payload["model_config"])
        model.load_state_dict(payload["state_dict"])
        start_it = int(payload["iteration"])
    else:
        model = DualHeadResNet(hidden_dim=args.channels, num_res_blocks=args.blocks)
    model.to(device).eval()
    optimizer = build_optimizer(model, args)
    if payload is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    buffer = ReplayBuffer(args.buffer_size, seed=args.seed + start_it)
    if args.resume and buffer_path.exists():
        buffer.load(buffer_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Thiết bị: {device} | mạng: {args.blocks} khối x {args.channels} kênh, {n_params / 1e6:.2f}M tham số "
          f"(đang dùng cấu hình {model.get_config()})")
    print(f"Tự chơi: {args.games_per_iter} ván/vòng, {args.simulations} mô phỏng/nước, "
          f"{args.workers} tiến trình x {args.parallel_games} ván song song, giới hạn {args.max_plies} ply")

    pool = None
    if args.workers > 0:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers, initializer=_worker_init, initargs=(device_str, 1))

    sp_cfg = SearchConfig(
        simulations=args.simulations, c_puct=args.c_puct, temp_plies=args.temp_plies,
        max_plies=args.max_plies, parallel_games=args.parallel_games,
        adjudicate_material=args.adjudicate_material, random_opening_plies=args.random_opening_plies,
        adjudicate_draws=args.adjudicate_draws,
    )

    try:
        if args.bench:
            n = max(args.workers, 1) * args.parallel_games
            t0 = time.time()
            records, st = generate(pool, model, device, n, sp_cfg, args.seed, max(args.workers, 1) if pool else 1)
            dt = time.time() - t0
            plies = sum(r["plies"] for r in records)
            gph = len(records) / dt * 3600
            print(f"[bench] {len(records)} ván, {plies / max(1, len(records)):.0f} ply/ván trung bình, "
                  f"{st['sims'] / dt:.0f} mô phỏng/giây, mạng chiếm {st['nn_seconds']:.1f}s tổng.")
            print(f"[bench] ước lượng ~{gph:,.0f} ván/giờ (mạng chưa học nên ván có thể dài hơn hoặc ngắn hơn sau này).")
            per_iter = args.games_per_iter / max(gph, 1e-9) * 3600
            print(f"[bench] mỗi vòng ({args.games_per_iter} ván) khoảng {per_iter / 60:.1f} phút tự chơi, "
                  f"{args.iterations} vòng khoảng {per_iter * args.iterations / 3600:.1f} giờ (chưa tính bước học).")
            return

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        new_log = not log_path.exists()
        log_fields = LOG_FIELDS
        if not new_log:  # chạy tiếp một log cũ: giữ đúng các cột sẵn có
            with open(log_path, newline="", encoding="utf-8") as f_old:
                header = next(csv.reader(f_old), None)
            if header:
                log_fields = header
        log_f = open(log_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(log_f, fieldnames=log_fields, extrasaction="ignore")
        if new_log:
            writer.writeheader()

        milestones = [float(x) for x in args.lr_milestones.split(",") if x.strip()]
        t_begin = time.time()
        it_times: List[float] = []
        last_metrics: Dict[str, float] = {}
        workers_eff = max(args.workers, 1)

        for it in range(start_it + 1, args.iterations + 1):
            it_t0 = time.time()
            lr = lr_for_iteration(args.lr, it, args.iterations, milestones)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # 1) Tự chơi
            records, st = generate(pool, model, device, args.games_per_iter, sp_cfg,
                                   args.seed * 100003 + it, workers_eff if pool else 1)
            sp_seconds = time.time() - it_t0
            added = buffer.add_records(records)
            res = {"white": 0, "black": 0, "draw": 0, "capped": 0}
            for r in records:
                res[r["result"]] += 1
            n_adj = sum(1 for r in records if r.get("adjudicated"))
            n_masked = sum(int((~r["vmask"]).sum()) for r in records)
            plies = [r["plies"] for r in records]

            # 2) Cập nhật mạng
            t1 = time.time()
            if len(buffer) >= args.min_buffer:
                last_metrics = train_steps(model, optimizer, buffer, args.train_steps, args.batch_size,
                                           device, args.clip_grad)
            train_seconds = time.time() - t1

            # 3) Lưu
            payload_new = make_payload(model, args, it, dict(last_metrics))
            best_payload = dict(payload_new)
            full = dict(payload_new)
            full["iteration"] = it
            full["optimizer_state_dict"] = optimizer.state_dict()
            _atomic_save(full, latest_path)
            _atomic_save(best_payload, best_path)
            if args.snapshot_every > 0 and it % args.snapshot_every == 0:
                _atomic_save(best_payload, ckpt_dir / "snapshots" / f"iter_{it:04d}.pt")
            if args.save_buffer_every > 0 and it % args.save_buffer_every == 0:
                buffer.save(buffer_path)

            # 4) Đánh giá nhanh với đối thủ ngẫu nhiên
            ev = {"eval_win": "", "eval_draw": "", "eval_loss": "", "eval_capped": ""}
            if args.eval_every > 0 and (it == 1 or it % args.eval_every == 0):
                ev_cfg = SearchConfig(simulations=args.eval_sims, c_puct=args.c_puct,
                                      max_plies=args.eval_max_plies, parallel_games=args.parallel_games)
                ev_recs, _ = generate(pool, model, device, args.eval_games, ev_cfg,
                                      args.seed * 7919 + it, workers_eff if pool else 1, eval_random=True)
                cnt = {"win": 0, "draw": 0, "loss": 0, "capped": 0}
                for r in ev_recs:
                    cnt[r["result"]] += 1
                ev = {"eval_win": cnt["win"], "eval_draw": cnt["draw"], "eval_loss": cnt["loss"],
                      "eval_capped": cnt["capped"]}

            elapsed = time.time() - t_begin
            it_times.append(time.time() - it_t0)
            row = {
                "iteration": it,
                "games": len(records),
                "avg_plies": round(float(np.mean(plies)), 1) if plies else 0,
                "white_wins": res["white"], "black_wins": res["black"],
                "draws": res["draw"], "capped": res["capped"],
                "samples_added": added, "buffer": len(buffer),
                "value_mask_frac": round(1 - n_masked / max(1, added), 4),
                "loss": round(last_metrics.get("loss", float("nan")), 5),
                "policy_loss": round(last_metrics.get("policy_loss", float("nan")), 5),
                "value_loss": round(last_metrics.get("value_loss", float("nan")), 6),
                "lr": lr,
                "selfplay_seconds": round(sp_seconds, 1),
                "train_seconds": round(train_seconds, 1),
                "sims_per_sec": round(st["sims"] / max(sp_seconds, 1e-9), 0),
                "elapsed_seconds": round(elapsed, 0),
                "adjudicated": n_adj,
                **ev,
            }
            writer.writerow(row)
            log_f.flush()
            remaining = (args.iterations - it) * (sum(it_times) / len(it_times))
            ev_txt = ""
            if ev["eval_win"] != "":
                ev_txt = f" | vs Random: {ev['eval_win']}W/{ev['eval_draw']}D/{ev['eval_loss']}L (+{ev['eval_capped']} chưa kết thúc)"
            print(
                f"[{it:3d}/{args.iterations}] ván {len(records)} (T {res['white']}/Đ {res['black']}/H {res['draw']}/cắt {res['capped']}) "
                f"ply TB {row['avg_plies']} | buffer {len(buffer)} | loss {row['loss']} (pol {row['policy_loss']}, "
                f"val {row['value_loss']}) | lr {lr:g} | tự chơi {sp_seconds:.0f}s, học {train_seconds:.0f}s | "
                f"còn ~{remaining / 3600:.1f}h{ev_txt}",
                flush=True,
            )
            if args.max_hours > 0 and elapsed / 3600 >= args.max_hours:
                print(f"Đã đạt giới hạn {args.max_hours}h, dừng sau vòng {it}. Chạy tiếp bằng --resume.")
                break

        buffer.save(buffer_path)
        log_f.close()
        print(f"Xong. Trọng số mới nhất: {best_path}")
    except KeyboardInterrupt:
        print("\nĐã dừng. Trạng thái vòng gần nhất đã được lưu; chạy tiếp bằng --resume.")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
