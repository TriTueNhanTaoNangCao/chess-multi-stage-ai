"""
Stage 7 (v2): MCTS cho tự chơi AlphaZero, chạy nhiều ván song song và gom suy luận theo lô.

Khác bản v1 (agents/stage7_alphazero/mcts.py):
- Không sao chép bàn cờ cho từng nút con. Mỗi ván dùng đúng một chess.Board; khi chọn nhánh
  thì push nước đi rồi pop lại, nên lịch sử nước đi luôn đầy đủ (phát hiện lặp thế đúng).
- Nút chỉ lưu mảng (P, N, W) của các nước con; nút con được tạo khi lần đầu được thăm.
- Nhiều ván cùng chạy; mỗi vòng mỗi ván đưa đúng một nút lá cần đánh giá, tất cả được gom
  thành một lô gửi cho mạng (GPU hoặc CPU). Mỗi ván chỉ có một nút lá đang chờ nên không cần
  virtual loss.
- Giữ lại cây con của nước đã đi làm cây mới (như AlphaGo Zero / AlphaZero).
- Mục tiêu học của Policy là phân bố số lượt thăm với tau = 1; tau chỉ dùng để chọn nước đi.
- Ván bị cắt vì hết giới hạn số ply không có kết quả thật, nên được đánh dấu
  value_mask = False (không huấn luyện Value với ván đó).

Quy ước dấu giống v1: giá trị của một nút tính theo góc nhìn của bên đang tới lượt tại nút đó.
Với nút cha, điểm chọn nhánh là -Q_con + U.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np

from games.chess.encoder import ChessEncoder

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_ENCODE_MOVE = ChessEncoder.encode_move
_ENCODE_BOARD = ChessEncoder.encode_board


@dataclass
class SearchConfig:
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temp_plies: int = 30      # tau = 1 trong temp_plies ply đầu, sau đó chọn nước thăm nhiều nhất
    max_plies: int = 200      # giới hạn độ dài ván (tự chơi: ván hết giới hạn không có nhãn Value)
    parallel_games: int = 32  # số ván chạy đồng thời trong một tiến trình
    # Hai tuỳ chọn KHÔNG có trong bài báo, mặc định tắt (xem HUONG_DAN_ALPHAZERO_V2.md):
    adjudicate_material: float = 0.0   # >0: ván hết giới hạn ply được chấm theo vật chất nếu chênh >= ngưỡng (đơn vị tốt)
    random_opening_plies: int = 0      # >0: mỗi ván tự chơi bắt đầu bằng 0..N nước đi ngẫu nhiên (không dùng để học)
    adjudicate_draws: bool = False     # True (cần adjudicate_material > 0): ván hòa do lặp thế / 50 nước cũng được chấm theo vật chất


class Node:
    """Nút cây tìm kiếm. N, W là thống kê của các cạnh con (W theo góc nhìn bên đi ở nút con)."""

    __slots__ = ("moves", "idxs", "P", "N", "W", "kids", "n", "terminal", "tvalue")

    def __init__(self, moves: List[chess.Move], idxs: np.ndarray, priors: np.ndarray):
        k = len(moves)
        self.moves = moves
        self.idxs = idxs
        self.P = priors
        self.N = np.zeros(k, dtype=np.float32)
        self.W = np.zeros(k, dtype=np.float32)
        self.kids: List[Optional["Node"]] = [None] * k
        self.n = 0
        self.terminal = False
        self.tvalue = 0.0

    @staticmethod
    def make_terminal(value: float) -> "Node":
        node = Node([], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
        node.terminal = True
        node.tvalue = float(value)
        return node


def encode_state(board: chess.Board) -> np.ndarray:
    """Tensor (14, 8, 8) float32 cho mạng."""
    return np.ascontiguousarray(_ENCODE_BOARD(board).transpose(2, 0, 1))


def terminal_value(board: chess.Board, legal_moves: List[chess.Move]) -> Optional[float]:
    """Giá trị kết thúc theo bên đang tới lượt; None nếu chưa kết thúc.

    Trong cây tìm kiếm coi hòa khi: hết nước (pat), thiếu quân, 50 nước (halfmove >= 100),
    lặp thế lần thứ ba.
    """
    if not legal_moves:
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 100:
        return 0.0
    if board.is_repetition(3):
        return 0.0
    return None


def legal_moves_and_indices(board: chess.Board) -> Tuple[List[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    turn = board.turn
    idxs = np.fromiter((_ENCODE_MOVE(m, turn) for m in moves), dtype=np.int64, count=len(moves))
    return moves, idxs


def _expand(moves: List[chess.Move], idxs: np.ndarray, logits_row: np.ndarray) -> Node:
    logits = logits_row[idxs].astype(np.float64)
    logits -= logits.max()
    e = np.exp(logits)
    priors = (e / e.sum()).astype(np.float32)
    return Node(moves, idxs, priors)


def _select(node: Node, c_puct: float) -> int:
    n_arr = node.N
    q = np.where(n_arr > 0, -node.W / np.maximum(n_arr, 1.0), 0.0)
    u = (c_puct * math.sqrt(max(1, node.n))) * node.P / (1.0 + n_arr)
    return int(np.argmax(q + u))


def _backup(path: List[Tuple[Node, int]], value: float) -> None:
    for node, a in reversed(path):
        node.W[a] += value
        node.N[a] += 1.0
        node.n += 1
        value = -value


class _Game:
    __slots__ = (
        "board", "root", "sims_done", "ply", "samples", "mode", "az_color",
        "phase", "pending", "record", "noise", "index",
    )

    def __init__(self, index: int, mode: str, az_color: Optional[bool], record: bool, noise: bool):
        self.board = chess.Board()
        self.root: Optional[Node] = None
        self.sims_done = 0
        self.ply = 0
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []
        self.mode = mode          # "selfplay" hoặc "eval"
        self.az_color = az_color  # chỉ dùng cho mode "eval" (đấu với bên đi ngẫu nhiên)
        self.phase = "root"       # "root": cần đánh giá nút gốc; "search": đang chạy mô phỏng
        self.pending = False
        self.record = record
        self.noise = noise
        self.index = index


def _simulate_one(g: _Game, cfg: SearchConfig):
    """Một lượt mô phỏng. Trả về None nếu đã xong (nút kết thúc), hoặc (state, ctx) nếu cần mạng."""
    node = g.root
    board = g.board
    path: List[Tuple[Node, int]] = []
    pushed = 0
    result = None
    value = 0.0
    while True:
        a = _select(node, cfg.c_puct)
        path.append((node, a))
        board.push(node.moves[a])
        pushed += 1
        child = node.kids[a]
        if child is None:
            moves, idxs = legal_moves_and_indices(board)
            tv = terminal_value(board, moves)
            if tv is not None:
                node.kids[a] = Node.make_terminal(tv)
                value = tv
            else:
                result = (encode_state(board), ("leaf", g, node, a, list(path), moves, idxs))
            break
        if child.terminal:
            value = child.tvalue
            break
        node = child
    for _ in range(pushed):
        board.pop()
    if result is None:
        _backup(path, value)
        return None
    return result


def _begin_search(g: _Game, cfg: SearchConfig, nprng: np.random.Generator) -> None:
    root = g.root
    g.phase = "search"
    g.sims_done = 0
    if len(root.moves) <= 1:
        g.sims_done = cfg.simulations  # chỉ có một nước, không cần tìm kiếm
        return
    if g.noise:
        eta = nprng.dirichlet([cfg.dirichlet_alpha] * len(root.moves))
        root.P = ((1.0 - cfg.dirichlet_epsilon) * root.P + cfg.dirichlet_epsilon * eta).astype(np.float32)


_PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_balance(board: chess.Board) -> int:
    """Chênh lệch vật chất (đơn vị tốt) theo góc nhìn quân Trắng."""
    total = 0
    for piece_type, value in _PIECE_VALUE.items():
        total += value * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    return total


# Hòa do lặp thế hoặc luật 50/75 nước: ván chưa phân thắng bại nên có thể chấm theo vật chất.
# Không áp dụng cho hết quân để chiếu hết (không bên nào thắng được) và hết nước đi (pat).
_ADJUDICABLE_DRAWS = {
    chess.Termination.THREEFOLD_REPETITION,
    chess.Termination.FIVEFOLD_REPETITION,
    chess.Termination.FIFTY_MOVES,
    chess.Termination.SEVENTYFIVE_MOVES,
}


def adjudicated_white_value(board: chess.Board, threshold: float) -> Optional[float]:
    """Giá trị chấm cho ván bị cắt, theo góc nhìn Trắng, trong [-1, 1]; None nếu chênh lệch chưa đủ ngưỡng."""
    if threshold <= 0:
        return None
    diff = material_balance(board)
    if abs(diff) < threshold:
        return None
    return float(np.sign(diff)) * min(1.0, abs(diff) / (2.0 * threshold))


def _finish(g: _Game, outcome: Optional[chess.Outcome], cfg: Optional[SearchConfig] = None) -> Dict[str, Any]:
    samples = g.samples
    t = len(samples)
    rec: Dict[str, Any] = {"plies": g.ply}
    if g.mode == "eval":
        if outcome is None:
            rec["result"] = "capped"
        elif outcome.winner is None:
            rec["result"] = "draw"
        else:
            rec["result"] = "win" if outcome.winner == g.az_color else "loss"
        rec["termination"] = outcome.termination.name if outcome is not None else "MAX_PLIES"
        return rec
    if outcome is None:
        z = np.zeros(t, dtype=np.float32)
        vmask = np.zeros(t, dtype=bool)
        rec["result"] = "capped"
        rec["termination"] = "MAX_PLIES"
        rec["adjudicated"] = False
        wv = adjudicated_white_value(g.board, cfg.adjudicate_material) if cfg is not None else None
        if wv is not None and t:
            z = np.array([wv if s[3] == chess.WHITE else -wv for s in samples], dtype=np.float32)
            vmask = np.ones(t, dtype=bool)
            rec["adjudicated"] = True
            rec["termination"] = "MAX_PLIES_MATERIAL"
    else:
        winner = outcome.winner
        z = np.array(
            [0.0 if winner is None else (1.0 if winner == s[3] else -1.0) for s in samples],
            dtype=np.float32,
        )
        vmask = np.ones(t, dtype=bool)
        rec["result"] = "draw" if winner is None else ("white" if winner else "black")
        rec["termination"] = outcome.termination.name
        rec["adjudicated"] = False
        if (winner is None and cfg is not None and cfg.adjudicate_draws
                and outcome.termination in _ADJUDICABLE_DRAWS and t):
            wv = adjudicated_white_value(g.board, cfg.adjudicate_material)
            if wv is not None:
                z = np.array([wv if s[3] == chess.WHITE else -wv for s in samples], dtype=np.float32)
                rec["adjudicated"] = True
    rec["states"] = np.stack([s[0] for s in samples]).astype(np.uint8) if t else np.zeros((0, 14, 8, 8), np.uint8)
    rec["idxs"] = [s[1] for s in samples]
    rec["probs"] = [s[2] for s in samples]
    rec["z"] = z
    rec["vmask"] = vmask
    return rec


def label_values(turns: List[bool], winner: Optional[bool]) -> List[float]:
    """Nhãn Value theo góc nhìn bên đi ở từng mẫu (tách riêng để dễ kiểm thử)."""
    return [0.0 if winner is None else (1.0 if winner == t else -1.0) for t in turns]


def _advance(g: _Game, cfg: SearchConfig, rng: random.Random, nprng: np.random.Generator):
    """Đưa ván tới trạng thái cần tìm kiếm. Trả về bản ghi nếu ván đã kết thúc, ngược lại None."""
    while True:
        outcome = g.board.outcome(claim_draw=True)
        if outcome is not None:
            return _finish(g, outcome, cfg)
        if g.ply >= cfg.max_plies:
            return _finish(g, None, cfg)
        if g.mode == "eval" and g.board.turn != g.az_color:
            move = rng.choice(list(g.board.legal_moves))
            g.board.push(move)
            g.ply += 1
            g.root = None
            continue
        break
    if g.root is None:
        g.phase = "root"
    else:
        _begin_search(g, cfg, nprng)
    return None


def _choose_and_play(g: _Game, cfg: SearchConfig, nprng: np.random.Generator) -> None:
    root = g.root
    visits = root.N.astype(np.float64)
    if visits.sum() <= 0:
        visits = np.ones(len(root.moves), dtype=np.float64)
    pi = visits / visits.sum()
    board = g.board
    if g.record:
        g.samples.append((
            encode_state(board).astype(np.uint8),
            root.idxs.astype(np.int16),
            pi.astype(np.float32),
            board.turn,
        ))
    if g.mode == "eval" or g.ply >= cfg.temp_plies:
        candidates = np.flatnonzero(visits == visits.max())
        a = int(nprng.choice(candidates))
    else:
        a = int(nprng.choice(len(pi), p=pi))
    board.push(root.moves[a])
    g.ply += 1
    child = root.kids[a]
    g.root = child if (child is not None and not child.terminal) else None


def _infer(model, device, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.stack(states)).to(device)
    with torch.inference_mode():
        logits, values = model(x)
    return logits.float().cpu().numpy(), values.reshape(-1).float().cpu().numpy()


def play_games(
    model,
    device,
    n_games: int,
    cfg: SearchConfig,
    seed: int,
    *,
    eval_random: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Chơi n_games ván, tối đa cfg.parallel_games ván đồng thời.

    - Tự chơi (mặc định): có nhiễu Dirichlet, ghi mẫu (state, pi, z, value_mask).
    - eval_random=True: mạng đấu với bên đi ngẫu nhiên, đổi màu theo chỉ số ván,
      không nhiễu, chọn nước thăm nhiều nhất, không ghi mẫu.
    """
    model.eval()
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    started = 0
    records: List[Dict[str, Any]] = []
    stats = {"sims": 0, "nn_evals": 0, "nn_batches": 0, "nn_seconds": 0.0}
    t_start = time.perf_counter()

    def new_game() -> _Game:
        nonlocal started
        idx = started
        started += 1
        if eval_random:
            return _Game(idx, "eval", chess.WHITE if idx % 2 == 0 else chess.BLACK, record=False, noise=False)
        g = _Game(idx, "selfplay", None, record=True, noise=True)
        if cfg.random_opening_plies > 0:
            k = rng.randint(0, cfg.random_opening_plies)
            for _ in range(k):
                legal = list(g.board.legal_moves)
                if not legal:
                    break
                g.board.push(rng.choice(legal))
            if g.board.outcome(claim_draw=True) is not None:
                g.board = chess.Board()  # mở đầu ngẫu nhiên đã kết thúc ván: bỏ, dùng thế ban đầu
        return g

    def settle(g: _Game) -> Optional[_Game]:
        """Đưa ván về trạng thái cần tìm kiếm. Ván đã kết thúc được ghi lại và thay bằng ván mới
        (nếu còn chỉ tiêu); trả về None nếu không còn ván nào để thay."""
        while True:
            rec = _advance(g, cfg, rng, nprng)
            if rec is None:
                return g
            records.append(rec)
            if started >= n_games:
                return None
            g = new_game()

    active: List[_Game] = []
    for _ in range(min(cfg.parallel_games, n_games)):
        g = settle(new_game())
        if g is not None:
            active.append(g)

    while active:
        states: List[np.ndarray] = []
        ctxs: List[tuple] = []
        for g in active:
            if g.phase == "root":
                moves, idxs = legal_moves_and_indices(g.board)
                states.append(encode_state(g.board))
                ctxs.append(("root", g, moves, idxs))
            else:
                while g.sims_done < cfg.simulations:
                    g.sims_done += 1
                    stats["sims"] += 1
                    res = _simulate_one(g, cfg)
                    if res is not None:
                        g.pending = True
                        states.append(res[0])
                        ctxs.append(res[1])
                        break
        if states:
            t0 = time.perf_counter()
            logits, values = _infer(model, device, states)
            stats["nn_seconds"] += time.perf_counter() - t0
            stats["nn_batches"] += 1
            stats["nn_evals"] += len(states)
            for i, ctx in enumerate(ctxs):
                if ctx[0] == "root":
                    _, g, moves, idxs = ctx
                    g.root = _expand(moves, idxs, logits[i])
                    _begin_search(g, cfg, nprng)
                else:
                    _, g, node, a, path, moves, idxs = ctx
                    node.kids[a] = _expand(moves, idxs, logits[i])
                    _backup(path, float(values[i]))
                    g.pending = False
        next_active: List[_Game] = []
        for g in active:
            if g.phase == "search" and not g.pending and g.sims_done >= cfg.simulations:
                _choose_and_play(g, cfg, nprng)
                g2 = settle(g)
                if g2 is not None:
                    next_active.append(g2)
            else:
                next_active.append(g)
        active = next_active

    stats["seconds"] = time.perf_counter() - t_start
    stats["games"] = len(records)
    return records, stats


def search_position(
    model,
    device,
    board: chess.Board,
    simulations: int,
    c_puct: float = 1.5,
    seed: int = 0,
    noise: bool = False,
) -> Node:
    """Tìm kiếm một thế cờ (dùng cho kiểm thử). Trả về nút gốc sau đủ mô phỏng."""
    model.eval()
    cfg = SearchConfig(simulations=simulations, c_puct=c_puct)
    nprng = np.random.default_rng(seed)
    g = _Game(0, "selfplay", None, record=False, noise=noise)
    g.board = board.copy(stack=True)
    moves, idxs = legal_moves_and_indices(g.board)
    logits, _ = _infer(model, device, [encode_state(g.board)])
    g.root = _expand(moves, idxs, logits[0])
    _begin_search(g, cfg, nprng)
    while g.sims_done < cfg.simulations:
        g.sims_done += 1
        res = _simulate_one(g, cfg)
        if res is not None:
            lg, vl = _infer(model, device, [res[0]])
            _, _, node, a, path, mv, ix = res[1]
            node.kids[a] = _expand(mv, ix, lg[0])
            _backup(path, float(vl[0]))
    return g.root
