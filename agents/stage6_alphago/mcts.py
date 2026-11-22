"""
Stage 6: Động cơ tìm kiếm AlphaGoMCTS (Search Engine).
Người thực hiện: Tùng.

Nguyên lý hoạt động theo Silver et al. (Nature 2016):
1. Selection: Duyệt từ root xuống lá theo công thức: Score = -Q_child + u(s, a).
2. Expansion: Dùng Policy Network cung cấp xác suất tiên nghiệm P(s, a) cho tất cả nước đi hợp lệ.
3. Hybrid Leaf Evaluation: Kết hợp giữa Value Network và Rollout:
   V_leaf = (1 - lambda) * V_net + lambda * z_rollout.
4. Backpropagation: Lan truyền giá trị ngược lên root, đổi dấu qua mỗi tầng cha - con.
"""
from __future__ import annotations
import math
import random
from typing import Callable, Dict, List, Optional, Tuple
import chess

from games.chess.board import ChessBoard
from .node import AlphaGoNode

# Bảng giá trị vật chất quy ước chuẩn cho rollout nhanh
MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


class AlphaGoMCTS:
    """
    Động cơ tìm kiếm Monte Carlo Tree Search cho AlphaGo.
    Kết hợp Policy Network để định hướng Prior P và Value Network để lượng giá lá.
    """

    def __init__(
        self,
        policy_fn: Callable[[ChessBoard], Dict[chess.Move, float]],
        value_fn: Optional[Callable[[ChessBoard], float]] = None,
        simulations: int = 100,
        c_puct: float = 1.5,
        rollout_weight: float = 0.5,
        max_rollout_depth: int = 15,
        seed: Optional[int] = None,
    ):
        """
        Args:
            policy_fn: Hàm nhận ChessBoard, trả về Dict[chess.Move, float] (xác suất hợp lệ đã chuẩn hóa).
            value_fn: Hàm nhận ChessBoard, trả về scalar v trong [-1.0, 1.0] (góc nhìn bên đang đi).
            simulations: Số lượt mô phỏng cho một lần tìm kiếm nước đi.
            c_puct: Hằng số điều hòa thăm dò PUCT (mặc định 1.5).
            rollout_weight: Trọng số lambda của rollout (0.0: chỉ dùng Value Net; 1.0: chỉ dùng Rollout).
            max_rollout_depth: Số nước đi ngẫu nhiên tối đa khi rollout.
            seed: Seed sinh số ngẫu nhiên để tái lập kết quả.
        """
        if simulations < 1:
            raise ValueError("simulations phải >= 1")
        if not (0.0 <= rollout_weight <= 1.0):
            raise ValueError("rollout_weight (lambda) phải nằm trong đoạn [0.0, 1.0]")

        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.simulations = simulations
        self.c_puct = c_puct
        self.rollout_weight = rollout_weight
        self.max_rollout_depth = max_rollout_depth
        self.seed = seed
        self.rng = random.Random(seed)
        self.root: Optional[AlphaGoNode] = None

    def reset(self) -> None:
        """Đặt lại trạng thái động cơ tìm kiếm và seed."""
        self.rng = random.Random(self.seed)
        self.root = None

    def search(self, board: ChessBoard) -> chess.Move:
        """
        Thực hiện MCTS và trả về nước đi tối ưu (Robust Child: số lượt viếng thăm N cao nhất).
        Không làm thay đổi bàn cờ của bên gọi.
        """
        if board.is_terminal():
            raise ValueError("Không thể tìm kiếm trên bàn cờ đã kết thúc")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            raise ValueError("Không có nước đi hợp lệ nào trên bàn cờ")

        # Nếu chỉ có 1 nước đi hợp lệ duy nhất, đi ngay không cần mô phỏng
        if len(legal_moves) == 1:
            return legal_moves[0]

        # 1. Khởi tạo root node với bản sao bàn cờ bảo toàn trọn vẹn move_stack
        root = AlphaGoNode(board=board.clone())
        self.root = root

        # Mở rộng root ngay từ đầu bằng Policy Prior
        root_policy = self.policy_fn(root.board)
        root.expand(root_policy)

        # 2. Vòng lặp mô phỏng simulations lượt
        for _ in range(self.simulations):
            self._simulate_one_path(root)

        # 3. Chọn nước đi tốt nhất theo Robust Child (số lượt thăm N cao nhất)
        # Nếu hòa N, phân định bằng giá trị khai thác trung bình (-mean_value của con)
        best_move = max(
            root.children.items(),
            key=lambda item: (item[1].visits, -item[1].mean_value)
        )[0]

        return best_move

    def get_action_probs(self, board: ChessBoard, temperature: float = 1.0) -> Dict[chess.Move, float]:
        """
        Trả về phân phối xác suất nước đi pi(a|s) dựa trên số lượt viếng thăm N(s, a)^(1/tau).
        Dùng cho phân tích hoặc tự chơi (Self-play).
        """
        if self.root is None or self.root.board.fen() != board.fen():
            self.search(board)

        assert self.root is not None
        visits_dict = {move: child.visits for move, child in self.root.children.items()}
        total_visits = sum(visits_dict.values())

        if total_visits == 0:
            # Nếu chưa thăm, chia đều theo số nước hợp lệ
            n = len(visits_dict)
            return {m: 1.0 / n for m in visits_dict}

        if temperature <= 1e-4:
            # Nhiệt độ cực nhỏ -> tương đương argmax (one-hot)
            best_m = max(visits_dict.items(), key=lambda it: it[1])[0]
            return {m: 1.0 if m == best_m else 0.0 for m in visits_dict}

        # pi(a|s) ~ N(s, a) ^ (1 / tau)
        powered = {m: (v ** (1.0 / temperature)) for m, v in visits_dict.items()}
        sum_powered = sum(powered.values())
        return {m: p / sum_powered for m, p in powered.items()}

    def _simulate_one_path(self, root: AlphaGoNode) -> None:
        """Thực hiện một đường đi trọn vẹn: Selection -> Expansion -> Evaluation -> Backprop."""
        # --- BƯỚC 1: SELECTION ---
        node = root
        search_path: List[AlphaGoNode] = [node]

        while node.is_expanded and not node.is_terminal():
            if not node.children:
                break
            node = node.best_child(c_puct=self.c_puct, rng=self.rng)
            search_path.append(node)

        # --- BƯỚC 2: EXPANSION ---
        if not node.is_terminal() and not node.is_expanded:
            policy_probs = self.policy_fn(node.board)
            node.expand(policy_probs)

        # --- BƯỚC 3: HYBRID LEAF EVALUATION ---
        # Điểm số leaf_value được tính theo góc nhìn của người đang tới lượt ở node lá
        leaf_value = self._evaluate_leaf(node)

        # --- BƯỚC 4: BACKPROPAGATION ---
        # Lan truyền ngược từ node lá lên root.
        # Lưu ý quy ước dấu (docs/HANDOFF.md):
        # Đổi dấu giá trị qua mỗi cạnh cha - con vì 2 người chơi đối kháng nhau!
        curr_value = leaf_value
        for curr_node in reversed(search_path):
            curr_node.update(curr_value)
            curr_value = -curr_value

    def _evaluate_leaf(self, node: AlphaGoNode) -> float:
        """
        Lượng giá nút lá theo công thức kết hợp của Silver et al. (2016):
        V_leaf = (1 - lambda) * V_net + lambda * z_rollout
        Giá trị trả về theo góc nhìn của bên đang tới lượt ở node lá này.
        """
        # Nếu thế cờ đã kết thúc (chiếu hết hoặc hòa)
        if node.is_terminal():
            # get_result_relative() trả về: -1.0 nếu bị chiếu hết, 0.0 nếu hòa, +1.0 nếu thắng
            winner_relative = node.board.get_result_relative()
            return float(winner_relative) if winner_relative is not None else 0.0

        # 1. Đánh giá từ Value Network (nếu có và trọng số 1 - lambda > 0)
        v_net = 0.0
        has_vnet = (self.value_fn is not None) and (self.rollout_weight < 1.0)
        if has_vnet:
            assert self.value_fn is not None
            v_net = float(self.value_fn(node.board))

        # 2. Đánh giá từ Rollout (nếu trọng số lambda > 0)
        z_rollout = 0.0
        if self.rollout_weight > 0.0:
            z_rollout = self._fast_rollout(node.board)

        # 3. Kết hợp tuyến tính theo trọng số rollout_weight (lambda)
        if not has_vnet:
            return z_rollout
        if self.rollout_weight == 0.0:
            return v_net

        return (1.0 - self.rollout_weight) * v_net + self.rollout_weight * z_rollout

    def _fast_rollout(self, board: ChessBoard) -> float:
        """
        Mô phỏng ngẫu nhiên nhanh (Playout) từ thế cờ hiện tại tối đa max_rollout_depth bước.
        Trả về kết quả theo góc nhìn của bên đang tới lượt ở bàn cờ gốc (board).
        """
        sim_board = board.board.copy(stack=True)
        claim_draw = board.claim_draw
        original_player = board.current_player  # +1 (Trắng), -1 (Đen)
        depth = 0

        while depth < self.max_rollout_depth:
            outcome = sim_board.outcome(claim_draw=claim_draw)
            if outcome is not None:
                winner = 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
                return winner * original_player

            legal_moves = list(sim_board.legal_moves)
            if not legal_moves:
                break
            move = self.rng.choice(legal_moves)
            sim_board.push(move)
            depth += 1

        # Nếu hết số bước mô phỏng mà chưa xong ván, tính chênh lệch điểm quân cờ
        outcome = sim_board.outcome(claim_draw=claim_draw)
        if outcome is not None:
            winner = 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
            return winner * original_player

        # Chênh lệch vật chất: delta = White - Black
        white_val = sum(len(sim_board.pieces(pt, chess.WHITE)) * val for pt, val in MATERIAL_VALUES.items())
        black_val = sum(len(sim_board.pieces(pt, chess.BLACK)) * val for pt, val in MATERIAL_VALUES.items())
        delta = float(white_val - black_val)
        normalized_white_score = delta / (abs(delta) + 10.0)

        # Trả về theo góc nhìn của người chơi tại nút lá ban đầu
        return normalized_white_score * original_player

