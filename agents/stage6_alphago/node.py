"""
Stage 6: Cấu trúc AlphaGoNode cho MCTS kết hợp Policy Prior.
Người thực hiện: Tùng.

Quy ước kỹ thuật (docs/HANDOFF.md):
- Value của node được tính theo góc nhìn của bên đang tới lượt tại chính node đó.
- Selection Score từ góc nhìn node cha: (-Q_child) + u(s, a).
- u(s, a) = c_puct * P(s, a) * sqrt(max(1, N_parent)) / (1 + N_child).
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional
import chess
from games.chess.board import ChessBoard


class AlphaGoNode:
    """
    Nút cây MCTS cho AlphaGo.
    Lưu trữ số lượt thăm N, tổng giá trị W, giá trị trung bình Q,
    và xác suất tiên nghiệm Prior P do Policy Network cung cấp.
    """

    def __init__(
        self,
        board: ChessBoard,
        parent: Optional[AlphaGoNode] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ):
        # Đảm bảo bàn cờ được bọc đúng chuẩn ChessBoard và bảo toàn move_stack
        if not isinstance(board, ChessBoard):
            if isinstance(board, chess.Board):
                raw_copy = board.copy(stack=True)
                wrapped = ChessBoard(claim_draw=True)
                wrapped.board = raw_copy
                try:
                    wrapped._initial_fen = board.root().fen(en_passant="fen")
                except Exception:
                    wrapped._initial_fen = raw_copy.fen(en_passant="fen")
                self.board = wrapped
            else:
                raise TypeError(f"Yêu cầu ChessBoard hoặc chess.Board, nhận được {type(board)}")
        else:
            self.board = board

        self.parent: Optional[AlphaGoNode] = parent
        self.move: Optional[chess.Move] = move
        self.prior: float = float(prior)            # P(s, a): Xác suất tiên nghiệm từ Policy Net
        self.visits: int = 0                        # N(s, a): Số lần node này được thăm
        self.value_sum: float = 0.0                 # W(s, a): Tổng giá trị tích lũy
        self.children: Dict[chess.Move, AlphaGoNode] = {}
        self.is_expanded: bool = False

    @property
    def mean_value(self) -> float:
        """Q(s, a): Giá trị trung bình từ góc nhìn của bên sắp đi tại chính node này."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def is_terminal(self) -> bool:
        """Kiểm tra thế cờ tại node này đã kết thúc chưa."""
        return self.board.is_terminal()

    def selection_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """
        Tính điểm kết hợp giữa Khai thác (-Q) và Thăm dò có Prior (u)
        từ góc nhìn của node cha dẫn tới node con này.
        """
        # 1. Khai thác (Exploitation):
        # self.mean_value theo góc nhìn người sắp đi ở con (đối thủ của cha).
        # Điểm nhìn từ cha là: -self.mean_value.
        exploitation = -self.mean_value if self.visits > 0 else 0.0

        # 2. Thăm dò (Exploration với Prior):
        # Prior P định hướng nhánh triển vọng; mẫu số (1 + N) giảm dần độ ưu tiên khi đã khám phá đủ.
        exploration = c_puct * self.prior * (math.sqrt(max(1, parent_visits)) / (1 + self.visits))

        return exploitation + exploration

    def best_child(self, c_puct: float = 1.5, rng: Optional[random.Random] = None) -> AlphaGoNode:
        """Chọn node con có điểm selection_score cao nhất từ góc nhìn của node hiện tại."""
        if not self.children:
            raise ValueError("Không có node con nào để lựa chọn!")

        best_score = -float("inf")
        best_candidates: List[AlphaGoNode] = []

        for child in self.children.values():
            score = child.selection_score(self.visits, c_puct=c_puct)
            if score > best_score:
                best_score = score
                best_candidates = [child]
            elif math.isclose(score, best_score, rel_tol=1e-9, abs_tol=1e-9):
                best_candidates.append(child)

        if len(best_candidates) == 1 or rng is None:
            return best_candidates[0]
        return rng.choice(best_candidates)

    def expand(self, policy_probs: Dict[chess.Move, float]) -> None:
        """
        Mở rộng node bằng cách tạo tất cả các node con ứng với các nước đi hợp lệ
        kèm xác suất tiên nghiệm Prior tương ứng từ Policy Network.
        """
        if self.is_expanded:
            return

        for move, prob in policy_probs.items():
            child_board = self.board.clone()
            child_board.step(move)
            child_node = AlphaGoNode(
                board=child_board,
                parent=self,
                move=move,
                prior=prob,
            )
            self.children[move] = child_node

        self.is_expanded = True

    def update(self, value_for_current_player: float) -> None:
        """
        Cập nhật thống kê khi có một lượt mô phỏng duyệt qua node này.
        Args:
            value_for_current_player: Điểm số tính theo góc nhìn của người đang tới lượt ở node này.
        """
        self.visits += 1
        self.value_sum += value_for_current_player

