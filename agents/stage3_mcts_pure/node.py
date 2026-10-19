"""
Thi: Cấu trúc Node cho thuật toán Pure Monte Carlo Tree Search (UCB1).
Quy ước giá trị (value): Giá trị của node được tính theo góc nhìn của người đang tới lượt đi tại chính node đó.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional
import chess
from games.chess.board import ChessBoard


@dataclass
class MCTSNode:
    board: ChessBoard                                       # Trạng thái bàn cờ tại node này (giữ nguyên lịch sử nước đi)
    parent: Optional[MCTSNode] = None                      # Node cha dẫn đến node này (None nếu là root)
    move: Optional[chess.Move] = None                      # Nước đi từ node cha dẫn tới trạng thái này
    visits: int = 0                                         # Số lần node này được viếng thăm qua các lượt mô phỏng
    value_sum: float = 0.0                                 # Tổng giá trị tích lũy (theo góc nhìn của bên đang tới lượt ở node này)
    children: Dict[chess.Move, MCTSNode] = field(default_factory=dict)  # Tập hợp các node con đã mở rộng
    untried_moves: Optional[List[chess.Move]] = None       # Danh sách các nước đi hợp lệ chưa được thử nghiệm

    def __post_init__(self):
        # Đảm bảo bàn cờ được bọc đúng chuẩn ChessBoard và bảo toàn trọn vẹn move_stack
        if not isinstance(self.board, ChessBoard):
            if isinstance(self.board, chess.Board):
                raw_copy = self.board.copy(stack=True)
                wrapped = ChessBoard(claim_draw=True)
                wrapped.board = raw_copy
                try:
                    wrapped._initial_fen = self.board.root().fen(en_passant="fen")
                except Exception:
                    wrapped._initial_fen = raw_copy.fen(en_passant="fen")
                self.board = wrapped
            else:
                raise TypeError(f"Yêu cầu ChessBoard hoặc chess.Board, nhận được {type(self.board)}")
        
        # Khởi tạo danh sách các nước đi hợp lệ chưa được mở rộng
        if self.untried_moves is None:
            if not self.board.is_terminal():
                self.untried_moves = list(self.board.get_legal_actions())
            else:
                self.untried_moves = []

    @property
    def mean_value(self) -> float:
        """Giá trị trung bình Q (tỷ lệ thắng) từ góc nhìn của bên đang tới lượt đi tại node này."""
        return self.value_sum / self.visits if self.visits else 0.0

    def is_fully_expanded(self) -> bool:
        """Kiểm tra node đã mở rộng hết chưa: đúng khi tất cả nước đi hợp lệ đều đã tạo node con."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Kiểm tra thế cờ tại node này đã kết thúc chưa (chiếu hết, hòa cờ theo luật)."""
        return self.board.is_terminal()

    def ucb1_score(self, parent_visits: int, c: float = 1.414) -> float:
        """
        Tính điểm UCB1 để node cha lựa chọn node con này.

        Lưu ý quan trọng về góc nhìn (HANDOFF.md):
            - self.mean_value được tính theo góc nhìn của người sắp đi ở CHÍNH node con này.
            - Người ở node cha là đối thủ (người vừa thực hiện nước đi dẫn tới đây).
            - Do đó, thành phần khai thác (exploitation) từ góc nhìn của node cha là: -self.mean_value.
        """
        if self.visits == 0:
            # Chưa từng được thăm: ưu tiên vô hạn để đảm bảo được thử ít nhất một lần
            return float("inf")
        
        # Thành phần khai thác: điểm của node cha = đối nghịch điểm của đối thủ tại con
        exploitation = -self.mean_value
        
        # Thành phần thăm dò: thúc đẩy khám phá các nhánh ít được thử nghiệm
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation + exploration

    def best_child(self, c: float = 1.414, rng: Optional[random.Random] = None) -> MCTSNode:
        """Chọn node con có điểm UCB1 cao nhất từ góc nhìn của node hiện tại."""
        if not self.children:
            raise ValueError("Không có node con nào để lựa chọn!")

        best_score = -float("inf")
        best_candidates: List[MCTSNode] = []

        for child in self.children.values():
            score = child.ucb1_score(self.visits, c)
            if score > best_score:
                best_score = score
                best_candidates = [child]
            elif score == best_score:
                best_candidates.append(child)

        # Nếu hòa điểm (ví dụ nhiều node có inf), dùng rng để chọn ngẫu nhiên đồng đều và tái lập được
        if len(best_candidates) == 1 or rng is None:
            return best_candidates[0]
        return rng.choice(best_candidates)

    def add_child(self, move: chess.Move, child_board: ChessBoard) -> MCTSNode:
        """Thêm và trả về một node con mới tương ứng với nước đi vừa thực hiện."""
        child = MCTSNode(
            board=child_board,
            parent=self,
            move=move,
        )
        self.children[move] = child
        return child
