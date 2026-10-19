"""
Stage 3: Agent Pure Monte Carlo Tree Search (MCTS) cho Cờ Vua.
Người thực hiện: Thi.

Quy trình 4 giai đoạn chuẩn của MCTS:
1. Selection (Lựa chọn): Duyệt cây từ gốc xuống theo công thức UCB1 đến khi gặp node chưa mở rộng hết hoặc kết thúc.
2. Expansion (Mở rộng): Rút một nước đi hợp lệ chưa thử và tạo node con mới gắn vào cây.
3. Rollout / Playout (Mô phỏng ngẫu nhiên): Đi ngẫu nhiên đến khi kết thúc ván hoặc chạm độ sâu tối đa (max_rollout_depth).
4. Backpropagation (Lan truyền ngược): Truyền kết quả mô phỏng ngược về gốc, cập nhật số lượt thăm và tổng giá trị.

Hợp đồng và quy ước kỹ thuật (docs/HANDOFF.md):
- Quy ước dấu giá trị: Giá trị lưu tại mỗi node tính theo góc nhìn của người đang tới lượt đi tại chính node đó.
- Điểm mục tiêu: white_score * current_player_at_state (+1.0 Thắng, -1.0 Thua, 0.0 Hòa, delta/(|delta|+10)).
- Node cha đánh giá node con bằng (-child.mean_value) + thành phần thăm dò UCB1.
- Bảo toàn lịch sử nước đi: ChessBoard.clone() giữ trọn move_stack và chính sách claim_draw.
- Tính tất định theo seed: Dùng self.rng cho mọi quyết định ngẫu nhiên để tái lập kết quả.
"""
from __future__ import annotations
import math
import random
from typing import Optional
import chess

from agents.base import BaseAgent
from games.chess.board import ChessBoard
from .node import MCTSNode

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


class PureMCTSAgent(BaseAgent):
    """
    Agent Pure MCTS chơi cờ vua nguyên bản, không sử dụng mạng nơ-ron hay tri thức heuristic tĩnh.
    Tự động cân bằng giữa Khai thác (Exploitation) và Thăm dò (Exploration) qua UCB1.
    """

    def __init__(
        self,
        simulations: int = 200,
        c: float = 1.414,
        max_rollout_depth: int = 30,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        player_id: Optional[int] = None,
    ):
        name = name or f"PureMCTS ({simulations})"
        super().__init__(name=name, player_id=player_id)
        if simulations < 1:
            raise ValueError("simulations phải >= 1")
        if c < 0:
            raise ValueError("Hằng số thăm dò c phải >= 0")
        if max_rollout_depth < 0:
            raise ValueError("max_rollout_depth phải >= 0")

        self.simulations = simulations              # Tổng số lượt mô phỏng cho một nước đi
        self.c = c                                  # Hằng số cân bằng thăm dò UCB1 (mặc định sqrt(2) ~ 1.414)
        self.max_rollout_depth = max_rollout_depth  # Độ sâu tối đa trong mỗi lượt rollout ngẫu nhiên
        self.seed = seed                            # Seed ngẫu nhiên để tái lập kết quả
        self.rng = random.Random(seed)
        self.root: Optional[MCTSNode] = None        # Lưu trữ gốc cây tìm kiếm sau mỗi lần act() để dễ kiểm tra

    def reset(self) -> None:
        """Đặt lại trạng thái tìm kiếm và khởi tạo lại bộ sinh số ngẫu nhiên theo seed."""
        self.rng = random.Random(self.seed)
        self.root = None

    def act(self, board: ChessBoard) -> chess.Move:
        """
        Quyết định và trả về nước đi chess.Move tối ưu bằng thuật toán UCB1 MCTS.
        Chấp nhận cả ChessBoard wrapper và raw chess.Board, bảo toàn 100% lịch sử nước đi.
        Tuyệt đối không làm thay đổi bàn cờ hay lịch sử nước đi của bên gọi.
        Ném ValueError nếu bàn cờ đã ở trạng thái kết thúc.
        """
        # Nếu người gọi truyền vào raw chess.Board, bọc lại và bảo toàn trọn vẹn move_stack
        if isinstance(board, chess.Board) and not isinstance(board, ChessBoard):
            raw_copy = board.copy(stack=True)
            wrapped = ChessBoard(claim_draw=True)
            wrapped.board = raw_copy
            try:
                wrapped._initial_fen = board.root().fen(en_passant="fen")
            except Exception:
                wrapped._initial_fen = raw_copy.fen(en_passant="fen")
            board = wrapped

        if board.is_terminal():
            raise ValueError("Không thể ra quyết định trên bàn cờ đã kết thúc")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            raise ValueError("Không có nước đi hợp lệ nào trên bàn cờ")

        # Xử lý nhanh: Nếu chỉ có duy nhất 1 nước đi hợp lệ, đi ngay mà không cần mô phỏng
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Khởi tạo node gốc bằng bản sao sâu giữ nguyên lịch sử ván cờ (stack=True)
        root = MCTSNode(board=board.clone())
        self.root = root

        for _ in range(self.simulations):
            # 1. Selection (Lựa chọn)
            node = self._select(root)

            # 2. Expansion (Mở rộng)
            if not node.is_terminal() and node.untried_moves:
                node = self._expand(node)

            # 3. Rollout (Mô phỏng ngẫu nhiên)
            white_score = self._rollout(node)

            # 4. Backpropagation (Lan truyền ngược)
            self._backpropagate(node, white_score)

        # Chọn nước đi theo chiến lược Robust Child: nhánh có số lượt viếng thăm (visits) cao nhất.
        # Nếu hòa visits, phân định bằng giá trị khai thác trung bình (-mean_value).
        best_move = max(
            root.children.items(),
            key=lambda item: (item[1].visits, -item[1].mean_value)
        )[0]

        return best_move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Duyệt dọc theo cây đến khi tìm được một node chưa mở rộng hết hoặc thế cờ kết thúc."""
        while node.is_fully_expanded() and not node.is_terminal():
            if not node.children:
                break
            node = node.best_child(c=self.c, rng=self.rng)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Rút một nước đi chưa thử từ node hiện tại, tạo node con mới và gắn vào cây."""
        move_idx = self.rng.randrange(len(node.untried_moves))
        move = node.untried_moves.pop(move_idx)

        # Clone bàn cờ giữ nguyên toàn bộ lịch sử (stack=True) cho node con
        child_board = node.board.clone()
        child_board.step(move)

        child = node.add_child(move, child_board)
        return child

    def _rollout(self, node: MCTSNode) -> float:
        """
        Mô phỏng ván cờ bằng các nước đi ngẫu nhiên đến khi kết thúc hoặc chạm trần max_rollout_depth.
        Giá trị trả về (white_score) theo góc nhìn của quân Trắng:
            +1.0: Trắng thắng tuyệt đối
            -1.0: Đen thắng tuyệt đối
             0.0: Hòa cờ theo luật
             delta / (|delta| + 10): Điểm chuẩn hóa chênh lệch quân sau 30 nước chưa kết thúc
        """
        if node.is_terminal():
            winner = node.board.check_winner()
            return float(winner) if winner is not None else 0.0

        # Sử dụng copy(stack=True) để bảo toàn toàn bộ move_stack từ đầu ván cờ,
        # đảm bảo nhận diện chính xác hòa do lặp lại vị trí (threefold repetition)
        # và luật 50 nước kết hợp với lịch sử trước đó theo đúng chính sách claim_draw.
        sim_board = node.board.board.copy(stack=True)
        claim_draw = node.board.claim_draw
        depth = 0

        while depth < self.max_rollout_depth:
            outcome = sim_board.outcome(claim_draw=claim_draw)
            if outcome is not None:
                if outcome.winner == chess.WHITE:
                    return 1.0
                elif outcome.winner == chess.BLACK:
                    return -1.0
                return 0.0

            moves = list(sim_board.legal_moves)
            if not moves:
                break
            move = self.rng.choice(moves)
            sim_board.push(move)
            depth += 1

        # Kiểm tra kết quả nếu ván cờ kết thúc do chiếu hết hoặc hòa theo luật
        outcome = sim_board.outcome(claim_draw=claim_draw)
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                return 1.0
            elif outcome.winner == chess.BLACK:
                return -1.0
            return 0.0

        # Sau 30 nước đi chưa kết thúc ván:
        # Tính chênh lệch điểm quân cờ: delta = material_white - material_black
        # Chuẩn hóa điểm trả về theo công thức: delta / (|delta| + 10)
        delta = self._evaluate_material_delta(sim_board)
        return delta / (abs(delta) + 10.0)

    @staticmethod
    def _evaluate_material_delta(board: chess.Board) -> float:
        """
        Tính chênh lệch điểm số quân cờ trên bàn giữa hai bên:
            delta = material_white - material_black
        Bảng giá trị quy ước:
            Tốt (Pawn): 1, Mã (Knight): 3, Tượng (Bishop): 3,
            Xe (Rook): 5, Hậu (Queen): 9, Vua (King): không tính điểm vật chất.
        """
        white_val = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in PIECE_VALUES.items())
        black_val = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in PIECE_VALUES.items())
        return float(white_val - black_val)

    def _backpropagate(self, leaf_node: MCTSNode, white_score: float | None = None, **kwargs) -> None:
        """
        Lan truyền kết quả mô phỏng từ leaf_node ngược lên tận root.
        Cập nhật visits và value_sum ở từng node theo góc nhìn của người đang tới lượt tại node đó:
            value_target = white_score * current_player_at_state
            - Trắng đi: Điểm số theo góc nhìn Trắng (white_score * (+1)).
            - Đen đi:   Điểm số theo góc nhìn Đen (white_score * (-1)).
        """
        if white_score is None:
            white_score = kwargs.get("winner_white", 0.0)

        curr: Optional[MCTSNode] = leaf_node
        while curr is not None:
            curr.visits += 1
            current_player = curr.board.current_player  # 1 nếu Trắng đi, -1 nếu Đen đi
            curr.value_sum += white_score * current_player
            curr = curr.parent
