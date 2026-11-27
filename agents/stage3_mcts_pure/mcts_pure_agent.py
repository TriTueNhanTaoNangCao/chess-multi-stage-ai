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
from .heuristic import evaluate_cutoff_score

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
        # 1. Validation cho simulations: chỉ nhận int, không nhận bool, giá trị >= 1
        if isinstance(simulations, bool) or not isinstance(simulations, int):
            raise TypeError("simulations phải là số nguyên (int), không chấp nhận bool hoặc kiểu khác")
        if simulations < 1:
            raise ValueError("simulations phải có giá trị >= 1")

        # 2. Validation cho max_rollout_depth: chỉ nhận int, không nhận bool, giá trị >= 0
        if isinstance(max_rollout_depth, bool) or not isinstance(max_rollout_depth, int):
            raise TypeError("max_rollout_depth phải là số nguyên (int), không chấp nhận bool hoặc kiểu khác")
        if max_rollout_depth < 0:
            raise ValueError("max_rollout_depth phải có giá trị >= 0")

        # 3. Validation cho c: nhận int/float, không nhận bool, phải hữu hạn và >= 0
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise TypeError("c phải là số thực hoặc số nguyên (float hoặc int), không chấp nhận bool hoặc kiểu khác")
        if not math.isfinite(c):
            raise ValueError("c phải là số hữu hạn (không chấp nhận NaN hoặc vô cực)")
        if c < 0:
            raise ValueError("c phải có giá trị >= 0")

        name = name or f"PureMCTS ({simulations})"
        super().__init__(name=name, player_id=player_id)

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
            self.root = None
            raise ValueError("Không thể ra quyết định trên bàn cờ đã kết thúc")

        legal_moves = board.get_legal_actions()
        if not legal_moves:
            self.root = None
            raise ValueError("Không có nước đi hợp lệ nào trên bàn cờ")

        # Xử lý nhanh: Nếu chỉ có duy nhất 1 nước đi hợp lệ, đi ngay mà không cần mô phỏng.
        # Xóa self.root để không giữ cây và thống kê của lượt trước.
        if len(legal_moves) == 1:
            self.root = None
            return legal_moves[0]

        # Kiểm tra và tái sử dụng cây tìm kiếm nếu tìm thấy node khớp với bàn cờ hiện tại
        reusable_root = self._find_reusable_node(board)
        if reusable_root is not None:
            # Tái sử dụng node: đưa node đó thành self.root
            if reusable_root is not self.root:
                # Ngắt liên kết với tổ tiên cũ và dọn dẹp các nhánh anh em ngoài subtree được giữ
                curr = reusable_root.parent
                reusable_root.parent = None
                reusable_root.move = None
                while curr is not None:
                    next_parent = curr.parent
                    curr.children.clear()
                    curr.parent = None
                    curr = next_parent
            self.root = reusable_root
            root = self.root
        else:
            # Không tìm thấy trạng thái tương thích: khởi tạo cây mới
            self.root = MCTSNode(board=board.clone())
            root = self.root

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

    @staticmethod
    def _get_board_state_key(board: ChessBoard) -> tuple:
        """
        Tạo khóa trạng thái đặc trưng duy nhất cho bàn cờ:
            (initial_fen, move_stack_tuple, claim_draw)
        Khóa này bảo toàn đầy đủ:
        - FEN ban đầu (quyền nhập thành, en passant gốc).
        - Toàn bộ chuỗi move_stack (thứ tự nước đi, luật 50 nước, lịch sử lặp lại 3 lần).
        - Chính sách xin hòa claim_draw.
        """
        initial_fen = getattr(board, "_initial_fen", None)
        if initial_fen is None:
            try:
                initial_fen = board.board.root().fen(en_passant="fen")
            except Exception:
                initial_fen = board.board.fen(en_passant="fen")
        moves = tuple(board.board.move_stack)
        claim_draw = getattr(board, "claim_draw", True)
        return (initial_fen, moves, claim_draw)

    def act_timed(self, board, limits):
        from agents.search_limits import rollout_position

        limits.check()
        moves = board.get_legal_actions()
        if len(moves) == 1:
            limits.publish(moves[0], simulations_completed=0)
            return moves[0]
        reusable = self._find_reusable_node(board)
        if reusable is not None:
            ancestor = reusable.parent
            reusable.parent, reusable.move = None, None
            while ancestor is not None:
                parent = ancestor.parent
                ancestor.children.clear()
                ancestor.parent = None
                ancestor = parent
            root = reusable
        else:
            root = MCTSNode(board=board.clone())
        self.root = root
        if root.children and any(c.visits for c in root.children.values()):
            best = max(root.children, key=lambda m: (root.children[m].visits, -root.children[m].mean_value))
            limits.publish(best, simulations_completed=0, root_visits=root.visits)
        for completed in range(1, self.simulations + 1):
            limits.check()
            node = root
            while node.is_fully_expanded() and not node.is_terminal() and node.children:
                limits.check()
                node = node.best_child(c=self.c, rng=self.rng)
            if not node.is_terminal() and node.untried_moves:
                node = self._expand(node)
            position = rollout_position(node.board, self.max_rollout_depth, self.rng, limits)
            outcome = position.outcome(claim_draw=board.claim_draw)
            white_score = ((0.0 if outcome.winner is None else (1.0 if outcome.winner else -1.0))
                           if outcome else evaluate_cutoff_score(position))
            limits.check()
            self._backpropagate(node, white_score)
            best = max(root.children, key=lambda m: (root.children[m].visits, -root.children[m].mean_value))
            limits.publish(best, simulations_completed=completed, root_visits=root.visits)
        return limits.best_move


    def _find_reusable_node(self, board: ChessBoard) -> Optional[MCTSNode]:
        """
        Tìm kiếm node trong cây hiện tại (self.root và các hậu duệ) khớp hoàn toàn với board.
        Hỗ trợ:
        - Cùng trạng thái (0 nước đi).
        - Sau 1 nước đi.
        - Sau 2 nước đi thật (AI đi rồi đối thủ đi).
        Nếu không tìm thấy node tương thích, trả về None để agent dựng cây mới.
        """
        if self.root is None:
            return None

        target_key = self._get_board_state_key(board)

        # 1. Kiểm tra chính gốc hiện tại (cùng trạng thái)
        if self._get_board_state_key(self.root.board) == target_key:
            return self.root

        # 2. Kiểm tra nhanh theo tiền tố chuỗi nước đi (move_stack)
        root_key = self._get_board_state_key(self.root.board)
        if root_key[0] == target_key[0] and root_key[2] == target_key[2]:
            root_moves = root_key[1]
            target_moves = target_key[1]
            if len(target_moves) > len(root_moves) and target_moves[:len(root_moves)] == root_moves:
                delta_moves = target_moves[len(root_moves):]
                curr: Optional[MCTSNode] = self.root
                for m in delta_moves:
                    if curr is not None and m in curr.children:
                        curr = curr.children[m]
                    else:
                        curr = None
                        break
                if curr is not None and self._get_board_state_key(curr.board) == target_key:
                    return curr

        # 3. Quét duyệt tìm kiếm theo độ sâu hậu duệ (độ sâu 1 và 2) để đảm bảo không bỏ sót
        for child in self.root.children.values():
            if self._get_board_state_key(child.board) == target_key:
                return child

        for child in self.root.children.values():
            for grandchild in child.children.values():
                if self._get_board_state_key(grandchild.board) == target_key:
                    return grandchild

        return None

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

        # Sau 30 nước đi chưa kết thúc ván (cutoff):
        # Đánh giá thế cờ đa thành phần: Vật chất, PST, Trung tâm, Phát triển, Tốt, Vua, Cơ động
        # Chuẩn hóa về (-1.0, 1.0) qua evaluate_cutoff_score
        return evaluate_cutoff_score(sim_board)

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
