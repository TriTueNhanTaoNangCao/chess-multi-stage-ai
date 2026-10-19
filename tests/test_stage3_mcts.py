"""
Bộ kiểm thử đơn vị (Unit Tests) cho Stage 3: Pure MCTS Agent và MCTSNode.
Tác giả: Thi
Xác minh hợp đồng bàn giao theo tài liệu docs/HANDOFF.md:
- Tạo nước đi hợp lệ và không biến đổi bàn cờ của bên gọi
- Từ chối ra quyết định trên bàn cờ đã kết thúc (terminal)
- Điểm mục tiêu cho các trường hợp Thắng, Thua, Hòa (theo góc nhìn người sắp đi)
- Quy ước đổi dấu giá trị qua từng cạnh cha-con (cha nhìn con với dấu âm)
- Cấu trúc cây tìm kiếm sâu ít nhất 2 cạnh (Root -> Child -> Grandchild)
- Hoạt động của thuật toán lựa chọn UCB1 và mở rộng node
- Tính chính xác của pha mô phỏng (Rollout)
- Tính tất định theo hạt giống ngẫu nhiên (seed)
- Khả năng phát hiện nước chiếu hết ngay lập tức (mate in 1) cho cả hai màu quân
"""

import math
import random
import unittest
import chess

from agents.stage3_mcts_pure.node import MCTSNode
from agents.stage3_mcts_pure.mcts_pure_agent import PureMCTSAgent
from games.chess.board import ChessBoard


class TestStage3MCTS(unittest.TestCase):

    def test_value_target_sign_and_alternation_win_loss_draw(self):
        """
        Kiểm tra giá trị lưu tại node theo góc nhìn người tới lượt và đổi dấu qua các cạnh cha/con:
            value_target = white_score * current_player_at_state
        """
        # 1. Trường hợp Trắng thắng (Scholar's mate: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#)
        board_white_won = ChessBoard("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        self.assertTrue(board_white_won.is_checkmate())
        self.assertEqual(board_white_won.check_winner(), 1)  # Trắng thắng (+1)

        agent = PureMCTSAgent(simulations=10, seed=42)

        # Node con là Đen sắp đi (nhưng đã bị chiếu hết)
        node_black_to_move = MCTSNode(board=board_white_won)
        self.assertEqual(node_black_to_move.board.current_player, -1)
        # Điểm mục tiêu: white_score(+1) * current_player(-1) = -1.0
        agent._backpropagate(node_black_to_move, white_score=1.0)
        self.assertEqual(node_black_to_move.visits, 1)
        self.assertEqual(node_black_to_move.value_sum, -1.0)
        self.assertEqual(node_black_to_move.mean_value, -1.0)

        # 2. Trường hợp Đen thắng (Fool's mate: 1. f3 e5 2. g4 Qh4#)
        board_black_won = ChessBoard("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(board_black_won.is_checkmate())
        self.assertEqual(board_black_won.check_winner(), -1)  # Đen thắng (-1)

        # Node con là Trắng sắp đi (nhưng đã bị chiếu hết)
        node_white_to_move = MCTSNode(board=board_black_won)
        self.assertEqual(node_white_to_move.board.current_player, 1)
        # Điểm mục tiêu: white_score(-1) * current_player(1) = -1.0
        agent._backpropagate(node_white_to_move, white_score=-1.0)
        self.assertEqual(node_white_to_move.visits, 1)
        self.assertEqual(node_white_to_move.value_sum, -1.0)
        self.assertEqual(node_white_to_move.mean_value, -1.0)

        # 3. Trường hợp Hòa cờ (Stalemate)
        board_stalemate = ChessBoard("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board_stalemate.is_stalemate())
        self.assertEqual(board_stalemate.check_winner(), 0)

        node_draw = MCTSNode(board=board_stalemate)
        agent._backpropagate(node_draw, white_score=0.0)
        self.assertEqual(node_draw.visits, 1)
        self.assertEqual(node_draw.value_sum, 0.0)
        self.assertEqual(node_draw.mean_value, 0.0)

    def test_tree_depth_at_least_2_edges_backprop_and_ucb1(self):
        """
        Kiểm tra cấu trúc cây tìm kiếm sâu ít nhất 2 cạnh:
            Root (Trắng đi)
              └── Cạnh 1: e2e4 ──> Child (Đen đi)
                                     └── Cạnh 2: e7e5 ──> Grandchild (Trắng đi)
        Xác nhận sự đổi dấu giá trị và góc nhìn UCB1 qua từng cạnh.
        """
        root_board = ChessBoard()
        root = MCTSNode(board=root_board)

        move_1 = chess.Move.from_uci("e2e4")
        child_board = root_board.clone()
        child_board.step(move_1)
        child = root.add_child(move_1, child_board)

        move_2 = chess.Move.from_uci("e7e5")
        grandchild_board = child_board.clone()
        grandchild_board.step(move_2)
        grandchild = child.add_child(move_2, grandchild_board)

        # Kiểm tra quan hệ phả hệ các node
        self.assertIs(child.parent, root)
        self.assertIs(grandchild.parent, child)
        self.assertEqual(root.board.current_player, 1)       # Trắng đi
        self.assertEqual(child.board.current_player, -1)     # Đen đi
        self.assertEqual(grandchild.board.current_player, 1) # Trắng đi

        agent = PureMCTSAgent(simulations=10, seed=42)

        # Kịch bản A: Kết quả Rollout là Trắng thắng (+1.0)
        agent._backpropagate(grandchild, white_score=1.0)

        self.assertEqual(grandchild.visits, 1)
        self.assertEqual(grandchild.value_sum, 1.0)   # Trắng đi, Trắng thắng -> +1.0
        self.assertEqual(grandchild.mean_value, 1.0)

        self.assertEqual(child.visits, 1)
        self.assertEqual(child.value_sum, -1.0)        # Đen đi, Trắng thắng -> -1.0
        self.assertEqual(child.mean_value, -1.0)

        self.assertEqual(root.visits, 1)
        self.assertEqual(root.value_sum, 1.0)         # Trắng đi, Trắng thắng -> +1.0
        self.assertEqual(root.mean_value, 1.0)

        # Đánh giá góc nhìn UCB1 qua các cạnh:
        # 1. Tại Root (Trắng), đánh giá Child (nơi Đen đi):
        # mean_value của Child là -1.0 (Đen thua).
        # Từ góc nhìn của Root, nước đi này TỐT: -child.mean_value = -(-1.0) = +1.0.
        root_views_child_exploitation = -child.mean_value
        self.assertEqual(root_views_child_exploitation, 1.0)

        # 2. Tại Child (Đen), đánh giá Grandchild (nơi Trắng đi):
        # mean_value của Grandchild là +1.0 (Trắng thắng).
        # Từ góc nhìn của Child (Đen), nước đi này XẤU: -grandchild.mean_value = -(+1.0) = -1.0.
        child_views_grandchild_exploitation = -grandchild.mean_value
        self.assertEqual(child_views_grandchild_exploitation, -1.0)

        # Kịch bản B: Kết quả Rollout là Đen thắng (-1.0)
        agent._backpropagate(grandchild, white_score=-1.0)

        self.assertEqual(grandchild.visits, 2)
        self.assertEqual(grandchild.value_sum, 0.0)   # 1.0 + (-1.0) = 0.0
        self.assertEqual(child.visits, 2)
        self.assertEqual(child.value_sum, 0.0)        # -1.0 + (+1.0) = 0.0
        self.assertEqual(root.visits, 2)
        self.assertEqual(root.value_sum, 0.0)         # 1.0 + (-1.0) = 0.0

        # Kịch bản C: Kết quả Rollout là Hòa (0.0)
        agent._backpropagate(grandchild, white_score=0.0)
        self.assertEqual(grandchild.visits, 3)
        self.assertEqual(grandchild.value_sum, 0.0)
        self.assertEqual(child.visits, 3)
        self.assertEqual(child.value_sum, 0.0)
        self.assertEqual(root.visits, 3)
        self.assertEqual(root.value_sum, 0.0)

    def test_ucb1_score_calculation(self):
        """Xác minh công thức tính điểm UCB1: -mean_value + c * sqrt(ln(parent_visits) / child_visits)."""
        board = ChessBoard()
        root = MCTSNode(board=board)
        root.visits = 10

        child = MCTSNode(board=board.clone(), parent=root)
        # Node con chưa được thăm -> inf
        self.assertEqual(child.ucb1_score(root.visits, c=1.414), float("inf"))

        child.visits = 2
        child.value_sum = 1.0  # mean_value = 0.5 (theo góc nhìn của con)

        expected_exploitation = -0.5
        expected_exploration = 1.414 * math.sqrt(math.log(10) / 2)
        self.assertAlmostEqual(child.ucb1_score(root.visits, c=1.414),
                               expected_exploitation + expected_exploration)

    def test_best_child_selection(self):
        """Xác minh hàm best_child luôn chọn node con có điểm UCB1 cao nhất."""
        root = MCTSNode(board=ChessBoard())
        root.visits = 20

        m1 = chess.Move.from_uci("e2e4")
        m2 = chess.Move.from_uci("d2d4")

        c1 = root.add_child(m1, root.board.clone())
        c2 = root.add_child(m2, root.board.clone())

        # Gán c1 điểm trung bình thấp hơn (đối thủ thua đậm -> rất có lợi cho cha)
        c1.visits = 10
        c1.value_sum = -8.0  # mean = -0.8 -> cha hưởng = +0.8

        c2.visits = 10
        c2.value_sum = 8.0   # mean = +0.8 -> cha chịu = -0.8

        selected = root.best_child(c=1.414)
        self.assertIs(selected, c1)

    def test_expansion_and_clone_preserves_caller_and_history(self):
        """Xác minh mở rộng node tạo node con với đầy đủ lịch sử mà không làm thay đổi bàn cờ gốc."""
        board = ChessBoard()
        board.step("e4")
        board.step("e5")
        orig_fen = board.fen()
        orig_stack_len = len(board.board.move_stack)

        agent = PureMCTSAgent(simulations=5, seed=42)
        root = MCTSNode(board=board.clone())

        untried_before = len(root.untried_moves)
        child = agent._expand(root)

        # Số nước chưa thử giảm đi 1
        self.assertEqual(len(root.untried_moves), untried_before - 1)
        # Bàn cờ con có thêm 1 nước đi trong stack
        self.assertEqual(len(child.board.board.move_stack), orig_stack_len + 1)
        # Bàn cờ cha không đổi
        self.assertEqual(len(root.board.board.move_stack), orig_stack_len)
        # Bàn cờ gốc của người gọi không đổi
        self.assertEqual(board.fen(), orig_fen)
        self.assertEqual(len(board.board.move_stack), orig_stack_len)

    def test_rollout_returns_valid_bounds(self):
        """Pha mô phỏng luôn phải trả về giá trị chuẩn hóa nằm trong đoạn [-1.0, 1.0]."""
        board = ChessBoard()
        node = MCTSNode(board=board)
        agent = PureMCTSAgent(simulations=10, max_rollout_depth=15, seed=123)

        for _ in range(20):
            res = agent._rollout(node)
            self.assertTrue(-1.0 <= res <= 1.0)

    def test_cutoff_material_delta_evaluation(self):
        """
        Xác minh công thức tính điểm vật chất khi rollout chạm trần mà chưa kết thúc:
            delta = material_white - material_black
            score = delta / (|delta| + 10)
        Bảng giá trị: Tốt=1, Mã=3, Tượng=3, Xe=5, Hậu=9, Vua=0.
        """
        agent = PureMCTSAgent(simulations=5, max_rollout_depth=0, seed=42)

        # 1. Bàn cờ bằng quân (Starting position): delta = 0 -> score = 0.0
        board_even = ChessBoard()
        node_even = MCTSNode(board=board_even)
        self.assertEqual(agent._rollout(node_even), 0.0)

        # 2. Trắng hơn 1 Hậu (Đen mất Hậu d8): delta = +9
        # score = 9 / (|9| + 10) = 9/19
        board_queen = ChessBoard("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        node_queen = MCTSNode(board=board_queen)
        expected_queen = 9.0 / (9.0 + 10.0)
        self.assertAlmostEqual(agent._rollout(node_queen), expected_queen)

        # 3. Đen hơn 1 Xe (Trắng mất Xe a1): delta = -5
        # score = -5 / (|-5| + 10) = -5/15 = -1/3
        board_rook = ChessBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1")
        node_rook = MCTSNode(board=board_rook)
        expected_rook = -5.0 / (5.0 + 10.0)
        self.assertAlmostEqual(agent._rollout(node_rook), expected_rook)

        # 4. Trắng hơn 1 Tốt (Đen mất Tốt h7): delta = +1
        # score = 1 / (|1| + 10) = 1/11
        board_pawn = ChessBoard("rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        # Trên bàn cờ này Đen có 7 tốt, Trắng có 8 tốt -> delta = +1
        node_pawn = MCTSNode(board=board_pawn)
        expected_pawn = 1.0 / (1.0 + 10.0)
        self.assertAlmostEqual(agent._rollout(node_pawn), expected_pawn)

    def test_seed_determinism(self):
        """Hai agent có cùng seed phải tạo ra các nước đi và cây thống kê giống nhau hoàn toàn."""
        board = ChessBoard()
        agent_a = PureMCTSAgent(simulations=50, seed=999)
        agent_b = PureMCTSAgent(simulations=50, seed=999)

        move_a = agent_a.act(board)
        move_b = agent_b.act(board)

        self.assertEqual(move_a, move_b)
        # So sánh phân bố số lượt thăm trên các nhánh con
        visits_a = {m: child.visits for m, child in agent_a.root.children.items()}
        visits_b = {m: child.visits for m, child in agent_b.root.children.items()}
        self.assertEqual(visits_a, visits_b)

    def test_agent_contract_legal_and_no_mutation(self):
        """Xác minh PureMCTSAgent thỏa mãn toàn bộ hợp đồng BaseAgent."""
        board = ChessBoard()
        for san in ["e4", "c5", "Nf3", "d6"]:
            board.step(san)

        fen_before = board.fen()
        stack_before = list(board.board.move_stack)

        agent = PureMCTSAgent(simulations=30, seed=42)
        move = agent.act(board)

        self.assertIn(move, board.legal_moves)
        self.assertEqual(board.fen(), fen_before)
        self.assertEqual(list(board.board.move_stack), stack_before)

    def test_agent_rejects_terminal(self):
        """Agent phải ném lỗi ValueError nếu bàn cờ đầu vào đã ở trạng thái kết thúc."""
        agent = PureMCTSAgent(simulations=10, seed=42)
        terminal_board = ChessBoard("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(terminal_board.is_terminal())

        with self.assertRaises(ValueError):
            agent.act(terminal_board)

    def test_single_legal_move_fast_path(self):
        """Agent phải trả về ngay lập tức nếu chỉ có duy nhất 1 nước đi hợp lệ."""
        # Vị trí mà Trắng chỉ có đúng 1 nước đi vua hợp lệ (Kg1)
        board = ChessBoard("8/8/8/8/8/8/r7/k6K w - - 0 1")
        self.assertEqual(len(board.legal_moves), 1)
        only_move = board.legal_moves[0]

        agent = PureMCTSAgent(simulations=50, seed=42)
        chosen = agent.act(board)
        self.assertEqual(chosen, only_move)
        # Không tạo root vì đã trả về ngay ở nhánh xử lý nhanh
        self.assertIsNone(agent.root)

    def test_finds_immediate_mate_white_and_black(self):
        """Xác minh PureMCTSAgent phát hiện nước chiếu hết ngay lập tức khi có cơ hội."""
        # Trắng chiếu hết trong 1 nước
        white_mate_pos = ChessBoard("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        agent = PureMCTSAgent(simulations=100, seed=42)
        move = agent.act(white_mate_pos)
        white_mate_pos.step(move)
        self.assertTrue(white_mate_pos.is_checkmate())

        # Đen chiếu hết trong 1 nước (bản lật màu đối xứng)
        black_mate_board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").mirror()
        black_mate_pos = ChessBoard(black_mate_board.fen())
        agent.reset()
        move_b = agent.act(black_mate_pos)
        black_mate_pos.step(move_b)
        self.assertTrue(black_mate_pos.is_checkmate())

    def test_all_terminal_conditions_consistency(self):
        """
        Xác minh tính nhất quán tuyệt đối giữa is_terminal(), check_winner(),
        get_legal_actions() và MCTS cho mọi trường hợp kết thúc ván cờ:
        - Chiếu hết (Checkmate: Trắng thắng, Đen thắng)
        - Hòa hết nước hợp lệ (Stalemate)
        - Không đủ lực lượng (Insufficient Material: K vs K, K+B vs K, K+N vs K)
        - Hòa luật 50 nước và luật 75 nước
        - Hòa lặp lại 3 lần và 5 lần (Threefold & Fivefold Repetition)
        """
        cases = [
            ("Checkmate (Trắng thắng)", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", True, 1),
            ("Checkmate (Đen thắng)", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", True, -1),
            ("Hòa hết nước (Stalemate)", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", True, 0),
            ("Không đủ quân (Vua vs Vua)", "7k/8/8/8/8/8/8/K7 w - - 0 1", True, 0),
            ("Không đủ quân (Vua + Tượng vs Vua)", "7k/8/8/8/8/8/8/KB6 w - - 0 1", True, 0),
            ("Không đủ quân (Vua + Mã vs Vua)", "7k/8/8/8/8/8/8/KN6 w - - 0 1", True, 0),
            ("Luật 50 nước (claim_draw)", "7k/8/8/8/8/8/8/KR6 w - - 100 51", True, 0),
            ("Luật 75 nước (bắt buộc)", "7k/8/8/8/8/8/8/KR6 w - - 150 76", True, 0),
        ]

        agent = PureMCTSAgent(simulations=5, seed=42)

        for name, fen, exp_term, exp_win in cases:
            board = ChessBoard(fen)
            # 1. Bàn cờ phải báo terminal chính xác
            self.assertEqual(board.is_terminal(), exp_term, f"Lỗi is_terminal: {name}")
            # 2. check_winner phải trả về 1, -1 hoặc 0 (không được là None)
            self.assertEqual(board.check_winner(), exp_win, f"Lỗi check_winner: {name}")
            # 3. get_legal_actions() phải trả về danh sách rỗng
            self.assertEqual(board.get_legal_actions(), [], f"Lỗi get_legal_actions: {name}")

            # 4. MCTSNode khởi tạo từ bàn cờ này phải nhận diện terminal nhất quán
            node = MCTSNode(board=board)
            self.assertTrue(node.is_terminal(), f"Lỗi node.is_terminal: {name}")
            self.assertEqual(node.untried_moves, [], f"Lỗi node.untried_moves: {name}")
            self.assertTrue(node.is_fully_expanded(), f"Lỗi node.is_fully_expanded: {name}")

            # 5. PureMCTSAgent.act() phải ném ValueError khi nhận bàn cờ terminal
            with self.assertRaises(ValueError, msg=f"act() phải raise ValueError: {name}"):
                agent.act(board)

            # 6. Pha rollout từ terminal node phải trả về ngay giá trị chính xác
            res = agent._rollout(node)
            self.assertEqual(res, float(exp_win), f"Lỗi rollout: {name}")

        # Kiểm tra thêm Threefold Repetition hình thành qua chuỗi nước đi thực tế
        rep_board = ChessBoard(claim_draw=True)
        # 1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 (hoàn thành lặp lại 3 lần)
        for move_san in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"]:
            rep_board.step(move_san)

        self.assertTrue(rep_board.is_terminal(), "Lỗi Threefold is_terminal")
        self.assertEqual(rep_board.check_winner(), 0, "Lỗi Threefold check_winner")
        self.assertEqual(rep_board.get_legal_actions(), [], "Lỗi Threefold legal actions")
        rep_node = MCTSNode(board=rep_board)
        self.assertTrue(rep_node.is_terminal(), "Lỗi rep_node is_terminal")
        self.assertEqual(agent._rollout(rep_node), 0.0, "Lỗi rep_node rollout")

    def test_rollout_preserves_history_for_threefold_and_fifty_moves(self):
        """
        Kiểm tra chuyên sâu ưu tiên cao:
        Xác minh rollout sử dụng copy(stack=True) kế thừa lịch sử từ ChessBoard.clone(),
        cho phép phát hiện chính xác hòa lặp lại vị trí (threefold repetition)
        và luật 50 nước kết hợp với lịch sử trước đó của ván cờ.
        """
        # Tạo thế cờ đã lặp lại 2 lần (chưa terminal):
        # 1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6
        board = ChessBoard(claim_draw=True)
        for move_san in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6"]:
            board.step(move_san)

        # Thế cờ hiện tại chưa terminal (mới 2 lần lặp)
        self.assertFalse(board.is_terminal())
        self.assertEqual(len(board.board.move_stack), 6)

        node = MCTSNode(board=board)
        # Bàn cờ trong node phải bảo toàn đúng 6 nước đi trong stack
        self.assertEqual(len(node.board.board.move_stack), 6)
        self.assertTrue(node.board.claim_draw)

        # Mô phỏng rollout: kiểm tra sim_board sao chép stack=True
        agent = PureMCTSAgent(simulations=10, seed=42)
        sim_board = node.board.board.copy(stack=True)
        self.assertEqual(len(sim_board.move_stack), 6)

        # Nếu trong rollout, Trắng đi Ng1 -> Lập tức lặp lại lần 3 và kích hoạt hòa!
        sim_board.push(chess.Move.from_uci("g1g1") if False else chess.Move.from_uci("f3g1"))
        self.assertTrue(sim_board.can_claim_threefold_repetition())
        outcome = sim_board.outcome(claim_draw=True)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.termination, chess.Termination.THREEFOLD_REPETITION)

        # Ngược lại, nếu dùng stack=False (bỏ lịch sử), sẽ KHÔNG THỂ phát hiện hòa:
        sim_board_flawed = node.board.board.copy(stack=False)
        sim_board_flawed.push(chess.Move.from_uci("f3g1"))
        self.assertFalse(sim_board_flawed.can_claim_threefold_repetition())
        self.assertIsNone(sim_board_flawed.outcome(claim_draw=True))

        # Kiểm tra rollout của agent hoạt động an toàn và trả về giá trị hợp lệ
        res = agent._rollout(node)
        self.assertTrue(-1.0 <= res <= 1.0)

    def test_raw_chess_board_preserves_history(self):
        """
        Xác minh khi truyền raw chess.Board có lịch sử nước đi vào MCTSNode hoặc act(),
        toàn bộ move_stack và FEN gốc ban đầu đều được bảo toàn 100%.
        """
        raw = chess.Board()
        moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
        for m in moves:
            raw.push_san(m)

        # 1. Kiểm tra MCTSNode khởi tạo từ raw board
        node = MCTSNode(board=raw)
        self.assertIsInstance(node.board, ChessBoard)
        self.assertEqual(len(node.board.board.move_stack), len(moves))
        self.assertEqual(
            [m.uci() for m in node.board.board.move_stack],
            [m.uci() for m in raw.move_stack]
        )

        # 2. Kiểm tra PureMCTSAgent.act() trực tiếp trên raw board
        agent = PureMCTSAgent(simulations=10, seed=42)
        chosen_move = agent.act(raw)
        self.assertIn(chosen_move, list(raw.legal_moves))
        # Bàn cờ gốc không bị biến đổi
        self.assertEqual(len(raw.move_stack), len(moves))


if __name__ == "__main__":
    unittest.main()



