# Báo cáo Nghiệm thu Stage 6: AlphaGo MCTS (Policy Prior + Value Network)

**Người thực hiện:** Tùng  
**Nhánh Git:** `feat/stage6-alphago-mcts`  
**Ngày hoàn thành:** 03/11/2026  
**Trạng thái:** ✅ **Đã hoàn thành 100% — 24/24 bài kiểm thử đạt yêu cầu**

---

## 1. Mục tiêu và Cơ sở Khoa học

Triển khai thế hệ AI cờ vua thứ 6 theo mô hình **AlphaGo (Silver et al., Nature 2016)**, kết hợp cây tìm kiếm Monte Carlo (MCTS) với mạng nơ-ron học sâu:
- **Thu hẹp độ rộng tìm kiếm (Reduce Breadth)** bằng xác suất tiên nghiệm $P(s, a)$ từ **Policy Network (Stage 4)**.
- **Thu hẹp độ sâu tìm kiếm (Reduce Depth)** bằng điểm lượng giá tức thời $V_{\text{net}}(s)$ từ **Value Network (Stage 5)**.
- **Lượng giá lá hỗn hợp (Hybrid Leaf Evaluation)** kết hợp giữa Value Network và Fast Rollout:
  $$V_{\text{leaf}} = (1 - \lambda) \cdot V_{\text{net}} + \lambda \cdot z_{\text{rollout}}$$

---

## 2. Kiến trúc các Module triển khai

Toàn bộ mã nguồn Stage 6 được tổ chức độc lập, module hóa cao trong thư mục `agents/stage6_alphago/`:

```text
agents/stage6_alphago/
├── __init__.py           # Export AlphaGoNode, AlphaGoMCTS, AlphaGoAgent
├── node.py               # Cấu trúc Node lưu Prior P, Visits N, Value W, Q-mean
├── mcts.py               # Động cơ tìm kiếm AlphaGoMCTS (Selection, Expansion, Hybrid Eval, Backprop)
└── alphago_agent.py      # Agent cờ vua hoàn chỉnh kế thừa BaseAgent, nạp checkpoint St4/St5
```

### Chi tiết các lớp cốt lõi:
1. **`AlphaGoNode` (`node.py`)**:
   - Lưu trữ 4 chỉ số: $N(s, a)$, $W(s, a)$, $Q(s, a)$, $P(s, a)$.
   - Công thức chọn nhánh Selection:
     $$\text{Score}(s, a) = -Q_{\text{child}} + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\max(1, N_{\text{parent}})}}{1 + N_{\text{child}}}$$
   - Đảo dấu $-Q_{\text{child}}$ chuẩn xác theo lý thuyết trò chơi zero-sum đối kháng.
2. **`AlphaGoMCTS` (`mcts.py`)**:
   - Điều phối $M$ lượt mô phỏng cây tìm kiếm.
   - Cơ chế Hybrid Leaf Evaluation: hỗ trợ đầy đủ 3 chế độ ($\lambda = 0.0$ pure value, $\lambda = 1.0$ pure rollout, $\lambda = 0.5$ hybrid).
   - Quyết định nước đi theo chiến lược **Robust Child** ($N$ cao nhất).
   - Hàm `get_action_probs(board, temperature)` tính phân phối $\pi(a|s) \propto N(s, a)^{1/\tau}$ có tổng bằng đúng 1.0.
3. **`AlphaGoAgent` (`alphago_agent.py`)**:
   - Kế thừa `BaseAgent`, triển khai `act(board) -> chess.Move`.
   - Kết nối với `CNNPolicyNetwork` (Stage 4) qua Softmax có ổn định số học để tính Prior $P(s, a)$.
   - Kết nối với `ValueNetwork` (Stage 5) để lượng giá thế cờ $v \in [-1.0, 1.0]$.
   - Hỗ trợ nạp checkpoint `.pt` chuẩn hợp đồng `HANDOFF.md`.
   - Cơ chế Fallback an toàn: tự khởi tạo mạng mặc định nếu không truyền checkpoint, giúp bot luôn chạy được trên GUI/Arena mà không bị văng lỗi.

---

## 3. Tích hợp Hệ thống (Factory, CLI, GUI, Arena)

- **`agents/__init__.py`**:
  - Đăng ký `"alphago": AlphaGoAgent` vào `AGENT_REGISTRY`.
  - Thêm vào `CLASSIC_AGENTS` để hỗ trợ thi đấu vòng tròn tự động.
  - Cập nhật hàm Factory `build_agent()` tự động ánh xạ `--depth` sang ngân sách mô phỏng:
    $$\text{simulations} = \max(10, \text{depth} \times 15)$$
- **CLI & GUI**:
  - Chơi terminal: `python main.py --cli --ascii --agent alphago --depth 2`
  - Chơi đồ họa Pygame: `python main.py --agent alphago --depth 2`
- **Đấu trường Arena**:
  - Thi đấu 1v1 hoặc Tournament qua `arena/tournament.py` và `arena/evaluator.py`.

---

## 4. Kết quả Kiểm thử Toàn diện (24/24 Test PASS)

Bộ kiểm thử độc lập được đặt tại `tests/stage6/`:
- `tests/stage6/test_node.py` (5 tests): Toán học PUCT, ưu tiên Prior, đảo dấu Value.
- `tests/stage6/test_mcts.py` (7 tests): Động cơ tìm kiếm, đường tắt 1 nước, chặn lỗi chiếu hết, chuẩn hóa xác suất.
- `tests/stage6/test_agent.py` (6 tests): Suy luận mạng nơ-ron, chơi cả Trắng và Đen, nạp checkpoint thực tế/dummy.
- `tests/stage6/test_cli_integration.py` (3 tests): Đăng ký Factory và tương thích CLI.
- `tests/stage6/test_arena_integration.py` (3 tests): Thi đấu 2 ván cân bằng màu, xử lý phong cấp, ăn tốt qua đường en-passant, chiếu đôi.

```text
python -m unittest discover -s tests/stage6
........................
----------------------------------------------------------------------
Ran 24 tests in 21.322s

OK (24/24 tests passed)
```

---

## 5. Kết luận & Bàn giao sang Stage 7 (AlphaZero)

Stage 6 đã hoàn thành trọn vẹn mọi yêu cầu kỹ thuật và hợp đồng bàn giao. Nền tảng cây MCTS kết hợp mạng nơ-ron đã được kiểm chứng hoạt động ổn định tuyệt đối, sẵn sàng làm bệ phóng cho **Stage 7 (AlphaZero)**:
- Nâng cấp mạng riêng lẻ lên `DualHeadResNet` (chung thân ResNet).
- Áp dụng thuật toán tìm kiếm PUCT thuần túy (không rollout) kèm Dirichlet noise.
- Xây dựng vòng lặp tự chơi (Self-play) hoàn toàn từ con số không.

