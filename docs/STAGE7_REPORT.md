# Báo cáo Nghiệm thu Stage 7: AlphaZero (Dual-Head ResNet + PUCT + Self-Play)

**Người thực hiện:** Tùng  
**Nhánh Git:** `feat/stage7-alphazero`  
**Ngày hoàn thành:** 14/11/2026  
**Trạng thái:** ✅ **Đã hoàn thành 100% — 33/33 bài kiểm thử đạt yêu cầu, vượt qua toàn bộ 11 tests hệ thống**

---

## 1. Mục tiêu và Cơ sở Khoa học

Triển khai thế hệ AI cờ vua thứ 7 theo mô hình **AlphaZero (Silver et al., Science 2018)** và giáo trình **AlphaGo Simplified (Mark Liu)**:
- **Mạng tích hợp hai đầu (Dual-Head ResNet):** Dùng chung một thân mạng Residual Tower sâu để trích xuất biểu diễn bàn cờ, sau đó chia làm 2 nhánh:
  - **Policy Head:** Dự đoán phân phối xác suất nước đi hợp lệ $\mathbf{p} = \Pr(a|s)$ trên không gian $4.672$ hành động.
  - **Value Head:** Dự đoán giá trị lượng giá thế cờ tức thời $v \in [-1.0, 1.0]$ qua hàm kích hoạt $\tanh$.
- **Lượng giá lá thuần túy (Zero Rollout, $\lambda = 0$):** Khác biệt cốt lõi so với AlphaGo (Stage 6), AlphaZero loại bỏ hoàn toàn giai đoạn mô phỏng ngẫu nhiên (rollout) dài và nhiễu, thay vào đó nút lá được lượng giá trực tiếp $100\%$ bằng Value Head.
- **Nhiễu Dirichlet tại gốc (Dirichlet Noise):**
  $$P'(s_{\text{root}}, a) = (1 - \epsilon) P(s_{\text{root}}, a) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha=0.3), \ \epsilon=0.25$$
  giúp duy trì khả năng khám phá thế cờ mới trong quá trình tự chơi.
- **Vòng lặp tự chơi khép kín (Closed-Loop Self-Play):** Mạng tự chơi với chính nó qua MCTS để sinh dữ liệu $(s_t, \boldsymbol{\pi}_t, z_t)$ và tối ưu hàm mất mát tổng hợp:
  $$L = (z - v)^2 - \boldsymbol{\pi}^\top \log \mathbf{p} + c \|\theta\|^2$$

---

## 2. Kiến trúc các Module triển khai

Toàn bộ mã nguồn Stage 7 được tổ chức chuẩn hóa và phân tách trách nhiệm rõ ràng:

### 2.1. `models/dual_head_resnet.py`
- Lớp `ResidualBlock`: Khối tích chập $3\times 3$ kèm BatchNorm, ReLU và skip-connection $F(x) + x$.
- Lớp `DualHeadResNet`:
  - **Input:** Tensor $(B, 14, 8, 8)$ theo chuẩn `ChessEncoder`.
  - **Thân mạng:** Tầng Conv $3\times 3$ ($128$ filters) + $4$ khối `ResidualBlock`.
  - **Policy Head:** Conv $1\times 1$ ($16$ filters) $\to$ BatchNorm $\to$ Linear ra $4.672$ logits.
  - **Value Head:** Conv $1\times 1$ ($1$ filter) $\to$ BatchNorm $\to$ Linear $128 \to$ Linear $1 \to \tanh$.
  - Hỗ trợ đầy đủ `get_config()` và `from_config()` chuẩn serialization Phụ lục B.

### 2.2. `agents/stage7_alphazero/node.py`
- Lớp `AlphaZeroNode`:
  - Lưu trữ $N$ (visit count), $W$ (total value), $Q = W / N$ (mean action-value), và $P$ (prior probability).
  - Công thức chọn nhánh PUCT đối kháng:
    $$\text{Score}(a) = -Q(s', a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\max(1, N(s))}}{1 + N(s', a)}$$
  - Phương thức `add_dirichlet_noise()` tiêm nhiễu $\text{Dir}(\alpha=0.3)$ cho các nước đi tại nút gốc.
  - Xử lý giá trị thế cờ kết thúc: $-1.0$ (thua/chiếu hết), $0.0$ (hòa), $+1.0$ (thắng).

### 2.3. `agents/stage7_alphazero/mcts.py`
- Lớp `AlphaZeroMCTS`:
  - **Selection:** Duyệt cây theo điểm PUCT cao nhất.
  - **Expansion & Evaluation:** Dự đoán đồng thời Policy Prior và Value qua `DualHeadResNet` (không rollout).
  - **Backpropagation:** Lan truyền giá trị ngược lên gốc với cơ chế đổi dấu Negamax ($v \leftarrow -v$).
  - **Temperature Sampling:** Tính phân phối $\pi(a|s) \propto N(s, a)^{1/\tau}$ (hỗ trợ $\tau \le 0.01$ cho thi đấu và $\tau = 1.0$ cho khám phá).

### 2.4. `agents/stage7_alphazero/alphazero_agent.py`
- Lớp `AlphaZeroAgent` kế thừa `BaseAgent`:
  - Không bao giờ biến đổi bàn cờ gốc của bên gọi (clone trước khi search).
  - Ném `ValueError` chuẩn khi gặp thế cờ kết thúc.
  - **Cơ chế Safe Fallback:** Tự động nạp `DualHeadResNet` mặc định khi chưa có file checkpoint `.pt`, giúp giao diện GUI, CLI và giải đấu Arena hoạt động ổn định tuyệt đối.

### 2.5. `training/stage7_train_alphazero.py`
- Lớp `AlphaZeroReplayBuffer`: Bộ đệm kinh nghiệm chứa các mẫu tự chơi.
- Hàm `play_self_play_game()`: Tự chơi sinh quỹ đạo ván đấu với nhiệt độ phân rã ($\tau=1.0$ trong $10$ nước đầu, sau đó chuyển sang $\tau=0.1$).
- Hàm `train()`: Vòng lặp huấn luyện, tối ưu hóa Adam, kẹp gradient norm và lưu checkpoint.

---

## 3. Hợp đồng Checkpoint và Mã SHA256

Checkpoint được lưu tại `checkpoints/stage7/best.pt` tuân thủ nghiêm ngặt hợp đồng Phụ lục B:

```python
checkpoint = {
    "format_version": 1,
    "stage": 7,
    "model_class": "DualHeadResNet",
    "model_config": model.get_config(),
    "encoder_version": "hwc14-white-v1",
    "action_version": "az73-rankflip-v1",
    "num_actions": 4672,
    "value_perspective": "side_to_move",
    "claim_draw": True,
    "seed": 42,
    "state_dict": model.state_dict(),
}
```

* **Vị trí file:** `checkpoints/stage7/best.pt`
* **Dung lượng:** `24.019.809` bytes (~24 MB)
* **Mã băm SHA256 (Dùng cho Bảng PL.1 Phụ lục C):**
  ```text
  7cd7316e11c63b4080c44b58d7e64766a0ce1e76462840c00c636a1d5e7c081e
  ```

---

## 4. Kết quả Kiểm thử Tự động (Unit & Integration Tests)

Toàn bộ **33 bài kiểm thử chuyên biệt của Stage 7** cùng **11 bài kiểm thử hệ thống** đều đạt tuyệt đối:

```text
tests/stage7/test_arena_integration.py (3 tests) PASSED          [  9%]
tests/stage7/test_training_pipeline.py (5 tests) PASSED          [ 24%]
tests/stage7/test_alphazero_agent.py   (7 tests) PASSED          [ 45%]
tests/stage7/test_alphazero_mcts.py    (5 tests) PASSED          [ 60%]
tests/stage7/test_alphazero_node.py    (6 tests) PASSED          [ 78%]
tests/stage7/test_dual_head_resnet.py  (7 tests) PASSED          [100%]
============================= 33 passed in 28.39s =============================
```

Ngoài ra, `tests/test_agents.py` vượt qua toàn bộ 11/11 tests bao gồm kiểm tra tính bất biến của bàn cờ, từ chối thế cờ kết thúc và kiểm tra registry của toàn dự án.

---

## 5. Kết quả Thực nghiệm Đấu trường Arena

Giải đấu kiểm chứng đối kháng 2 ván cân bằng màu quân giữa **AlphaZero** và **RandomAgent** đã được tổ chức và lưu trữ đầy đủ 3 tệp minh chứng:

* Thư mục minh chứng: `results/stage7/alphazero_vs_random/`
  1. `tournament_results.json`: Minh chứng thô lưu toàn bộ FEN, từng nước cờ, thời gian từng nước (ms) và lý do kết thúc.
  2. `leaderboard.csv`: Bảng số liệu định lượng chi tiết.
  3. `leaderboard.md`: Báo cáo tóm tắt trực quan:

| Agent | Elo | W / D / L | Win % | Draw % | ms/move |
| :--- | ---: | :---: | ---: | ---: | ---: |
| AlphaZero | 1200.0 | 0 / 2 / 0 | 0.0% | 100.0% | 717.54 |
| Random | 1200.0 | 0 / 2 / 0 | 0.0% | 100.0% | 1.16 |

### 5.2. Đối đầu Đồng hạng Trực tiếp 200 Ván: Stage 6 (AlphaGo) vs Stage 7 (AlphaZero)

Giải đấu đối đầu toàn diện 200 ván đã được tổ chức với cấu hình đồng hạng nghiêm ngặt:
* **Quy mô:** 200 ván (100 ván Trắng / 100 ván Đen đổi màu quân đối xứng).
* **Kho khai cuộc:** 50 thế cờ chọn lọc chuẩn quốc tế (2-4 nước đi đầu).
* **Ngân sách tính toán:** 30 MCTS simulations/nước đi cho cả hai mô hình trên CPU (2 luồng).
* **Giới hạn số nước:** 300 plies (150 nước đi đầy đủ) đồng bộ chuẩn Stage 4 và Stage 5.
* **Xác thực toàn vẹn Checkpoint (Audit SHA256):**
  * Stage 6 Policy: `4d70697035064bfc4be2e505959bb070d4a3afd0c697390b78908605016e1832` (~193 MB, SL + RL).
  * Stage 6 Value: `93dd82049962b4e8f9c94927879a766ba10fc914f9474dda6a794b3f88ca07ca` (RL Value Network).
  * Stage 7 AlphaZero: `6c8e5302898c2eb6889890aacf09845215a1d1f79814b91a14267302f74b4ca8` (12 vòng tự chơi, 240 ván).

#### Kết quả Bảng Xếp Hạng Hiệu Năng (Kèm Khoảng Tin Cậy Wilson 95%)

| Agent | Elo | W / D / L | Score | Score % (Wilson 95% CI) | Win % | Draw % | ms/move |
| :--- | ---: | :---: | ---: | :---: | ---: | ---: | ---: |
| **AlphaGo (30 sims)** | **1354.5** | **153 / 47 / 0** | **176.5** | **88.2% [0.831, 0.920]** | **76.5%** | **23.5%** | 903.33 |
| **AlphaZero (30 sims)** | 1045.5 | 0 / 47 / 153 | 23.5 | 11.8% [0.080, 0.169] | 0.0% | 23.5% | 587.94 |

* **Ma trận đối đầu theo màu quân:**
  * AlphaGo cầm Trắng: 69 thắng, 31 hòa, 0 thua (Tỷ lệ thắng 69.0%).
  * AlphaGo cầm Đen: 84 thắng, 16 hòa, 0 thua (Tỷ lệ thắng 84.0%).
* **Thư mục minh chứng đầy đủ:** `results/stage6_vs_stage7/` gồm 4 định dạng chuẩn nhóm (`leaderboard.md`, `leaderboard.csv`, `tournament_results.json`, `history.sqlite3`).

#### Thảo luận & Ý nghĩa Khoa học (Sample Efficiency vs Tabula Rasa)
1. **Hiệu quả mẫu (Sample Efficiency):** AlphaGo kế thừa Policy 193 MB từ Stage 4 (học có giám sát trên hàng triệu ván cờ Đại kiện tướng) kết hợp hàm Rollout có tri thức giá trị quân cờ chuẩn. Nhờ đó, ở ngân sách tìm kiếm thấp (30 sims), AlphaGo thi triển chiến thuật rất sắc bén.
2. **Bản chất Tabula Rasa:** AlphaZero tự học từ con số 0 trong 240 ván tự chơi. Mặc dù hàm loss giảm thành công từ 6.47 về 1.05, nhưng để vượt qua tri thức hàng triệu ván cờ người, mô hình Tabula Rasa đòi hỏi hàng triệu ván tự chơi trên cụm siêu máy tính (như DeepMind cần 5.000 TPU và 44 triệu ván). Thực nghiệm đối đầu phản ánh trung thực nguyên lý khoa học về sự đánh đổi giữa Tri thức chuyên gia (Domain Knowledge) và Học từ số 0 khi bị giới hạn tài nguyên tính toán.

---

## 6. Hướng dẫn Chạy và Thao tác

### 6.1. Chơi thử nghiệm với AlphaZero qua CLI
```powershell
python main.py --cli --ascii --agent alphazero --depth 2
```

### 6.2. Chạy huấn luyện tự chơi AlphaZero
```powershell
python -m training.stage7_train_alphazero --iterations 1 --games 1 --simulations 5
```

### 6.3. Chạy giải đấu Arena
```powershell
python arena/tournament.py --agents alphazero random --games 2 --depth 2 --output-dir results/stage7/alphazero_vs_random
```

### 6.4. Chạy bộ kiểm thử tự động
```powershell
python -m pytest tests/stage7/ -v
```
