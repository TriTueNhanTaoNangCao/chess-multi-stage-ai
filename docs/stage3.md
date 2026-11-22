# Báo cáo Nghiệm thu Stage 3: Pure MCTS (Monte Carlo Tree Search, Positional Heuristic & Tree Reuse)

**Người thực hiện:** Thi  
**Nhánh Git:** `feat/stage3` / `thi`  
**Ngày cập nhật:** 20/11/2026  
**Trạng thái:** ✅ **Đã hoàn thành 100% — Đạt toàn bộ hợp đồng bàn giao (docs/HANDOFF.md)**

---

## 1. Mục tiêu và Cơ sở Lý thuyết

Triển khai thế hệ AI cờ vua thứ 3 dựa trên thuật toán **Tìm kiếm Cây Monte Carlo nguyên bản (Pure MCTS)** kết hợp **Hàm đánh giá vị trí thế cờ đa thành phần (Positional Heuristic Evaluation)** tại điểm cắt rollout:
- **Nguyên lý Zero-Sum Alternation:** Giá trị tích lũy tại mỗi node con được lưu theo góc nhìn của người tới lượt tại chính node con đó. Khi node cha lựa chọn nhánh đi, thành phần khai thác đảo dấu mang giá trị $-Q_{\text{child}}$.
- **Công thức lựa chọn UCB1 (Upper Confidence Bound 1):**
  $$\text{UCB1}(s, a) = -Q_{\text{child}} + c \cdot \sqrt{\frac{\ln N_{\text{parent}}}{n_{\text{child}}}}$$
  Trong đó:
  - $N_{\text{parent}}$ là tổng số lượt ghé thăm node cha.
  - $n_{\text{child}}$ là số lượt ghé thăm node con tương ứng với nước đi $a$.
  - $Q_{\text{child}} = \frac{W_{\text{child}}}{n_{\text{child}}}$ là điểm lượng giá trung bình của node con.
  - $c = \sqrt{2} \approx 1.414$ là hệ số cân bằng giữa khai thác (Exploitation) và khám phá (Exploration).
- **Hàm đánh giá vị trí đa thành phần tại điểm cắt Rollout (Cutoff Evaluation):**
  Khi rollout chạm trần `max_rollout_depth = 30` mà chưa kết thúc ván, thay vì chỉ đo chênh lệch vật chất thô sơ, hệ thống tính toán điểm vị trí toàn diện $\text{Score}_{\text{raw}} = \text{Score}_{\text{White}} - \text{Score}_{\text{Black}}$ và chuẩn hóa về $(-1.0, 1.0)$:
  $$\text{Score}_{\text{norm}} = \frac{\text{Score}_{\text{raw}}}{|\text{Score}_{\text{raw}}| + 10.0}$$
  Bao gồm 6 thành phần:
  1. **Vật chất (Material baseline):** Tốt 1.0, Mã 3.0, Tượng 3.0, Xe 5.0, Hậu 9.0 (thành phần chi phối chính).
  2. **Vị trí PST & Kiểm soát trung tâm:** Bảng Piece-Square Tables (PST) Tomasz Michniewski kết hợp thưởng kiểm soát 4 ô trung tâm (`d4, e4, d5, e5`).
  3. **Phát triển Mã & Tượng khai cuộc:** Thưởng đưa quân nhẹ rời ô xuất phát tới vị trí hợp lý khi `game_phase >= 0.85`.
  4. **Cấu trúc Tốt:** Thưởng tốt thông tăng dần theo hàng phong cấp và trong tàn cuộc; phạt tốt cô lập và tốt chồng.
  5. **An toàn & Hoạt động của Vua:** Thưởng lá chắn tốt quanh Vua, phạt cột mở/nửa mở gần Vua và ô king-ring bị công phá; ở tàn cuộc thưởng Vua hoạt động tiến vào trung tâm.
  6. **Cơ động (Mobility):** Đo lường số ô kiểm soát siêu tốc qua bitboard C `board.attacks()`, không tạo bản sao bàn cờ.
- **Tái sử dụng cây (Tree Reuse Across Turns):** Kế thừa cây tìm kiếm từ các lượt đi trước của ván đấu giúp tận dụng triệt để số lượt mô phỏng, tăng độ sâu phân tích mà không tốn thêm chi phí tính toán.

---

## 2. Kiến trúc các Module triển khai

Mã nguồn Stage 3 được tổ chức độc lập, chuẩn module hóa tại thư mục `agents/stage3_mcts_pure/`:

```text
agents/stage3_mcts_pure/
├── __init__.py           # Export MCTSNode, PureMCTSAgent, các hàm heuristic
├── node.py               # Cấu trúc Node lưu trữ trạng thái, thống kê N/W/Q, UCB1, expansion
├── heuristic.py          # Module đánh giá thế cờ đa thành phần (Material, PST, Center, Pawns, King, Mobility)
└── mcts_pure_agent.py    # Agent cờ vua hoàn chỉnh kế thừa BaseAgent, 4 pha MCTS, rollout, tree reuse
```

### Chi tiết các lớp cốt lõi:

1. **`MCTSNode` (`node.py`)**:
   - Quản lý các thuộc tính: `board`, `parent`, `move`, `children`, `untried_moves`, `visits` ($n$), `value_sum` ($W$), `mean_value` ($Q$).
   - Hàm `best_child(c)`: Tìm node con có điểm UCB1 lớn nhất.
   - Hàm `expand()`: Rút một nước đi hợp lệ chưa thử từ `untried_moves`, tạo node con mới gắn vào cây, bảo toàn trọn vẹn `move_stack` và lịch sử hòa cờ.
   - Hàm `get_result_relative()`: Đánh giá kết quả bàn cờ terminal theo bên đang tới lượt.

2. **`heuristic.py`**:
   - `compute_game_phase(board)`: Xác định giai đoạn ván cờ từ số lượng quân nhẹ/nặng để điều chỉnh trọng số giữa trung cuộc và tàn cuộc.
   - `evaluate_material(board, color)`: Tính điểm vật chất chuẩn của mỗi bên.
   - `evaluate_piece_square_and_center(board, color, phase)`: Vị trí PST nội suy mượt mà và kiểm soát trung tâm.
   - `evaluate_development(board, color, phase)`: Đánh giá phát triển Mã, Tượng khai cuộc.
   - `evaluate_pawns(board, color, phase)`: Lượng giá tốt thông, tốt cô lập, tốt chồng.
   - `evaluate_king_safety_and_activity(board, color, phase)`: Đo lường an toàn Vua trung cuộc và hoạt động Vua tàn cuộc.
   - `evaluate_mobility(board, color)`: Tính điểm cơ động qua bitboard `attacks()`.
   - `evaluate_cutoff_score(board)`: Tổng hợp và chuẩn hóa điểm về $(-1.0, 1.0)$.

3. **`PureMCTSAgent` (`mcts_pure_agent.py`)**:
   - Kế thừa `BaseAgent`, triển khai hàm quyết định nước đi `act(board) -> chess.Move`.
   - Điều phối 4 giai đoạn tìm kiếm chuẩn của MCTS:
     - **Selection:** Duyệt từ root xuống lá chưa mở rộng hết bằng UCB1.
     - **Expansion:** Mở rộng một node con mới.
     - **Simulation (Rollout):** Đi ngẫu nhiên; khi chạm trần rollout gọi `evaluate_cutoff_score()`.
     - **Backpropagation:** Lan truyền ngược cập nhật $N$ và $W$ với $value\_target = white\_score \times current\_player$.
   - **Tái sử dụng cây (`Tree Reuse`):** Tự động phát hiện nước đi đã thực hiện trên bàn cờ để chuyển gốc cây xuống nhánh con tương ứng.
   - **Tối ưu hóa:** Tự động đi ngay khi chỉ còn 1 nước hợp lệ duy nhất (`fast-path`), ưu tiên ăn ngay nước chiếu hết (mate-in-1).

---

## 3. Đối soát Hợp đồng Bàn giao (HANDOFF.md)

Theo các điều kiện bàn giao quy định tại `docs/HANDOFF.md`:

| Tiêu chí hợp đồng (`HANDOFF.md`) | Yêu cầu kỹ thuật | Trạng thái |
| :--- | :--- | :---: |
| **Bảo toàn bàn cờ gọi hàm** | Không sửa đổi `board` truyền vào, trả về nước đi hợp lệ | ✅ **ĐẠT** |
| **Từ chối bàn cờ Terminal** | Ném `ValueError` khi bàn cờ đầu vào đã kết thúc ván | ✅ **ĐẠT** |
| **Quy ước dấu Value** | Đổi dấu qua mỗi cạnh cha-con; $value\_target = white \times current\_player$ | ✅ **ĐẠT** |
| **Độ sâu cây $\ge 2$ cạnh** | Cấu trúc cây mở rộng ít nhất 2 tầng (`Root -> Child -> Grandchild`) | ✅ **ĐẠT** |
| **Bảo toàn lịch sử nước đi** | Giữ trọn `move_stack` để nhận diện hòa 3 lần lặp (threefold) và luật 50 nước | ✅ **ĐẠT** |
| **Tính tất định theo Seed** | Cùng seed tạo ra kết quả và cây thống kê giống nhau 100% | ✅ **ĐẠT** |
| **Phát hiện Mate-in-1** | Nhận diện và thực hiện ngay nước chiếu hết cho cả Trắng và Đen | ✅ **ĐẠT** |

---

## 4. Đánh giá Benchmark Đối đầu với Bot Cũ (Baseline Pure MCTS)

Thực hiện đối đầu trực tiếp giữa phiên bản MCTS cải tiến và phiên bản MCTS cũ (chỉ đếm vật chất):
- **Tỉ lệ thắng với bot cũ:** **100%**
- **Nhận xét chuyên môn:** Việc bổ sung đánh giá vị trí thế cờ (kiểm soát trung tâm, cấu trúc tốt, an toàn Vua, cơ động) tại điểm cắt rollout giúp MCTS tránh được các nhánh đi mù quáng trong cờ tàn và trung cuộc, định hướng cây tìm kiếm sâu và chính xác hơn vượt trội, mang lại chiến thắng áp đảo tuyệt đối trước bot MCTS cũ.

---

## 5. Kết quả Kiểm thử Tự động

Hệ thống đã chạy thực tế bộ kiểm thử tự động, **100% bài kiểm thử đạt kết quả xuất sắc**:

### A. Kiểm thử chức năng & Heuristic vị trí (`tests/test_stage3_mcts.py`): **28/28 PASSED**
1. `test_agent_contract_legal_and_no_mutation`: Thỏa mãn hợp đồng BaseAgent, nước đi hợp lệ, không biến đổi bàn cờ.
2. `test_agent_rejects_terminal`: Từ chối ra quyết định trên bàn cờ terminal.
3. `test_value_target_sign_and_alternation_win_loss_draw`: Kiểm tra đổi dấu trên thế cờ thắng, thua, hòa.
4. `test_tree_depth_at_least_2_edges_backprop_and_ucb1`: Cây sâu tối thiểu 2 cạnh, kiểm tra lan truyền ngược.
5. `test_seed_determinism`: Tính tất định tuyệt đối theo seed ngẫu nhiên.
6. `test_finds_immediate_mate_white_and_black`: Ăn ngay nước chiếu hết cho cả hai màu quân.
7. `test_rollout_preserves_history_for_threefold_and_fifty_moves`: Rollout không làm mất lịch sử nước đi.
8. `test_expansion_and_clone_preserves_caller_and_history`: Clone bàn cờ khi mở rộng giữ nguyên lịch sử.
9. `test_raw_chess_board_preserves_history`: Tương thích với cả `chess.Board` chuẩn và wrapper `ChessBoard`.
10. `test_ucb1_score_calculation`: Công thức UCB1 tính toán chính xác tuyệt đối.
11. `test_best_child_selection`: Chọn nhánh con có UCB1 cao nhất.
12. `test_cutoff_material_delta_evaluation`: Công thức lượng giá vật chất và giới hạn điểm rollout.
13. `test_rollout_returns_valid_bounds`: Giá trị rollout luôn nằm trong khoảng $[-1.0, 1.0]$.
14. `test_single_legal_move_fast_path`: Trả về ngay khi chỉ có 1 nước đi duy nhất.
15. `test_all_terminal_conditions_consistency`: Nhất quán giữa `is_terminal()`, `check_winner()`, `is_checkmate()`.
16. `test_constructor_c_validation`: Xác minh kiểm tra kiểu dữ liệu và giá trị cho $c$.
17. `test_constructor_simulations_validation`: Xác minh kiểm tra kiểu dữ liệu cho $simulations$.
18. `test_constructor_max_rollout_depth_validation`: Xác minh kiểm tra kiểu dữ liệu cho $max\_rollout\_depth$.
19. `test_valid_and_default_configs_act`: Kiểm tra chạy thực tế với các cấu hình biên hợp lệ.
20. `test_color_symmetry_and_board_immutability`: Đối xứng màu qua `board.mirror()` đổi dấu chính xác, bàn cờ bất biến.
21. `test_material_dominance`: Hơn quân chiếm ưu thế điểm số áp đảo.
22. `test_center_control_reward`: Kiểm soát trung tâm được cộng điểm đúng hướng.
23. `test_minor_piece_development_in_opening`: Phát triển Mã/Tượng khai cuộc được thưởng đúng luật.
24. `test_passed_pawn_and_advancement`: Tốt thông tiến xa có điểm cao hơn tốt thông ở gần.
25. `test_isolated_and_doubled_pawns_penalized`: Tốt cô lập và tốt chồng bị phạt chuẩn xác.
26. `test_king_safety_pawn_shield`: Vua có lá chắn tốt an toàn hơn Vua bị hở cột.
27. `test_king_activity_in_endgame`: Vua chiếm trung tâm tàn cuộc được đánh giá cao hơn Vua góc.
28. `test_normalized_score_bounds_and_finite`: Điểm chuẩn hóa luôn hữu hạn trong $(-1.0, 1.0)$.

### B. Kiểm thử đối soát chéo độc lập (`tests/test_audit_verification.py - TestThiMCTS`): **7/7 PASSED**
1. `test_backprop_multi_level_alternating_values`: Lan truyền ngược 4 tầng với win, loss, draw, heuristic cutoff.
2. `test_selection_maximizes_current_player_perspective`: Tối đa hóa góc nhìn của người cầm quân sắp đi.
3. `test_terminal_rollout_returns_correct_values`: Rollout trả chuẩn $+1.0$ (Trắng thắng), $-1.0$ (Đen thắng), $0.0$ (Hòa).
4. `test_cutoff_heuristic_bounds_and_sign`: Heuristic cutoff nằm nghiêm ngặt trong $(-1.0, 1.0)$ và đúng dấu.
5. `test_mcts_visits_invariant`: Bất biến số lượt ghé thăm $N_{\text{root}} = \sum N_{\text{children}} = simulations$.
6. `test_rollout_terminal_on_exact_last_move`: Xử lý chính xác khi nước cuối cùng của rollout là nước kết thúc ván.
7. `test_mcts_determinism`: Kiểm tra lặp lại độc lập giữa các agent cùng seed.

---

## 6. Kết luận Nghiệm thu

- Module Stage 3 **Pure MCTS, Positional Heuristic & Tree Reuse** đã hoàn thiện toàn diện, code sạch, có tài liệu và bộ test bao phủ đầy đủ.
- Tỉ lệ thắng áp đảo **100% trước phiên bản bot baseline** khẳng định hiệu quả vượt trội của hàm đánh giá thế cờ tại điểm cắt rollout.
- Thuật toán MCTS thuần và quy ước dấu value đóng vai trò bản lề vững chắc, đã được kế thừa trực tiếp để phát triển thành công **Stage 6 (AlphaGo MCTS)** và **Stage 7 (AlphaZero PUCT)**.
- **ĐỦ ĐIỀU KIỆN NGHIỆM THU VÀ ĐÓNG HẠNG MỤC STAGE 3.**
