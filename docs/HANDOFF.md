# Hợp đồng bàn giao Stage 3–7

Đây là quy ước triển khai tiếp theo. Các agent/model Stage 3–7 vẫn là skeleton; chưa có checkpoint thật hoặc bộ nạp model trong GUI/Arena hiện tại.

## 1. Board, encoder và dấu value

Agent nhận `ChessBoard`; encoder chấp nhận cả wrapper này và `chess.Board`:

```python
features = ChessEncoder.encode_board(board)        # HWC float32 (8,8,14)
tensor = ChessEncoder.encode_board_tensor(board)   # NCHW (1,14,8,8)
mask = ChessEncoder.get_legal_action_mask(board)   # (4672,)
```

- Kiểm tra `board.is_terminal()` trước khi chạy model/search. Mask và decoder trả rỗng/None ở terminal.
- Với wrapper, encoder kế thừa `board.claim_draw`. Với raw board, mặc định `claim_draw=True`; muốn tiếp tục trước hòa tự động thì truyền `claim_draw=False` cho mask/indices/decoder.
- Tensor theo góc nhìn Trắng cố định; action index lật rank khi Đen đi. Không dùng nhầm cách định hướng của hai biểu diễn.
- `encode_state(board)` giữ root FEN, move stack, FEN reset và draw policy. `decode_state(state)` trả raw board; `decode_state(state, as_wrapper=True)` khôi phục wrapper cùng policy và vị trí reset.
- Metadata đi kèm để tái dựng luật; tensor 14 kênh riêng lẻ chưa chứa đủ thông tin lịch sử/quyền nhập thành để làm state Markov đầy đủ. Nếu nâng encoder, tăng version và huấn luyện lại.

Quy ước MCTS/value: value của node theo **người đang tới lượt tại chính node đó**.

```text
Parent: Đen sắp đi
    Đen đi Qh4#
Child: Trắng sắp đi, đã bị chiếu hết
    child.get_result_relative() = -1
    value nhìn từ parent = -(-1) = +1
```

Backprop đổi dấu qua mỗi cạnh cha/con. Nếu child lưu Q theo bên sắp đi của child, khi parent chọn nước phải dùng `-Q_child` cộng exploration. Value target của một state là `winner_white * current_player_at_state`; hòa bằng 0. Không đổi dấu hai lần. Thi phải thêm test thắng/thua/hòa và ít nhất cây sâu 2 cạnh trước khi export MCTS.

## 2. Trách nhiệm model và thứ tự bàn giao

| Hạng mục | Người viết/chịu trách nhiệm | Người phối hợp | Điều kiện bàn giao |
| :--- | :--- | :--- | :--- |
| `models/cnn_policy.py` | Tùng | Thái định nghĩa nhu cầu SL | Forward NCHW14 → logits (N,4672), test shape và gradient |
| `models/value_net.py` | Tùng | Thái định nghĩa baseline/target RL | Output (N,1), tanh, góc nhìn side-to-move |
| `models/dual_head_resnet.py` | Tùng | Thái góp ý dữ liệu | Policy logits + value, cùng encoder version |
| PGN/dataset, SL/RL loops | Thái | Tùng cung cấp network | Split theo ván, legal mask, seed và validation |
| UCB1 MCTS | Thi | Thắng review contract | Legal move, history, dấu value, seed |
| AlphaGo/PUCT, AlphaZero loop | Tùng | Thi/Thái | Tích hợp các hợp đồng đã kiểm thử |

**CNNPolicyNetwork và ValueNetwork phải được Tùng dựng trước khi Thái tích hợp training Stage 4–5.** Phần model hỗ trợ này không chờ tới Stage 6. Thái có thể làm parser PGN, dataset và vòng training với model giả trong unit test trong thời gian chờ. Nếu Thái nhận làm thay CNN/Value, nhóm cần ghi nhận chuyển trách nhiệm trong PR; không tạo bản mạng trùng ở thư mục agent.

Mỗi bàn giao model gồm: constructor/config, forward contract, test shape/gradient, encoder version và ví dụ một batch. Reviewer là người sử dụng module ở stage kế tiếp.

## 3. Quy ước checkpoint .pt

Tên gợi ý:

```text
checkpoints/stage4_cnn_hwc14-v1_seed42_step010000.pt
checkpoints/stage5_policy_hwc14-v1_seed42_step020000.pt
checkpoints/stage7_resnet_hwc14-v2_seed42_step050000.pt
```

Tên hỗ trợ tìm kiếm; **metadata trong checkpoint mới là nguồn xác định cấu hình**. Schema thống nhất:

```python
checkpoint = {
    "format_version": 1,
    "stage": 4,
    "model_class": "CNNPolicyNetwork",
    "model_config": {"in_channels": 14},  # thêm width/layers theo model thật
    "encoder_version": "hwc14-white-v1",
    "action_version": "az73-rankflip-v1",
    "num_actions": 4672,
    "value_perspective": "side_to_move",
    "claim_draw": True,
    "seed": 42,
    "step": 10000,
    "training_config": {},              # optimizer, lr, batch size, loss, ...
    "data_split": {},                   # PGN/source, split theo game, fingerprint
    "metrics": {},                      # validation metrics; không tự bịa Elo
    "state_dict": model.state_dict(),
    # "optimizer_state_dict": optimizer.state_dict(),  # khi cần resume
}
torch.save(checkpoint, path)
```

Lưu state_dict và metadata dạng primitives/tensors; không pickle toàn bộ class model. Khi nạp: dùng checkpoint đáng tin cậy, map_location phù hợp, kiểm tra format/encoder/action/model version và shape; gọi `load_state_dict(..., strict=True)`, `eval()`, inference không gradient. Không im lặng thay model bằng bot Random nếu file thiếu/sai.

## 4. Tích hợp checkpoint vào factory, GUI và Arena

**Công việc của người triển khai Stage 4–7 trước khi đưa agent vào registry:**

1. Constructor nhận `checkpoint`, `device`, `seed`; validate/load đúng schema ở trên.
2. Mở rộng `build_agent(key, depth=None, seed=None, *, checkpoint=None, device="cpu")` và chuyển đúng kwargs tới class tương ứng. Các bot cổ điển không được nhận nhầm checkpoint.
3. Thêm `--checkpoint` vào main.py, giữ đường dẫn khi restart/undo; khi chuyển loại bot phải chọn checkpoint tương ứng hoặc báo chưa có model.
4. Cho Arena nhận config riêng từng participant: agent key, checkpoint path/hash, seed và search budget. Lưu toàn bộ config vào JSON để tái lập.
5. Test file thiếu, encoder sai version, sai architecture, legal masking và nạp CPU; sau đó mới thêm registry và công bố bot có thể chơi.

Chữ ký factory ở mục này là **thiết kế cần thực hiện khi có model**, chưa phải API đã hỗ trợ. Factory hiện tại chỉ nhận key/depth/seed và chạy các bot cổ điển. Không có tham số `--checkpoint` giả hoạt động hoặc kết quả training giả.

## 5. Benchmark và lưu kết quả

- Số ván phải chẵn để cân bằng màu. CLI hỗ trợ cấu hình depth riêng: `--agents alphabeta:2 alphabeta:4`.
- Arena lưu `root_fen`, `initial_moves`, `claim_draw` và các nước trong ván. Dùng `restore_initial_board(record)` trước khi replay các nước mới.
- Lỗi reset một bot: `reset_forfeit`; cả hai cùng lỗi reset: `double_forfeit`, tính 0.5 điểm mỗi bên và ghi rõ lý do. Đây là adjudication của Arena, không phải hòa theo luật.
- ms/move gồm clone state, gọi act và kiểm tra move; không gồm reset, render hoặc ghi file. Không so trực tiếp với latency neural forward riêng lẻ.
- Nếu đổi heuristic/model/search budget, chạy benchmark mới; giữ cấu hình và cập nhật bảng README từ dữ liệu thực.
