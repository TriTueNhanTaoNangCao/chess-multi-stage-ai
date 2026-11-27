# Nhiệm vụ Stage 4 và Stage 5

Tài liệu tổng hợp phần Policy học giám sát, Policy học tăng cường và Value Network.
Các số liệu thực nghiệm và thời gian Kaggle dưới đây do người thực hiện cung cấp;
chưa được đối chiếu lại toàn bộ với artifact trong lượt lập tài liệu này.
**Chưa xác minh** nghĩa là chưa có thông tin đủ để điền, không phải giá trị bằng 0.

## Stage 4 — Policy học giám sát

| Hạng mục | Thông tin |
| --- | --- |
| **Nguồn dữ liệu** | Lichess Elite tháng **01, 02, 03/2024** |
| **Số ván** | **300.000**, mỗi tháng 100.000 ván |
| **Lọc ván** | Cờ tiêu chuẩn, xuất phát từ bàn chuẩn, có kết quả hợp lệ. Loại bullet, ván lỗi và ván trùng chuỗi nước UCI |
| **Lấy mẫu** | Ngẫu nhiên tối đa **32 thế cờ/ván** |
| **Nhãn** | Nước đi thực tế trong PGN, kèm legal mask |
| **Chia tập** | Theo ván, **90% train / 10% validation** trong từng tháng |
| **Train** | **270.000 ván — 8.572.676 mẫu** |
| **Validation** | **30.000 ván — 952.874 mẫu** |
| **Tổng mẫu** | **9.525.550** |
| **Đầu vào** | Tensor **14 × 8 × 8**, hướng Trắng cố định |
| **Đầu ra** | **4.672 điểm hành động**, áp legal mask |
| **Kiến trúc** | 3 khối Conv 3×3–BatchNorm–ReLU, **128 kênh/khối**. Flatten → FC 512 → ReLU → Dropout 0,2 → FC 4.672 |
| **Loss** | Legal-masked cross-entropy |
| **Optimizer / learning rate** | Adam / **`1e-3`** |
| **Batch size / seed** | **64 / 42** |
| **Ngân sách huấn luyện** | Tối đa **20 epoch**, early stopping **patience 3** |
| **Tiêu chí chọn checkpoint** | **Validation loss thấp nhất** |
| **GPU** | **Một NVIDIA Tesla T4**, qua device CUDA mặc định |
| **CPU** | Chưa xác minh |

| Kết quả và thời gian | **Lần chạy mới** |
| --- | --- |
| Số epoch hoàn thành | **7** |
| Epoch được chọn | **4** |
| Validation loss tại checkpoint chọn | **1,97981** |
| Validation top-1 | **40,70%** |
| Validation top-5 | **77,57%** |
| Thời gian tải và giải nén PGN | Chưa xác minh riêng |
| Thời gian chuẩn bị dataset | **Khoảng 39 phút 35 giây**, từ lúc có PGN đến khi xuất mẫu xong |
| Thời gian huấn luyện | **Khoảng 1 giờ 26 phút 37 giây**, gồm huấn luyện và validation qua 7 epoch |
| Tổng thời gian | **Khoảng 2 giờ 6 phút 52 giây**, từ đầu phiên đến kiểm tra checkpoint xong |
| GPU / CPU của lần chạy | **Tesla T4 / CPU chưa ghi nhận** |
| Dừng sớm | Sau **3 epoch liên tiếp** không cải thiện validation loss |

*Kết quả lần chạy lại được cập nhật theo thông tin log do người thực hiện cung cấp.
Chưa đối chiếu định danh/hash checkpoint mới với checkpoint bàn giao bên dưới;
không mặc định checkpoint local đã được thay thế.*

## Stage 5 — B16 và Paper Policy

| Hạng mục | **B16** | **Paper Policy** |
| --- | --- | --- |
| **Khởi tạo Policy** | Trọng số Stage 4 | Trọng số B16 |
| **Kiến trúc Policy** | Cùng CNN Stage 4 | Cùng CNN Stage 4 |
| **Phương pháp cập nhật** | REINFORCE, baseline **0**, Value disabled | REINFORCE, baseline từ **Value đóng băng** |
| **Số iteration** | **400** | **80** |
| **Ván mỗi iteration** | **16** | **32** |
| **Ngân sách số ván thử** | **6.400** | **2.560** |
| **Chunk size** | **16** | **8** |
| **Policy learning rate** | **`1e-5`** | **`1e-5`** |
| **Seed** | **42** | **2073** |
| **Nhóm đối thủ** | Pool tối đa **4** phiên bản Policy, giữ Stage 4 | Pool tối đa **4** phiên bản Policy, giữ Stage 4 |
| **Khoảng cách thêm snapshot** | Mỗi **40 iteration** | Mỗi **5 iteration** |
| **Khởi đầu / giới hạn ván** | Chuẩn / **300 ply** | Chuẩn / **300 ply** |
| **Reward thắng / hòa / thua trong code** | **+1 / 0 / −1**, theo màu của bên đi tại mỗi mẫu | **+1 / 0 / −1**; tín hiệu cập nhật trừ thêm baseline Value |
| **Tiêu chí checkpoint sử dụng** | **`last.pt` cuối quá trình** | **`last.pt` cuối quá trình** |
| **Thời gian toàn bước theo log** | Khoảng **45 phút 20 giây** | Khoảng **17 phút 33 giây** |

*Ngân sách ván thử không đồng nghĩa mọi ván đều kết thúc theo luật hoặc được dùng cập nhật.
Ván chạm trần chưa terminal được đánh dấu truncated, không gán hòa giả.*

## Stage 5 — Sinh dữ liệu và huấn luyện Value

| Hạng mục | Thông tin |
| --- | --- |
| **Policy dùng sinh dữ liệu** | Stage 4 và B16 |
| **Luồng collection** | Lấy U nguyên đều trong [1,100]; Stage 4 sampling legal-masked, temperature 1, chơi **U−1 ply** → chọn đều một nước hợp lệ → lưu thế cờ ngay sau nước đó và bên tới lượt → B16 chơi cả hai bên đến kết thúc, sampling legal-masked, temperature 1 |
| **Nhãn Value** | **+1 / 0 / −1** theo kết quả của bên tới lượt tại thế cờ đã lưu; mỗi ván tối đa một mẫu |
| **Số ván thử** | **24.576** |
| **Shard size / số shard** | **512 / 48** |
| **Mốc lấy mẫu `u`** | **1–100**, đếm ply từ đầu ván; nước ngẫu nhiên là ply thứ U |
| **Giới hạn ván** | **300 ply tổng từ bàn chuẩn**, không phải 300 ply thêm sau U |
| **Seed collection** | **2071** |
| **Số mẫu thực tế và phân chia** | **Chưa chốt**; cần đối chiếu manifest hoặc dataset |
| **Chia tập** | Train/validation/test **80/10/10 theo ván**, có kiểm tra trùng tensor giữa split |
| **Thời gian sinh và xuất dữ liệu** | Khoảng **2 giờ 46 phút 36 giây** |
| **Khởi tạo Value** | Tạo mạng mới bằng khởi tạo mặc định PyTorch, seed **2072**; không nạp trọng số Stage 4 hoặc B16 vào Value |
| **Đầu vào** | Tensor **14 × 8 × 8** |
| **Kiến trúc** | Conv 3×3 **14 → 128 → 64**, Conv 1×1 **64 → 1**. Flatten → FC **64 → 256 → 64 → 1** |
| **Activation** | BatchNorm và ReLU ở các khối convolution; ReLU sau hai FC ẩn; đầu ra **tanh** |
| **Giá trị dự đoán** | **[−1, 1]**, theo góc nhìn bên đến lượt |
| **Loss / optimizer** | MSE / Adam |
| **Learning rate / batch size** | **`1e-4` / 128** |
| **Ngân sách huấn luyện** | Tối đa **100 epoch**, patience **10** |
| **Epoch hoàn thành** | **12** |
| **Tiêu chí chọn checkpoint** | **Validation MSE thấp nhất** |
| **Epoch được chọn / validation MSE** | **2 / 0,242904** |
| **Thời gian train do trainer đo** | **9,79 giây** |
| **Thời gian toàn bước train Value** | Khoảng **13,7 giây**, gồm khởi động và kiểm tra đầu ra |
| **Cách dùng sau huấn luyện** | Giữ cố định trọng số để làm baseline cho Paper Policy |

**Chênh lệch số mẫu chưa giải quyết:** log ghi `samples=23476`, còn tổng các nhãn theo tập là **23.130**.
Chưa kết luận nguyên nhân hoặc dùng một trong hai số làm tổng mẫu chính thức khi chưa đối chiếu manifest/dataset.
Ván kết thúc trước mốc lấy mẫu hoặc truncated không được gán hòa giả để tạo mẫu.

## Tài nguyên và tổng thời gian Stage 5

| Hạng mục | Thông tin |
| --- | --- |
| **Môi trường** | Kaggle, notebook Stage 5 **Version 2**, chạy thành công theo thông tin người thực hiện cung cấp |
| **GPU được cấp** | **Tesla T4 ×2** |
| **GPU sử dụng theo code** | **Một GPU mặc định**, `device="cuda"` |
| **CPU / số core** | Chưa xác minh |
| **RAM / VRAM sử dụng thực tế** | Chưa xác minh |
| **Tổng thời gian notebook** | **3 giờ 50 phút 30,2 giây** |
| **Phạm vi tổng thời gian** | Setup, B16, collection Value, train Value, Paper Policy, đóng ZIP và hoàn tất notebook |

## Source đối chiếu và checkpoint bàn giao

| Thành phần | File / hàm |
| --- | --- |
| Chọn ván, đọc PGN và trích mẫu Stage 4 | `scripts/stage4/pgn_candidates_300k.py:ranked_offsets`, `scripts/stage4/pgn_reader_300k.py:iter_candidates`, `scripts/stage4/pgn_samples_300k.py:extract_samples` |
| Tạo dataset / train Stage 4 | `scripts/stage4/build_dataset_300k.py:build_dataset`, `scripts/stage4/train_300k.py:main`, `training/stage4_train_sl.py:train` |
| Kiến trúc Policy / Value | `models/cnn_policy.py:CNNPolicyNetwork`, `models/value_net.py:ValueNetwork` |
| Điều phối Stage 5 | `scripts/stage5/train_stage5.py:main`, `b16_config`, `paper_config` |
| Reward và cập nhật Policy | `training/stage5/selfplay.py`, `training/stage5/updates.py:reinforce_policy_update` |
| Collector / Value training | `scripts/stage5/stage5_alphago_value_data.py`, `scripts/stage5/stage5_alphago_value_train.py`, `training/stage5/value_training.py:train_value` |

Các đường dẫn checkpoint chính đã chốt:

- Stage 4: `checkpoints/stage4/stage4_policy.pt`.
- B16: `checkpoints/stage5/stage5_8h/b16/last.pt`.
- Value: `checkpoints/stage5/stage5_8h/value/best.pt`.
- Paper Policy: `checkpoints/stage5/stage5_8h/paper_policy/last.pt`.

*Danh sách trên là đường dẫn bàn giao đã thống nhất, không xác nhận file hiện đang có trên máy.
Source tái lập được hoàn thiện sau thực nghiệm; không mặc định đó là mã notebook Kaggle gốc
hoặc bảo đảm checkpoint tái lập trùng từng byte.*
