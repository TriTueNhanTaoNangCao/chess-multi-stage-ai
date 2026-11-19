# Hướng dẫn sửa lại và chạy lại: AlphaGo (Stage 6) vs AlphaZero (Stage 7)

Repo: chess-multi-stage-ai (commit c67497a). Ngày kiểm tra: 19/11/2026.

## 1. Vì sao phải làm lại

Kết quả 200 ván hiện có (`results/stage6_vs_stage7`) không dùng được làm kết quả chính, vì:

1. Ghi "30 sims" nhưng thực tế chạy 15 sims. Trong log, `root_visits` luôn bằng 15 còn `simulation_budget` ghi 30. Nguyên nhân: `scripts/benchmark_stage6_vs_stage7.py` gán `agent.simulations = 30` sau khi bộ tìm kiếm `agent.mcts` đã được tạo với 15 sims.
2. AlphaGo dùng mạng ngẫu nhiên. `checkpoints/stage6/policy.pt` và `value.pt` chỉ có các khóa tối thiểu (không có epoch, metrics). Đó là file do hàm `ensure_checkpoints` trong script cũ tự tạo.
3. AlphaZero gần như chưa học. `checkpoints/stage7/best.pt` có `step = 1`, `num_iterations = 1`, `games_per_iteration = 2`, 15 sims, tối đa 80 ply (cấu hình kiểm thử, khoảng 10 bước gradient). SHA256 `551A33E0E0CE3F85...EC53A` khác với `7cd7316e...c081e` ghi trong `docs/STAGE7_REPORT.md`.
4. Nhiễu Dirichlet chỉ bật cho AlphaZero, không bật cho AlphaGo, nên hai bên không cùng điều kiện.
5. Máy chạy CPU (`device="cpu"` mặc định), GPU không được dùng.
6. Một số mô tả sai trong script: "Luật 200 nước" không phải luật FIDE (đó là giới hạn max_plies, xử hòa); "TCEC Opening Book" thực ra là danh sách tự soạn 50 khai cuộc; "200 thế cờ hoàn toàn khác biệt" không đúng (mỗi khai cuộc dùng 4 ván).
7. Mã huấn luyện Stage 7 chỉ lưu checkpoint ở cuối và không ghi log từng vòng, nên không vẽ được đường cong học tập.

## 2. File code cần dán vào đâu

| File tôi đã gửi | Dán/đè vào đường dẫn trong repo | Công dụng |
| --- | --- | --- |
| `benchmark_stage6_vs_stage7.py` | `scripts/benchmark_stage6_vs_stage7.py` (đè file cũ) | Sửa lỗi 15 sims, bỏ tạo checkpoint ngẫu nhiên, tắt nhiễu không đối xứng, thêm `--device`, `--az-noise`, `--allow-untrained`, ghi SHA256 checkpoint, thêm Wilson 95% |
| `stage7_train_alphazero.py` | `training/stage7_train_alphazero.py` (đè file cũ) | Ghi `training_log.csv`, lưu checkpoint mỗi vòng, thêm `--device`, `--max_plies` |

Hai file `.patch` đi kèm chỉ để xem phần thay đổi. Dùng file `.py` đầy đủ là đủ.

Không push các file `.pt` lên git (thư mục `checkpoints/` chỉ có `.gitkeep`).

## 3. Các bước làm lại

### Bước 1. Lập danh sách checkpoint đang có

```
Get-ChildItem checkpoints -Recurse -Filter *.pt | Select-Object FullName, Length, LastWriteTime
```

### Bước 2. Sửa checkpoint của AlphaGo

1. Policy: chép checkpoint Stage 4 vào chỗ AlphaGo đọc.

```
Copy-Item checkpoints\stage4\best_lichess.pt checkpoints\stage6\policy.pt -Force
```

2. Value: lấy checkpoint value đã huấn luyện của Stage 5 (hỏi bạn phụ trách Stage 5; có thể là `best.pt` của thư mục huấn luyện value, hoặc kết quả Kaggle `stage5_alphago_value_2071`), chép vào `checkpoints\stage6\value.pt`.
3. Kiểm tra lại các file đã chép có khóa huấn luyện (epoch, metrics...):

```
python -c "import torch; [print(f, list(torch.load(f, weights_only=False).keys())) for f in ['checkpoints/stage6/policy.pt','checkpoints/stage6/value.pt']]"
```

4. Nếu không có value đã huấn luyện: chạy benchmark với `--allow-untrained` và ghi rõ trong báo cáo rằng AlphaGo dùng value ngẫu nhiên.

### Bước 3. Huấn luyện lại Stage 7 (chạy qua đêm)

1. Sao lưu `checkpoints\stage7\best.pt` hiện tại (quá trình huấn luyện sẽ ghi đè).
2. Đo thử một vòng để ước lượng thời gian:

```
python training\stage7_train_alphazero.py --iterations 1 --games 4 --simulations 30 --device cpu --checkpoint_dir checkpoints\stage7_test
```

Thử thêm với `--device cuda`, chỉ dùng GPU nếu nhanh hơn.

3. Chạy thật (ví dụ 12 vòng x 20 ván x 30 sims, khoảng 3 giờ, điều chỉnh theo thời gian đo ở trên):

```
python training\stage7_train_alphazero.py --iterations 12 --games 20 --simulations 30 --device cpu --checkpoint_dir checkpoints\stage7
```

4. Kết quả: `checkpoints\stage7\best.pt` và `checkpoints\stage7\training_log.csv`. Push file log, không push `.pt`.

Lưu ý trung thực: ở quy mô này nhiều ván tự chơi sẽ hòa do chạm 80 ply, nên mạng cải thiện rất ít. Đó là một checkpoint huấn luyện thật ở quy mô nhỏ cùng đường cong học tập thật, không phải AlphaZero mạnh. Báo cáo phải viết đúng như vậy.

### Bước 4. Chạy lại benchmark

1. Chạy thử 2 ván:

```
python scripts\benchmark_stage6_vs_stage7.py --games 2 --max-plies 200 --simulations 30 --threads 2 --output-dir results\test_2van
```

Phải thấy 3 dòng checkpoint kèm SHA256, không có dòng "THIẾU", không có lỗi "Số mô phỏng thực tế ... khác ngân sách khai báo".

2. Chạy chính thức (đề yêu cầu tối thiểu 100 ván mỗi cặp, khuyến nghị 200):

```
python scripts\benchmark_stage6_vs_stage7.py --games 200 --max-plies 200 --simulations 30 --threads 2 --output-dir results\stage6_vs_stage7_30sims
```

Ước lượng khoảng 7 giờ trên CPU (100 ván khoảng 3,5 giờ).

3. Phân tích phụ (làm sau): chạy lại với `--max-plies 300`, 60 đến 100 ván, thư mục kết quả riêng.

### Bước 5. Push

- Push: script đã sửa, `training_log.csv`, thư mục kết quả mới (`leaderboard.md`, `leaderboard.csv`, `tournament_results.json`, `history.sqlite3`).
- Không push: file `.pt`.
- Thư mục kết quả cũ `results/stage6_vs_stage7`: giữ lại nếu muốn nhưng ghi rõ là bản "15 sims, mạng chưa huấn luyện", không dùng làm kết quả chính.

## 4. Việc cần sửa trong báo cáo sau khi có kết quả mới

- Chương 4: bảng thắng/hòa/thua, score và khoảng tin cậy Wilson 95%, ms/nước và p95, thống kê loại kết thúc, phân bố độ dài ván, đường cong học tập từ `training_log.csv`.
- Phụ lục C: cập nhật SHA256 của `best.pt` theo file cuối cùng.
- Chương 3: nếu đổi giới hạn ply hoặc số sims so với thiết kế (T, seed, 500 thế cờ), ghi cho khớp với những gì đã chạy.
- Chương 5 (hạn chế): quy mô huấn luyện Stage 7, ván tự chơi nhiều hòa do giới hạn 80 ply, 30 sims, nguyên nhân hòa nhiều.
- Không kết luận AlphaGo hay AlphaZero mạnh hơn nếu khoảng tin cậy Wilson của score chứa 0,5.
