# Benchmark toàn bộ bot cờ vua

## Phương án còn 8 giờ

Không sửa thông số của đợt `classic_200` đang chạy. Dừng bằng Ctrl+C, chờ export xong, rồi dùng runner rút gọn độc lập:

```powershell
python "thực nghiệm/chess/run_budget.py" plan --hours 7
python "thực nghiệm/chess/run_budget.py" run --hours 7
python "thực nghiệm/chess/run_budget.py" report
python "thực nghiệm/chess/run_budget.py" verify
```

Mục tiêu rút gọn mặc định: 11 bot × 55 cặp × 10 ván/cặp = 550 ván, 5 Trắng/5 Đen cho mỗi bot ở từng cặp; chỉ dùng 5 khai cuộc đầu đã chốt. Depth, simulations, giới hạn 300 ply, seed và checkpoint giữ như `experiment_with_checkpoints.json`. Đây là thay đổi cỡ mẫu thực nghiệm, phải ghi trong báo cáo; không được gọi là hoàn thành kế hoạch 200 ván/cặp và không đủ căn cứ để khẳng định chắc chắn thứ hạng.

Runner đọc đợt cổ điển ở chế độ chỉ đọc, kiểm tra code/environment/seed/khai cuộc/cấu hình tương thích trước khi tái sử dụng. Chỉ lấy những ván thuộc 5 khai cuộc đầu và có đủ hai màu. Bản sao tái sử dụng giữ `origin` với run ID và đường dẫn nguồn; không đổi/xóa dữ liệu gốc. Kết quả neural từ máy khác vẫn giữ báo cáo riêng, không âm thầm gộp số đo thời gian vào đợt local. Có thể chỉ định nguồn khác bằng `--reuse path`; nguồn không tương thích sẽ bị từ chối.

Đợt mới lưu tại `results/reduced_8h/`. Runner chạy lần lượt hai ván đổi màu ở mỗi cặp, ưu tiên các cặp giữa nhóm cổ điển và học máy, rồi đi sang khai cuộc tiếp theo. Hai màu được ghi cùng một transaction. Nếu hết thời gian hoặc Ctrl+C giữa một cặp, phần cặp chưa đủ hai màu bị bỏ khỏi bộ rút gọn; không xử thua bot vì hết ngân sách toàn thí nghiệm.

`--hours 7` là tối đa cho việc thu thập trong lần gọi, gồm thời gian chuẩn bị và giới hạn worker. Sau đó còn thời gian xuất báo cáo/đóng process, nên cần chừa khoảng một giờ trước hạn nộp. Không bảo đảm thu thập đủ 550 ván. `progress.json` báo `pairs_with_results`, `expected_pairs`, `completed_games`, `expected_games`, `reused_games` và `acceptance`. Khi chưa đủ, nhãn là `PARTIAL_REDUCED_NOT_COMPLETE`; `verify` trả mã lỗi nếu chưa đủ hoặc có lỗi bot. `report` vẫn xuất dữ liệu hợp lệ của các cặp đã hoàn thành.

Có thể giảm thời gian cho phép, ví dụ `--hours 6.5`, hoặc thay cỡ mẫu trước khi bắt đầu bằng `--games-per-pair 4` (2 ván mỗi màu, 220 ván mục tiêu, cỡ mẫu rất nhỏ). Không thay cỡ mẫu trên cùng output đã tạo; dùng thư mục mới. Mỗi lần chạy lại cùng cấu hình sẽ resume và được cấp ngân sách thời gian mới theo `--hours` của lần gọi đó.

Kết quả toàn bộ ván cũ 200/cặp tiếp tục nằm ở `classic_200`; kết quả rút gọn chỉ là một bộ con cân bằng màu cộng với ván mới, không thay thế các báo cáo cũ. Đã kiểm thử tái sử dụng đủ hai màu, từ chối môi trường khác, transaction hoàn tác và ngắt worker theo deadline; không thay đổi mã nguồn đang được runner gốc băm để resume.

Bộ này độc lập với notebook ung thư vú ở thư mục cha. Chạy lệnh từ thư mục gốc `chess-multi-stage-ai`. Không chạy notebook hoặc dùng các `best_model_fold*.pt` của ung thư vú cho cờ vua.

## Cấu hình đã chuẩn bị

- 11 bot: Random, Think1, Think2, Think3, Minimax, AlphaBeta, Pure MCTS, Supervised, RL Policy, AlphaGo, AlphaZero.
- 55 cặp, 200 ván/cặp, tổng **11.000 ván**. Mỗi bot đấu tổng cộng 2.000 ván.
- Mỗi cặp có 100 ván mỗi cách phân màu, chạy hai ván đổi màu liên tiếp trên cùng thế xuất phát.
- `openings_100.json`: 100 vị trí khác nhau sau 8 ply, Trắng tới lượt. Các tiền tố khai cuộc lấy từ script Stage 6–7 của project rồi nối nước hợp lệ không bắt quân/không chiếu bằng seed 20261120. Đây là tập tự tạo cố định, chưa được chuyên gia thẩm định; không đại diện toàn bộ khai cuộc.
- Minimax depth 2; AlphaBeta depth 2, quiescence 6; Pure MCTS/AlphaGo/AlphaZero 100 simulations. AlphaZero tắt noise, temperature 0.
- Giới hạn 300 ply **sau khai cuộc**; tự nhận hòa theo chính sách `claim_draw=True` của project.
- CPU, Torch 2 luồng; giới hạn 120 giây cho mỗi lần tính nước, 300 giây khởi tạo worker.
- Seed của một bot được tính từ seed gốc, ID bot và chỉ số khai cuộc; cùng seed cho hai ván đổi màu. Bot tất định có thể vẫn lặp ván. Không coi 200 ván là 200 mẫu độc lập.

Sửa `experiment_cfg.json` nếu cần **trước khi chạy chính thức**. Resume từ chối khi code, cấu hình, phiên bản môi trường, khai cuộc hoặc nội dung checkpoint thay đổi. Khi thay đổi, dùng một thư mục output mới. Cấu hình này là một mức ngân sách cụ thể, không khẳng định các thuật toán có cùng chi phí tính toán.

## 1. Chuẩn bị môi trường và trọng số

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_all_models.py plan
```

Đặt các checkpoint **cờ vua đã huấn luyện** tại những đường dẫn sau, hoặc sửa chúng trong `experiment_cfg.json`:

| Bot | Đường dẫn mặc định |
|---|---|
| Supervised | `checkpoints/stage4/best_lichess.pt` |
| RL Policy | `checkpoints/stage5/paper_frozen_2073/last.pt` |
| AlphaGo Policy | `checkpoints/stage6/policy.pt` |
| AlphaGo Value | `checkpoints/stage6/value.pt` |
| AlphaZero | `checkpoints/stage7/best.pt` |

AlphaGo Policy có thể tham chiếu cùng checkpoint Policy Stage 5 nếu đó là trọng số được nhóm chọn. Value phải là checkpoint riêng của `ValueNetwork`. Không cần sao chép một file sang hai chỗ nếu cấu hình đã tham chiếu đúng.

Runner không tạo trọng số, không huấn luyện và không dùng mạng ngẫu nhiên thay file còn thiếu. `check` kiểm tra metadata, kiến trúc và nạp nghiêm ngặt; việc file có chất lượng huấn luyện tốt vẫn cần đối chiếu báo cáo của nhóm. Chỉ nạp checkpoint đáng tin cậy do nhóm cung cấp; bộ nạp project đọc định dạng PyTorch pickle.

```powershell
.\.venv\Scripts\python.exe scripts/run_all_models.py check
```

Chỉ tiếp tục khi lệnh báo `Preflight OK`. CUDA chỉ dùng khi máy và bản Torch hỗ trợ, sau khi đổi `device` trong cấu hình; không đổi thiết bị giữa một đợt đang chạy.

## 2. Chạy thử không cần checkpoint để kiểm tra máy

```powershell
.\.venv\Scripts\python.exe scripts/run_all_models.py smoke --agents random think1 --output results/all_models_demo
.\.venv\Scripts\python.exe scripts/run_all_models.py verify --output results/all_models_demo
```

Lệnh này chỉ có hai bot cổ điển, hai ván, tối đa 8 ply sau khai cuộc. Dùng để kiểm tra cơ chế chạy/lưu dữ liệu, không dùng để kết luận sức mạnh AI.

## 3. Chạy thử toàn bộ 11 bot sau khi đủ checkpoint

```powershell
.\.venv\Scripts\python.exe scripts/run_all_models.py smoke
.\.venv\Scripts\python.exe scripts/run_all_models.py verify --output results/all_models_200_smoke
```

Smoke có 110 ván (2 ván/cặp), giới hạn 8 ply, dùng đúng depth/simulations/model của cấu hình chính thức. Không thay đổi file cấu hình gốc. Smoke giúp bắt lỗi kết nối và đo sơ bộ thời gian tính nước; độ dài ván thật có thể lớn hơn nhiều.

Để ước lượng thời gian ván thật, có thể chạy chính đợt chính thức với `--limit` bên dưới rồi tiếp tục. Các ván này được giữ lại trong tổng 11.000 ván.

## 4. Chạy chính thức và resume

```powershell
# Tuỳ chọn: chạy trước 10 ván mới, cùng độ dài/ngân sách chính thức
.\.venv\Scripts\python.exe scripts/run_all_models.py run --limit 10

# Chạy tiếp những ván còn thiếu đến đủ 11.000
.\.venv\Scripts\python.exe scripts/run_all_models.py run
```

Nếu mất điện, đóng terminal hoặc nhấn Ctrl+C, chạy lại **đúng lệnh `run` với cùng cấu hình/output**. Ván đã lưu được bỏ qua; ván bị ngắt giữa chừng sẽ chạy lại từ đầu. Không có khôi phục cây tìm kiếm giữa một ván. Không chạy hai runner trên cùng thư mục output; có khóa để ngăn ghi đồng thời.

Các bot chạy trong process con theo từng ván. Hết thời gian tính nước thì process bị kết thúc, bot đang đi bị xử thua, có cờ `timeout=true`, lý do `forfeit` và lỗi riêng. Giới hạn bao gồm overhead worker/IPC nhỏ, không phải đồng hồ cờ vua chuẩn. Hết thời gian nạp model hoặc worker chết ngoài luồng xử lý lỗi thì runner dừng, giữ các ván đã hoàn tất, để bạn sửa nguyên nhân rồi chạy tiếp. Nếu thay code/model/config để sửa lỗi, phải dùng output mới.

Giới hạn 120 giây là một phần thể thức. Nếu muốn ngân sách lớn hơn, đổi trước khi bắt đầu đợt. Không âm thầm bỏ ván thua do lỗi ra khỏi tỷ lệ; báo cáo ghi riêng các ván có lỗi để người đọc đánh giá.

## 5. Xem tiến độ và xuất báo cáo khi tạm dừng

```powershell
Get-Content results/all_models_200/progress.json

# Dùng sau khi dừng runner; tái tạo báo cáo từ SQLite, không cần checkpoint
.\.venv\Scripts\python.exe scripts/run_all_models.py report --output results/all_models_200
```

Mỗi ván hoàn thành được ghi vào SQLite bằng transaction. `progress.json` cập nhật từng ván; bảng CSV/MD cập nhật mỗi 100 ván. JSON đầy đủ và PGN được xuất khi kết thúc/tạm dừng bình thường hoặc dùng `report`. Sau mất điện, lấy SQLite làm nguồn chính và tạo lại báo cáo; `progress.json` có thể chậm một ván nếu ngắt đúng lúc ghi.

## 6. Kiểm tra nghiệm thu

```powershell
.\.venv\Scripts\python.exe scripts/run_all_models.py verify --output results/all_models_200
```

Lệnh phát lại tất cả bản ghi, đối chiếu FEN, màu, khai cuộc, kết quả và lịch đấu; yêu cầu đủ 11.000 ván. Lịch cố định bảo đảm mỗi cặp đủ 100 ván mỗi màu. Nếu có lỗi bot/timeout, dữ liệu vẫn được xuất nhưng `verify` trả mã lỗi để yêu cầu xem xét; không báo một đợt có lỗi là benchmark sạch.

## File kết quả

| File | Nội dung |
|---|---|
| `manifest.json` | Cấu hình đã chốt, 100 khai cuộc, SHA-256 code/checkpoint và phiên bản môi trường |
| `history.sqlite3` | Nguồn bản ghi chính; tương thích cấu trúc lịch sử của GUI |
| `progress.json` | Số ván đã hoàn thành/tổng số và trạng thái |
| `run.log` | Tiến độ các ván đã lưu |
| `leaderboard.csv`, `leaderboard.md` | Thành tích toàn giải, Elo nội bộ, thời gian trung bình |
| `matchups.csv` | Thành tích của mỗi bot trước từng đối thủ, gồm phân tách Trắng/Đen |
| `tournament_results.json` | Cấu hình, tổng hợp và đầy đủ từng ván: nước đi, thời gian, telemetry, lỗi |
| `games.pgn` | Toàn bộ ván để mở bằng phần mềm cờ vua; bắt đầu từ FEN sau khai cuộc. JSON lưu cả chuỗi khai cuộc |

Các cột có hậu tố `_pct` là **phần trăm 0–100**, khác các báo cáo Arena cũ lưu tỷ lệ 0–1. Thắng % = W/N × 100; điểm % = (W+D/2)/N × 100. `white_win_pct`, `black_win_pct` là tỷ lệ thắng khi cầm mỗi màu. `error_games`/`timeout_games` trong bảng mỗi bot là số ván bot **tham gia có lỗi**, không có nghĩa chính bot đó gây lỗi; xem bản ghi để biết bên phạm lỗi. Elo khởi tạo 1200, K=32, tính theo thứ tự lịch cố định; không phải Elo FIDE.

Khi chưa chơi ván nào ở một màu, tỷ lệ màu đó là `null`/ô trống. Kết quả đang chạy chưa thể hiện cân bằng màu hoàn chỉnh. Không dùng báo cáo smoke làm báo cáo chính thức.

## Chạy một nhóm bot riêng

```powershell
.\.venv\Scripts\python.exe scripts/run_all_models.py run --agents random think1 think2 think3 --output results/classic_stage1_200
```

Dùng ID trong cấu hình, không phải tên class. Luôn dùng output riêng cho nhóm khác hoặc ngân sách khác. Lượt này là một thí nghiệm riêng, không tự cộng vào đợt 11.000 ván.

## Xác minh bộ runner

### Rút gọn phần cổ điển còn thiếu và đấu chéo trong 9 giờ

Nếu đã có kết quả nội bộ 4 bot checkpoint, dùng lịch sau để chỉ chạy nội bộ 7 bot cổ điển và 28 cặp đấu chéo. Không chạy lại 6 cặp nội bộ checkpoint.

1. Nhấn Ctrl+C ở runner đang chạy, đợi terminal trở về dấu nhắc. Giữ nguyên thư mục kết quả cũ.
2. Xem kế hoạch và chạy:

```powershell
python "thực nghiệm/chess/run_cross.py" plan --scope classic-cross --games-per-pair 10 --hours 9 --output results/remaining_classic_cross_9h_10
python "thực nghiệm/chess/run_cross.py" run --scope classic-cross --games-per-pair 10 --hours 9 --output results/remaining_classic_cross_9h_10
```

Mục tiêu của bộ rút gọn là 49 cặp × 10 = 490 ván: 210 nội bộ cổ điển và 280 đấu chéo. Mỗi cặp có 5 ván mỗi màu, trên cùng 5 khai cuộc đầu của lịch gốc. Script tự chọn các cặp đổi màu hoàn chỉnh, tương thích code/config/môi trường từ `results/classic_200` và `results/reduced_9h_10` nếu có; loại bản trùng và giữ nguyên dữ liệu nguồn. Với mốc cũ 2.466/4.200, thường tái sử dụng được 130 ván trong bộ rút gọn, cần chạy thêm 80 cổ điển + 280 đấu chéo = 360 ván. Dòng `reusable` của lệnh `plan` cho con số thực tế.

Lịch chạy phủ một khai cuộc cho các cặp trước rồi mới sang khai cuộc tiếp theo. Giữ nguyên depth, simulations và max_plies; không thay đổi thể thức giữa kết quả cũ và mới. Giới hạn 9 giờ tính từ mỗi lần gọi `run`; hết giờ dừng worker, xuất báo cáo và chỉ nhận các cặp đủ hai màu. Ván dở vì hết tổng ngân sách không bị tính thua. Không bảo đảm hoàn thành 490 ván trong thời hạn; thiếu thì báo `PARTIAL`. Xuất báo cáo có thể cần thêm thời gian sau khi ngừng đánh.

```powershell
python "thực nghiệm/chess/run_cross.py" report --output results/remaining_classic_cross_9h_10
python "thực nghiệm/chess/run_cross.py" verify --output results/remaining_classic_cross_9h_10
```

Chạy hai lệnh này sau khi runner dừng. `verify` chỉ đạt khi đủ lịch rút gọn và không có lỗi bot. Kết quả ở output riêng gồm SQLite, PGN, JSON từng ván, bảng thắng/hòa/thua và phần trăm theo màu. Có thể tiếp tục kế hoạch 4.200 ván cũ bằng đúng lệnh và output cũ; bộ rút gọn không thay thế hoặc xóa các ván chính thức đã đánh. Thêm `--scope cross` và output riêng nếu chỉ muốn 28 cặp đấu chéo.

### Checkpoint đã bổ sung và nhóm chạy riêng

`experiment_with_checkpoints.json` giữ ngân sách benchmark như cấu hình gốc nhưng trỏ RL Policy tới `checkpoints/stage5/last.pt` và AlphaZero tới `checkpoints/stage7/stage7/best.pt`, theo vị trí file hiện có. Cấu hình này không thay đổi đợt `classic_200` đã chạy.

```powershell
python scripts/run_all_models.py check --config "thực nghiệm/chess/experiment_with_checkpoints.json" --agents supervised rl_policy alphago alphazero
python scripts/run_all_models.py smoke --config "thực nghiệm/chess/experiment_with_checkpoints.json" --agents supervised rl_policy alphago alphazero --output results/neural_smoke
python scripts/run_all_models.py verify --output results/neural_smoke
python scripts/run_all_models.py run --config "thực nghiệm/chess/experiment_with_checkpoints.json" --agents supervised rl_policy alphago alphazero --output results/neural_200
```

Nhóm bốn bot có 6 cặp = 1.200 ván chính thức, không trùng 21 cặp của nhóm bảy bot cổ điển. Hai nhóm có output riêng nên có thể chạy cùng lúc, nhưng chia sẻ CPU/RAM; số đo thời gian và nguy cơ timeout bị ảnh hưởng. Hai nhóm cộng lại 5.400 ván, vẫn thiếu 28 cặp đối đầu giữa nhóm cổ điển và nhóm học máy (5.600 ván). Runner chưa tự gộp hai thư mục thành một giải 11 bot; không dùng tổng Elo của hai đợt làm Elo chung.

Đã chạy 32 test liên quan: 7 test benchmark mới, 18 test dịch vụ web và 7 test Arena. Bao gồm lịch 11.000 ván, cân bằng màu/seed, chạy dở rồi tiếp tục, chặn resume sai fingerprint, phát lại FEN, xuất PGN, tính tỷ lệ và kết thúc một worker Minimax depth 6 vượt ngân sách 1 giây trong bài test. Kiểm tra kiểu runner bằng `ty` đạt.

Đã chạy riêng CLI với Random–Think1 trong thư mục tạm: giới hạn một ván → chạy lại hoàn tất ván thứ hai → `verify` trả `complete=true`, `clean=true`. Chưa chạy đợt 11 bot hoặc xác nhận các checkpoint thật vì năm file trọng số cấu hình mặc định chưa có trên máy.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_all_models_benchmark.py tests/test_web_services.py tests/test_arena.py -q
.\.venv\Scripts\ty.exe check scripts/run_all_models.py
```


## Optional per-move time limits

Use `--move-time-limit 10` (or 15) with `run_all_models.py`, `run_budget.py`, or `run_cross.py`. Alternatively use `experiment_timed.json`, which sets `move_time_limit_seconds` to 10 and points to the local trained checkpoints. Omit the field/flag or set the JSON field to null for the original unrestricted search. Depth/simulation caps remain in effect; fast agents return early.

Timed search publishes its best-so-far recommendation in RAM. It does not write a history of search nodes to disk. At the deadline, use a completed depth (Minimax/Alpha-Beta) or a completed-simulation recommendation (MCTS); use a legal fallback if none exists. Reaching the think-time limit is not a timeout loss. Real bot errors and the original worker-health watchdog remain separate. Model startup/recovery is outside the move's think clock.

Per-move JSON telemetry includes `time_limit_seconds`, `time_exhausted`, `fallback_used`, `hard_stop`, `completed_depth`, and `simulations_completed`. Read these alongside actual `seconds`; process shutdown/IPC and GUI refresh can add a small observable overhead. Timed AlphaZero is competitive robust-child selection; its existing unrestricted and training probability APIs are unchanged.

Always use a new output directory for a changed budget. Timed budget/cross runs do not automatically import old unrestricted records. Explicit reuse validates source, environment, checkpoints, openings, and the time budget. Preserve all old results and fingerprints. To resume an old run frozen before this feature, use source revision `52fca94558a73579dd06d60bf3a99befd5acb017` with its original configuration/environment, rather than editing its manifest.

```powershell
python scripts/run_all_models.py smoke --config "thực nghiệm/chess/experiment_timed.json" --agents random alphabeta --output results/timed_smoke_10s
python scripts/run_all_models.py verify --output results/timed_smoke_10s
python "thực nghiệm/chess/run_cross.py" run --scope classic-cross --games-per-pair 10 --hours 9 --move-time-limit 10 --output results/classic_cross_timed_10s
```
