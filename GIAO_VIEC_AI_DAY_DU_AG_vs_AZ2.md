# Giao việc cho AI: làm đủ yêu cầu của thầy, trọng tâm AlphaGo (`alphago`) đấu AlphaZero v2 (`alphazero_v2`)

Repo `chess-multi-stage-ai`, chạy từ thư mục gốc, Python 3.11, Windows PowerShell (`.venv\Scripts\python.exe`). Làm T1 đến T9 lần lượt. Sau mỗi task báo đúng 3 mục: file đã đổi; lệnh và kết quả nguyên văn; vấn đề gặp. Task chỉ xong khi lệnh nghiệm thu đạt.

## Quy tắc cứng
1. Chỉ tạo hoặc sửa file nêu trong task. Script mới đặt trong `analysis/`.
2. Không sửa `agents/`, `arena/`, `games/`, `models/`, `services/`, `training/`. Ngoại lệ duy nhất: hàm `provenance()` trong `scripts/run_all_models.py` (T3). Lý do: `manifest.json` băm mọi `.py` trong `agents`, `arena`, `games`, `models`, `services`, `training`, `scripts`; sửa sau khi chạy chính thức thì runner từ chối tiếp tục.
3. Không sửa `experiment_cfg*.json` gốc và `openings_100.json`. Không huấn luyện, không tạo checkpoint giả, không tải model ngoài.
4. Không chạy trận chính thức. Không bịa số liệu: chưa biết thì ghi `[[chưa có]]`.
5. Chỉ dùng thư viện chuẩn, `numpy`, `chess`, `torch`, `psutil`. Không dùng `fork` (Windows): dùng `spawn`.

## Yêu cầu của thầy và nơi đáp ứng

| Mã | Yêu cầu | Đáp ứng bằng |
|---|---|---|
| R1 | Lưu checkpoint và seed | T1 (SHA256, metadata), manifest của runner (seed, mã băm) |
| R2 | Không chọn siêu tham số bằng ván đánh giá cuối | Mục "Quy ước chạy" (khóa tham số trước khi chạy) + người ghi tiêu chí chọn checkpoint |
| R3 | Khai báo CPU/GPU, thời gian huấn luyện | T3 (phần cứng vào manifest) + người cung cấp thời gian huấn luyện |
| R4 | Round-robin, ≥100 ván/cặp (khuyến nghị 200), chia đều màu, nhiều seed | T2, T7, T8 + P0, P3, P6 |
| R5 | Cố định game, luật, phần cứng | T2 (`max_plies`, `claim_draw`), T3, T9 |
| R6 | Ít nhất hai chế độ (ngân sách rõ ràng; cấu hình mặc định) | T2: chế độ A và B; P0, P4 |
| R7 | W/D/L, score, ms/nước, p95, nút/mô phỏng, RAM/VRAM, thời gian huấn luyện | T4, T6 |
| R8 | Ma trận đối đầu + Wilson/Bootstrap | T4 |
| R9 | Ảnh hưởng thứ tự đi, seed, độ sâu, ngân sách mô phỏng | T4 (score theo màu), T5, T8, P6 |
| R10 | Ít nhất 2 ablation | T2 (cấu hình ablation), T5; AB1 và AB2 bắt buộc |
| R11, R12 | Mục 3.6 và đầu mục huấn luyện | Người gửi ảnh yêu cầu; AI không làm được |

## Quy ước chạy (AI phải tuân theo khi soạn lệnh)
- Một cặp = 200 ván = 100 thế mở đầu × 2 màu (mỗi bên 100 Trắng, 100 Đen). Sàn 100; dưới 100 không đưa vào kết quả chính. Ablation: 100 ván mỗi điều kiện.
- `max_plies` 400 (tính sau 8 ply khai cuộc, ván dài nhất 408 ply), `claim_draw` true, `torch_threads` 2, một thiết bị (`cpu` hoặc `cuda`) cho cả arena, không đổi giữa chừng.
- Seed 42 là kết quả chính; seed 123 lặp 100 ván cho cặp trọng tâm. Agent tất định (Minimax, Alpha-Beta, AlphaZero tắt nhiễu) kiểm độ nhạy bằng `openings_100_b.json` (T8).
- Chế độ A: 30 mô phỏng cho `mcts_pure`, `alphago`, `alphazero`, `alphazero_v2`; Minimax và Alpha-Beta độ sâu 2, Alpha-Beta tắt quiescence. Chế độ B: 100 mô phỏng; Alpha-Beta độ sâu 2, quiescence độ sâu 6.
- `alphago`: λ (`rollout_weight`) 0,5, `max_rollout_depth` 15, `c_puct` 1,5. `alphazero_v2`: nhiệt độ 0, không nhiễu.
- Mỗi cặp một thư mục output riêng (`results/final_<A|B>_..._s<seed>`). `--agents` chạy mọi cặp trong nhóm được chọn, nên chia theo cặp, không chia theo nhóm.
- Ngắt thì chạy lại đúng lệnh cũ để tiếp tục (ván đang dở bị mất). Không sửa file nào trong repo và không thay checkpoint khi đang chạy; nếu runner từ chối tiếp tục vì mã băm đổi thì chạy lại cặp đó ở thư mục mới.
- Xong mỗi thư mục: `verify` phải đạt (không `error_games`, không `timeout_games`), rồi sao lưu.
- Tham số agent và checkpoint không đổi sau khi đã thấy kết quả đối kháng (R2). Mọi đợt chạy hoặc vào báo cáo, hoặc ghi "bỏ" kèm lý do.
- Độ trễ p95, ms/nước và RAM/VRAM lấy từ phép đo T6 trên máy nhàn rỗi, không lấy từ ván chạy song song.

## Sự thật về repo (AI cứ tin)
- Runner `scripts/run_all_models.py`: `plan`, `check`, `smoke`, `run`, `report`, `verify`. Config phải có đúng khóa: `schema_version`(=1), `seed`, `games_per_pair`(chẵn), `max_plies`, `claim_draw`, `device`, `torch_threads`, `move_timeout_seconds`, `startup_timeout_seconds`, `openings_file`, `output_dir`, `agents`. Mỗi agent: `{"id","key","kwargs"}`; `kwargs` không chứa `seed`, `device`, `name`, `player_id`.
- Mã ván `{a}__{b}__o{index:03d}__l{leg}`, `a` và `b` theo thứ tự trong config; `leg 0`: `a` cầm Trắng. Mã cặp `a__b`.
- Đọc ván: `from services.history import HistoryStore; HistoryStore(path).list_games()`. Mỗi ván có `pair`, `white`, `black`, `result` (`1-0`, `0-1`, `1/2-1/2`), `reason`, `opening_index`, `leg`, `moves`, `error`, `timeout` (có thể vắng), `telemetry`. Mỗi phần tử `telemetry` (một nước): `ply`, `color`, `seconds`, và tùy agent `nodes_visited`, `cutoffs`, `simulation_budget`, `root_visits`, `error`.
- Thư mục output có `manifest.json`, `history.sqlite3`, `leaderboard.csv`, `matchups.csv`, `tournament_results.json`, `games.pgn`, `run.log`, `progress.json`.
- Checkpoint cần metadata: `model_class`, `encoder_version`=`hwc14-white-v1`, `claim_draw`=True, `action_version`=`az73-rankflip-v1`, `num_actions`=4672 (trừ Value), `value_perspective`=`side_to_move` (Value, AlphaZero), `state_dict`.
- Tham số agent: `minimax(depth)`, `alphabeta(depth,use_quiescence,q_depth)`, `mcts_pure(simulations,c,max_rollout_depth)`, `alphago(checkpoint_policy,checkpoint_value,simulations,c_puct,rollout_weight,max_rollout_depth)`, `alphazero(checkpoint,simulations,temperature,add_noise)`, `supervised(checkpoint)`, `rl_policy(checkpoint)`.

## Task

**T1. `analysis/inspect_checkpoints.py`** (R1). Mặc định kiểm `checkpoints/stage4/best_lichess.pt`, `stage5/paper_frozen_2073/last.pt`, `stage6/policy.pt`, `stage6/value.pt`, `stage7/best.pt`, `stage7_v2/best.pt`. Mỗi file một dòng: tồn tại, MB, SHA256 đủ 64 ký tự, các trường metadata (đánh dấu thiếu hoặc sai), `training_config` nếu có (`seed`, `iterations` hoặc `epochs`). Nạp bằng `torch.load(path, map_location="cpu", weights_only=False)`. So SHA256 `stage6/policy.pt` với `stage5/.../last.pt` (TRÙNG hoặc KHÁC). Ghi `results/checkpoint_inventory.md`. Thiếu file thì báo "THIẾU", không dừng. Nghiệm thu: chạy không lỗi.

**T2. Cấu hình** (R4, R5, R6, R9, R10). Tạo `analysis/make_configs.py` đúng nội dung ở phụ lục, chạy `python analysis/make_configs.py` để sinh `experiment_cfg_final_A.json`, `experiment_cfg_final_B.json`, `experiment_cfg_ablation.json` trong `thực nghiệm/chess/`. File ablation có 13 agent: không chạy nguyên file (78 cặp), luôn chọn cặp bằng `--agents`. Nghiệm thu:
```
python scripts/run_all_models.py plan --config "thực nghiệm/chess/experiment_cfg_final_A.json"
python scripts/run_all_models.py plan --config "thực nghiệm/chess/experiment_cfg_final_B.json"
python scripts/run_all_models.py plan --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents alphago alphazero_v2
python scripts/run_all_models.py plan --config "thực nghiệm/chess/experiment_cfg_ablation.json" --agents alphago_l0 alphazero_v2
```
Kết quả phải là `12 agents; 66 pairs; 13200 games` (hai lệnh đầu; dòng `Missing checkpoints` là bình thường khi chưa đặt checkpoint), `2 agents; 1 pairs; 200 games`, `2 agents; 1 pairs; 100 games`. Mắt: ở A `alphabeta` có `use_quiescence:false`, các agent mô phỏng có `simulations:30`; ở B `q_depth:6`, `use_quiescence:true`, `simulations:100`; cả ba file `max_plies:400`.

**T3. `provenance()` trong `scripts/run_all_models.py`** (R3, R5). Thêm vào `env` của manifest: `cpu` (`platform.processor()`), `cpu_cores_physical` và `cpu_cores_logical` (`psutil.cpu_count`), `ram_gb` (`round(psutil.virtual_memory().total/2**30,1)`), `gpu` (`torch.cuda.get_device_name(0)` hoặc `null`), `cuda` (`torch.version.cuda`). Bọc `psutil` bằng `try/except ImportError` (ghi `null`). Giá trị chỉ là chuỗi, số hoặc `null` (runner dùng `allow_nan=False`). Không sửa chỗ khác. Làm T3 trước mọi lần chạy chính thức rồi commit. Nghiệm thu: `python scripts/run_all_models.py smoke --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents minimax alphabeta --output results/smoke_t3` chạy xong, `manifest.json` có khóa mới, `python -m pytest -q` qua.

**T4. `analysis/arena_stats.py`** (R7, R8, R9). Tham số: `--input <thư mục>...`, `--output <thư mục>`, `--bootstrap 10000`, `--bootstrap-seed 20261124`, `--allow-mixed`. Gộp nhiều thư mục: đọc `manifest.json`; nếu `max_plies`, `claim_draw`, `seed`, `games_per_pair` hoặc nội dung agent trùng ID khác nhau thì dừng báo lỗi (trừ `--allow-mixed`); một cặp xuất hiện ở hai thư mục thì dừng báo lỗi. Định nghĩa, với A là agent đứng trước trong mã cặp:
- W/D/L của A; `score=(W+0,5D)/N`.
- Wilson 95%, `z=1,959964`, `successes=W+0,5D`, `n=N`: `centre=(p+z²/(2n))/(1+z²/n)`, `half=z·sqrt(p(1−p)/n+z²/(4n²))/(1+z²/n)`, kẹp [0,1].
- Bootstrap lấy mẫu theo `opening_index` (mỗi đơn vị gồm hai ván `leg 0` và `leg 1`), có hoàn lại, `numpy.random.default_rng(bootstrap_seed)`, phân vị 2,5 và 97,5.
- Score của A khi cầm Trắng và khi cầm Đen; phân bố `reason` (chiếu hết, lặp thế, hết nước đi, chạm trần 400...); độ dài ván trung bình.
- Mỗi agent, từ `telemetry` của nước thuộc đúng màu agent đó, bỏ phần tử có `error`: ms trung bình, trung vị, p95 (`numpy.percentile(x,95,method="linear")`), `nodes_visited` và `cutoffs` trung bình nếu có, `simulation_budget` phổ biến nhất nếu có.
- Số ván có `timeout` hoặc `error`. Không tính Elo.

Đầu ra: `pairs.csv`, `agents.csv`, `matrix_score.csv`, `matrix_score.md` (ma trận agent × agent, ô hàng i cột j là score của i trước j kèm Wilson, đường chéo `-`), `summary.md` (số ván, số cặp, fingerprint đã gộp, cảnh báo "hai ván cùng thế mở đầu không độc lập; không tính Elo"). Nghiệm thu: (1) Wilson(75,100) ≈ [0,657; 0,825]; (2) chạy `smoke` cho `minimax alphabeta` vào `results/smoke_t4` rồi `arena_stats` ra đủ 5 file, `pairs.csv` có 1 dòng N=2; (3) gộp hai thư mục khác `max_plies` thì báo lỗi.

**T5. `analysis/compare_conditions.py`** (R9, R10). Tham số: `--cond nhãn=thư_mục` (lặp nhiều lần), `--output`, `--bootstrap 10000`, `--bootstrap-seed 20261124`. Mỗi thư mục là một điều kiện (một cặp); thư mục có nhiều cặp thì báo lỗi. Dùng lại hàm Wilson và Bootstrap của T4 (nhập từ `analysis.arena_stats`). Bảng `conditions.csv` và `conditions.md`: nhãn, cặp, seed, N, W, D, L, score của agent đứng trước, Wilson thấp/cao, Bootstrap thấp/cao, ms trung bình mỗi agent. Dùng cho AB1 đến AB5, seed 42 so với 123, hai bộ thế mở đầu. Nghiệm thu: hai nhãn trỏ cùng một thư mục smoke ra 2 dòng giống nhau.

**T6. `analysis/measure_resources.py`** (R7). Tham số: `--config`, `--agents` (mặc định mọi agent), `--device cpu|cuda`, `--moves 20`, `--openings 5`, `--output`. Mỗi agent một tiến trình `spawn` riêng. Trong tiến trình: ghi RSS trước nạp, dựng agent bằng `build_agent(spec, seed, device, claim_draw)` của runner, ghi RSS sau nạp, cho agent tự chơi hai bên `--moves` nước từ `--openings` thế đầu của `openings_file`, đo từng nước bằng `time.perf_counter`; RSS đỉnh = max RSS sau mỗi nước (Windows thêm `peak_wset`); `cuda` thì `torch.cuda.reset_peak_memory_stats()` trước nạp, đọc `max_memory_allocated` và `max_memory_reserved` cuối; `cpu` thì VRAM `null`. Cột: `agent,device,moves,rss_before_mb,rss_after_load_mb,rss_peak_mb,vram_alloc_peak_mb,vram_reserved_peak_mb,ms_mean,ms_p95`. Nghiệm thu: `--agents random minimax --moves 6 --openings 2` ra 2 dòng không lỗi.

**T7. `analysis/plan_pairs.py`** (R4). Tham số: `--config`, `--agents` (mặc định mọi agent), `--root`, `--split N` (số luồng chạy, mặc định 1), `--exclude A__B ...`, `--weights` (mặc định `random`,`think*`: 1; `minimax`,`alphabeta`,`supervised`,`rl_policy`: 2; `mcts_pure`: 6; `alphago`,`alphazero`,`alphazero_v2`: 8; chi phí cặp = tổng hai trọng số). Mã cặp `a__b` theo thứ tự trong config. Phân cặp cho N luồng theo tham lam (cặp nặng nhất vào luồng nhẹ nhất), ghi `run_machine_<k>.ps1`; mỗi cặp hai lệnh: `python scripts/run_all_models.py run --config <config> --agents <a> <b> --output <root>/<a>__<b>` rồi `... verify --output <root>/<a>__<b>`. In bảng số cặp và chi phí mỗi luồng. Nghiệm thu: 12 agent ra 66 cặp không trùng; `--exclude alphago__alphazero_v2` ra 65; 7 agent lõi (`alphago alphazero_v2 rl_policy supervised alphabeta minimax mcts_pure`) bỏ cặp trọng tâm ra 20.

**T8. `analysis/make_openings.py`** (R2, R9). Sinh `thực nghiệm/chess/openings_100_b.json` cùng cấu trúc `openings_100.json` (khóa `description`, `generator_seed`, `openings`; mỗi thế 8 nước UCI). Lấy 50 tiền tố từ `OPENING_BOOK` trong `scripts/benchmark_stage6_vs_stage7.py` (đọc bằng phân tích file nguồn, không nhập module), mỗi tiền tố tạo 2 thế bằng cách nối nước hợp lệ không bắt quân và không chiếu, chọn ngẫu nhiên `random.Random(20261201)`, đủ 8 ply, Trắng đi tiếp. Loại thế trùng (so 4 trường đầu của FEN) với nhau hoặc với `openings_100.json`, sinh bù cho đủ 100. Nghiệm thu: file có 100 thế; script tự kiểm mọi thế dài 8, hợp lệ, chưa kết thúc, Trắng đi, không trùng; `plan` đạt với bản sao `experiment_cfg_final_A.json` đổi `openings_file` sang file mới.

**T9. Thư viện và tài liệu cũ** (R5). Thêm `psutil>=5.9` vào `requirements.txt`. Thêm đúng dòng sau lên đầu `docs/STAGE6_REPORT.md`, `docs/STAGE7_REPORT.md`, `results/leaderboard.md` (không xóa nội dung): `> Kết quả hoặc đề xuất cũ (trước 24/11/2026): không dùng cho báo cáo cuối.` Với `docs/HANDOFF.md` mục 7, thêm dưới tiêu đề mục: `> Đề xuất này (300 ply) đã được thay bằng arena 400 ply; không chạy theo lịch này.` Không sửa `leaderboard.csv` ở gốc. Nghiệm thu: `python -c "import psutil"` chạy; `git diff --stat` chỉ hiện các file trên và chỉ thêm dòng.

## Hoàn thành phần AI khi
T1 đến T9 đạt nghiệm thu, `git diff --stat` không có file ngoài danh sách.

## Việc của người sau khi AI xong (AI chỉ hỗ trợ khi được yêu cầu)

**Chuẩn bị.** Đổi `checkpoints/stage6` thành `stage6_old`; đặt Policy và Value Stage 5 bản 300k vào `checkpoints/stage6/policy.pt` và `value.pt` (SHA256 của `policy.pt` phải trùng `stage5/paper_frozen_2073/last.pt`); đặt AlphaZero v2 vào `checkpoints/stage7_v2/best.pt`; chạy T1; `pip install psutil`; commit và ghi `git rev-parse HEAD`. Chọn `--device` (đo `smoke` và `--limit 10` trên cả `cpu` và `cuda`, chọn bên nhanh hơn) rồi sinh lại config: `python analysis/make_configs.py --device <cpu|cuda>`.
```
python scripts/run_all_models.py check --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents alphago alphazero_v2
python scripts/run_all_models.py smoke --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents alphago alphazero_v2 --output results/smoke_final
python scripts/run_all_models.py run --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents alphago alphazero_v2 --output results/timing_ag_az2 --limit 10
```
`check` phải ra `Preflight OK`. Số đo cũ duy nhất: AlphaZero v2 đấu v1, 30 mô phỏng, GPU, 300 ply ≈ 59 giây/ván, nên 200 ván cặp trọng tâm ước 3 đến 4 giờ; dùng `timing_ag_az2` để chốt số thật.

**Thứ tự chạy** (làm từ trên xuống, dừng khi hết giờ; xong mỗi thư mục chạy `verify`):

| Ưu tiên | Việc | Ván |
|---|---|---|
| P0 | AlphaGo–AlphaZero v2, chế độ A, seed 42 | 200 |
| P1 | Đo tài nguyên trên máy nhàn rỗi (T6), làm trước khi chạy song song | vài phút mỗi agent |
| P2 | AB1 và AB2 (ablation bắt buộc) | 100 mỗi điều kiện |
| P3 | Arena chế độ A, 7 agent lõi (`alphago`, `alphazero_v2`, `rl_policy`, `supervised`, `alphabeta`, `minimax`, `mcts_pure`), 20 cặp còn lại | 200 mỗi cặp |
| P4 | AlphaGo–AlphaZero v2, chế độ B | 200 (hạ 100 nếu thiếu giờ) |
| P5 | Mở rộng arena: `alphazero` (v1), `random`, `think1` đến `think3` | 200 mỗi cặp |
| P6 | Seed 123 cho cặp trọng tâm; bộ thế `openings_100_b.json`; AB3 đến AB5 | 100 mỗi điều kiện |

Hoãn: AlphaZero v2 đấu v1 ở 400 ply (nằm trong P5 nếu chạy `alphazero`).

**Lệnh.**
```
# P0
python scripts/run_all_models.py run --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents alphago alphazero_v2 --output results/final_A_ag_vs_az2_s42
python scripts/run_all_models.py verify --output results/final_A_ag_vs_az2_s42
python analysis/arena_stats.py --input results/final_A_ag_vs_az2_s42 --output results/stats_final_A_ag_vs_az2
# P1
python analysis/measure_resources.py --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents alphago alphazero_v2 rl_policy supervised alphabeta minimax mcts_pure --device <cpu|cuda> --output results/resources_<device>.csv
# P3 (3 luồng song song; cặp trọng tâm đã chạy ở P0)
python analysis/plan_pairs.py --config "thực nghiệm/chess/experiment_cfg_final_A.json" --agents alphago alphazero_v2 rl_policy supervised alphabeta minimax mcts_pure --root results/final_A_pairs_s42 --split 3 --exclude alphago__alphazero_v2
# P4
python scripts/run_all_models.py run --config "thực nghiệm/chess/experiment_cfg_final_B.json" --agents alphago alphazero_v2 --output results/final_B_ag_vs_az2_s42
python scripts/run_all_models.py verify --output results/final_B_ag_vs_az2_s42
# P6, seed 123
python analysis/make_configs.py --seed 123 --games 100 --device <cpu|cuda> --out-dir results/_cfg_s123
python scripts/run_all_models.py run --config "results/_cfg_s123/experiment_cfg_final_A.json" --agents alphago alphazero_v2 --output results/final_A_ag_vs_az2_s123
```

**Ablation** (config `experiment_cfg_ablation.json`, 100 ván mỗi cặp, mỗi dòng một lệnh `run --config "thực nghiệm/chess/experiment_cfg_ablation.json" --agents <...> --output results/<thư mục>` kèm `verify`):

| Mã | `--agents` | Thư mục |
|---|---|---|
| AB1 | `rl_policy alphago` | `ablation_policy_vs_ag` |
| AB2 | `alphago_l0 alphazero_v2`; `alphago alphazero_v2`; `alphago_l1 alphazero_v2` | `ablation_lambda_0`, `ablation_lambda_05`, `ablation_lambda_1` |
| AB3 | `az2_s10 alphago`; `alphazero_v2 alphago`; `az2_s100 alphago`; `az2_s200 alphago` | `ablation_sims_10`, `_30`, `_100`, `_200` |
| AB4 | `minimax alphabeta` | `ablation_prune` |
| AB5 | `alphabeta_d1 mcts_pure`; `alphabeta mcts_pure`; `alphabeta_d3 mcts_pure` | `ablation_depth_1`, `_2`, `_3` |

So sánh: `python analysis/compare_conditions.py --cond L0=results/ablation_lambda_0 --cond L05=results/ablation_lambda_05 --cond L1=results/ablation_lambda_1 --output results/stats_ab2`. AB1 và AB2 bắt buộc; AB3 đến AB5 làm nếu còn giờ. Ghi trung thực: AB1 đo tác dụng của "thêm Value và tìm kiếm", chưa tách riêng MCTS; AB4 chỉ khác số nút và thời gian vì về lý thuyết cùng giá trị.

**Dữ liệu người phải cung cấp** (R2, R3, R12; AI không tự có): với từng checkpoint (Stage 4, Stage 5, Value, AlphaZero v1, v2): tên file, ngày huấn luyện, CPU/GPU, tổng thời gian huấn luyện, số ván hoặc vòng, seed, tiêu chí chọn checkpoint tốt nhất (phải không dựa vào ván đánh giá cuối); mã commit; ảnh yêu cầu mục 3.6 và đầu mục huấn luyện của thầy.

## Phụ lục: nội dung `analysis/make_configs.py`

```python
"""Tạo experiment_cfg_final_A.json, experiment_cfg_final_B.json, experiment_cfg_ablation.json.

Chạy từ gốc repo:  python analysis/make_configs.py
Nguồn: thực nghiệm/chess/experiment_cfg_az_v2.json (không bị sửa).
Chỉ đổi giá trị, giữ đúng bộ khóa mà scripts/run_all_models.py yêu cầu.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "thực nghiệm" / "chess" / "experiment_cfg_az_v2.json"

PATHS = {
    "supervised": "checkpoints/stage4/best_lichess.pt",
    "rl_policy": "checkpoints/stage5/paper_frozen_2073/last.pt",
    "alphago_policy": "checkpoints/stage6/policy.pt",
    "alphago_value": "checkpoints/stage6/value.pt",
    "alphazero": "checkpoints/stage7/best.pt",
    "alphazero_v2": "checkpoints/stage7_v2/best.pt",
}


def by_id(cfg):
    return {a["id"]: a for a in cfg["agents"]}


def apply_paths(agents):
    agents["supervised"]["kwargs"]["checkpoint"] = PATHS["supervised"]
    agents["rl_policy"]["kwargs"]["checkpoint"] = PATHS["rl_policy"]
    agents["alphago"]["kwargs"]["checkpoint_policy"] = PATHS["alphago_policy"]
    agents["alphago"]["kwargs"]["checkpoint_value"] = PATHS["alphago_value"]
    agents["alphazero"]["kwargs"]["checkpoint"] = PATHS["alphazero"]
    agents["alphazero_v2"]["kwargs"]["checkpoint"] = PATHS["alphazero_v2"]


def set_budget(agents, sims, quiescence):
    for key in ("mcts_pure", "alphago", "alphazero", "alphazero_v2"):
        agents[key]["kwargs"]["simulations"] = sims
    kw = {"depth": 2}
    if quiescence:
        kw.update(q_depth=6, use_quiescence=True)
    else:
        kw.update(use_quiescence=False)
    agents["alphabeta"]["kwargs"] = kw


def common(cfg, args, output_dir):
    cfg["max_plies"] = 400
    cfg["claim_draw"] = True
    cfg["seed"] = args.seed
    cfg["games_per_pair"] = args.games
    cfg["device"] = args.device
    cfg["torch_threads"] = args.threads
    cfg["output_dir"] = output_dir


def make_mode(base, args, mode):
    cfg = copy.deepcopy(base)
    agents = by_id(cfg)
    apply_paths(agents)
    if mode == "A":
        set_budget(agents, 30, quiescence=False)
    else:
        set_budget(agents, 100, quiescence=True)
    common(cfg, args, f"results/final_{mode}_arena_s{args.seed}")
    return cfg


def agent(base_agents, src_id, new_id, **override):
    item = copy.deepcopy(base_agents[src_id])
    item["id"] = new_id
    item["kwargs"].update(override)
    return item


def make_ablation(base, args):
    cfg = make_mode(base, args, "A")
    cfg["games_per_pair"] = args.ablation_games
    a = by_id(cfg)
    agents = [
        a["minimax"], a["alphabeta"],
        agent(a, "alphabeta", "alphabeta_d1", depth=1),
        agent(a, "alphabeta", "alphabeta_d3", depth=3),
        a["mcts_pure"], a["rl_policy"],
        agent(a, "alphago", "alphago_l0", rollout_weight=0.0),
        a["alphago"],
        agent(a, "alphago", "alphago_l1", rollout_weight=1.0),
        agent(a, "alphazero_v2", "az2_s10", simulations=10),
        a["alphazero_v2"],
        agent(a, "alphazero_v2", "az2_s100", simulations=100),
        agent(a, "alphazero_v2", "az2_s200", simulations=200),
    ]
    cfg["agents"] = agents
    cfg["output_dir"] = f"results/ablation_s{args.seed}"
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=200, help="ván mỗi cặp ở chế độ A và B, số chẵn")
    parser.add_argument("--ablation-games", type=int, default=100, help="ván mỗi cặp ở file ablation, số chẵn")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out-dir", default=str(SRC.parent))
    args = parser.parse_args()
    base = json.loads(SRC.read_text(encoding="utf-8-sig"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "experiment_cfg_final_A.json": make_mode(base, args, "A"),
        "experiment_cfg_final_B.json": make_mode(base, args, "B"),
        "experiment_cfg_ablation.json": make_ablation(base, args),
    }
    for name, cfg in files.items():
        (out / name).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Đã tạo", out / name, f"({len(cfg['agents'])} agent)")


if __name__ == "__main__":
    main()
```
