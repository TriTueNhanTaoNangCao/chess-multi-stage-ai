# Ma trận Tổng hợp các Chế độ Ngân sách, Độ nhạy & Ablation

| Nhóm | Điều kiện | Cặp (A__B) | N | W–D–L (A) | Score (A) | Wilson 95% | Bootstrap 95% | ms TB (A / B) |
| :--- | :--- | :--- | ---: | :---: | ---: | :---: | :---: | :---: |
| Ngân sách (A vs B) | `ModeA_30sims` | `alphago__alphazero_v2` | 200 | 164-35-1 | **0.9075** | [0.8593, 0.9403] | [0.8800, 0.9325] | 812.2 / 674.8 |
| Ngân sách (A vs B) | `ModeB_100sims` | `alphago__alphazero_v2` | 200 | 175-23-2 | **0.9325** | [0.8890, 0.9597] | [0.9075, 0.9550] | 1988.7 / 1595.2 |
| Độ nhạy Seed (50 thế đầu) | `Seed42` | `alphago__alphazero_v2` | 100 | 80-19-1 | **0.8950** | [0.8196, 0.9411] | [0.8550, 0.9350] | 990.7 / 828.7 |
| Độ nhạy Seed (50 thế đầu) | `Seed123` | `alphago__alphazero_v2` | 100 | 87-12-1 | **0.9300** | [0.8625, 0.9657] | [0.8950, 0.9600] | 1035.4 / 898.7 |
| Độ nhạy Khai cuộc | `Openings_A` | `alphago__alphazero_v2` | 200 | 164-35-1 | **0.9075** | [0.8593, 0.9403] | [0.8800, 0.9325] | 812.2 / 674.8 |
| Độ nhạy Khai cuộc | `Openings_B` | `alphago__alphazero_v2` | 100 | 91-8-1 | **0.9500** | [0.8882, 0.9785] | [0.9200, 0.9800] | 953.5 / 785.1 |
| AB1 (Policy+Value+MCTS) | `AB1: AlphaGo vs RL Policy` | `alphago__rl_policy` | 100 | 74-24-2 | **0.8600** | [0.7786, 0.9147] | [0.8100, 0.9050] | 633.6 / 6.4 |
| AB2 (λ Value vs Rollout) | `L0_value_only` | `alphago_l0__alphazero_v2` | 100 | 66-34-0 | **0.8300** | [0.7445, 0.8911] | [0.7800, 0.8750] | 470.3 / 515.0 |
| AB2 (λ Value vs Rollout) | `L05_hybrid` | `alphago__alphazero_v2` | 100 | 80-19-1 | **0.8950** | [0.8196, 0.9411] | [0.8550, 0.9350] | 990.7 / 828.7 |
| AB2 (λ Value vs Rollout) | `L1_rollout_only` | `alphago_l1__alphazero_v2` | 100 | 78-21-1 | **0.8850** | [0.8078, 0.9337] | [0.8400, 0.9250] | 1099.2 / 950.2 |
| AB4 (Cắt tỉa Alpha-Beta) | `AB4: Minimax vs AlphaBeta` | `minimax__alphabeta` | 100 | 38-27-35 | **0.5150** | [0.4183, 0.6106] | [0.5000, 0.5400] | 385.0 / 55.6 |
| Ngân sách Timed 2s (Tesla T4) | `Timed 2s: alphabeta__alphago` | `alphabeta__alphago` | 100 | 83-7-10 | **0.8650** | [0.7844, 0.9186] | [0.8050, 0.9200] | 1982.0 / 1928.8 |
| Ngân sách Timed 2s (Tesla T4) | `Timed 2s: alphabeta__alphazero_v2` | `alphabeta__alphazero_v2` | 100 | 100-0-0 | **1.0000** | [0.9630, 1.0000] | [1.0000, 1.0000] | 1981.7 / 1888.8 |
| Ngân sách Timed 2s (Tesla T4) | `Timed 2s: alphabeta__mcts_pure` | `alphabeta__mcts_pure` | 100 | 100-0-0 | **1.0000** | [0.9630, 1.0000] | [1.0000, 1.0000] | 1982.2 / 1833.3 |
