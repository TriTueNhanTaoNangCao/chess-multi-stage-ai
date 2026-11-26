# Ma trận Score đối đầu (Wilson 95% CI)

Ô ở hàng `i`, cột `j` là `score = (W + 0.5*D) / N` của `i` khi đấu với `j`, kèm `[Wilson low, Wilson high]`.

| Agent | alphabeta | mcts_pure | alphago | alphazero_v2 |
| :--- | :---: | :---: | :---: | :---: |
| `alphabeta` | - | 1.0000 [0.963, 1.000] | 0.8650 [0.784, 0.919] | 1.0000 [0.963, 1.000] |
| `mcts_pure` | 0.0000 [0.000, 0.037] | - | - | - |
| `alphago` | 0.1350 [0.081, 0.216] | - | - | - |
| `alphazero_v2` | 0.0000 [0.000, 0.037] | - | - | - |
