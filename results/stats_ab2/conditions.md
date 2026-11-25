# So sánh các điều kiện thực nghiệm

- Bootstrap: `10000` lần (seed `20261124`)
- Giới hạn: `50` thế mở đầu đầu tiên

| Nhãn | Cặp | Seed | N | W-D-L | Score | Wilson 95% | Bootstrap 95% | ms TB (A) | ms TB (B) |
| :--- | :--- | ---: | ---: | :---: | ---: | :---: | :---: | ---: | ---: |
| `L0_value_only` | `alphago_l0__alphazero_v2` | 42 | 100 | 66-34-0 | 0.8300 | [0.7445, 0.8911] | [0.7800, 0.8750] | 470.334 | 515.007 |
| `L05_hybrid` | `alphago__alphazero_v2` | 42 | 100 | 80-19-1 | 0.8950 | [0.8196, 0.9411] | [0.8550, 0.9350] | 990.731 | 828.676 |
| `L1_rollout_only` | `alphago_l1__alphazero_v2` | 42 | 100 | 78-21-1 | 0.8850 | [0.8078, 0.9337] | [0.8400, 0.9250] | 1099.206 | 950.164 |
