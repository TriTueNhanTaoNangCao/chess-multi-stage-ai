# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 200
- **Tổng số cặp:** 1
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\final_B_ag_vs_az2_s42` | fingerprint: `c45c66c2d5237737fcca7757e0c9f1242db4daa7ef10c1fb20d549c11621f10a` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `alphago__alphazero_v2` | 200 | 175-23-2 | 0.9325 | [0.8890, 0.9597] | [0.9075, 0.9550] | 0.9600 / 0.9050 | 67.92 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `alphago` | 200 | 0.9325 | 1988.65 | 1974.865 | 2812.092 | - | - | 100 | 0 / 0 |
| `alphazero_v2` | 200 | 0.0675 | 1595.222 | 1556.62 | 2411.694 | - | - | 100 | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 177 | 88.50% |
| `threefold_repetition` | 19 | 9.50% |
| `fifty_moves` | 2 | 1.00% |
| `insufficient_material` | 1 | 0.50% |
| `stalemate` | 1 | 0.50% |
