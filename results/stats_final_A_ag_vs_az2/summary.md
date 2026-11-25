# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 200
- **Tổng số cặp:** 1
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\final_A_ag_vs_az2_s42` | fingerprint: `f311ec23936bf579177e745762188806e5ff31ea708dd7d0e6aeb0256f4448fa` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `alphago__alphazero_v2` | 200 | 164-35-1 | 0.9075 | [0.8593, 0.9403] | [0.8800, 0.9325] | 0.9200 / 0.8950 | 77.6 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `alphago` | 200 | 0.9075 | 812.218 | 693.118 | 1652.692 | - | - | 30 | 0 / 0 |
| `alphazero_v2` | 200 | 0.0925 | 674.832 | 579.092 | 1420.859 | - | - | 30 | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 165 | 82.50% |
| `threefold_repetition` | 30 | 15.00% |
| `stalemate` | 3 | 1.50% |
| `fifty_moves` | 2 | 1.00% |
