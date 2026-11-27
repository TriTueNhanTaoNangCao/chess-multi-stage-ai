# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 100
- **Tổng số cặp:** 1
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\ablation_lambda_1` | fingerprint: `55cf95127a21f13d3038c64bcaf9bf87b6fca1324bd3fd1e9cde43d4edde67b8` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `alphago_l1__alphazero_v2` | 100 | 78-21-1 | 0.8850 | [0.8078, 0.9337] | [0.8400, 0.9250] | 0.9200 / 0.8500 | 83.49 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `alphago_l1` | 100 | 0.8850 | 1099.206 | 1082.588 | 1839.22 | - | - | 30 | 0 / 0 |
| `alphazero_v2` | 100 | 0.1150 | 950.164 | 934.19 | 1697.661 | - | - | 30 | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 79 | 79.00% |
| `threefold_repetition` | 20 | 20.00% |
| `stalemate` | 1 | 1.00% |
