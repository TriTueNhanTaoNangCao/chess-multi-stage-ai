# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 100
- **Tổng số cặp:** 1
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\ablation_prune` | fingerprint: `5f37090a882692c4d57173170fa6ce428d9a4acc925cc38d9f0dde4cd3edb672` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `minimax__alphabeta` | 100 | 38-27-35 | 0.5150 | [0.4183, 0.6106] | [0.5000, 0.5400] | 0.6400 / 0.3900 | 65.5 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `minimax` | 100 | 0.5150 | 384.963 | 360.564 | 878.522 | 1058.629 | - | - | 0 / 0 |
| `alphabeta` | 100 | 0.4850 | 55.626 | 44.552 | 135.919 | 160.402 | 27.57 | - | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 73 | 73.00% |
| `threefold_repetition` | 27 | 27.00% |
