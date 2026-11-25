# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 100
- **Tổng số cặp:** 1
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\ablation_policy_vs_ag` | fingerprint: `b7c653edd4cc810197f49148f82d909a5e52e5d43812e0d3bd1de8897e865ebe` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `rl_policy__alphago` | 100 | 2-24-74 | 0.1400 | [0.0853, 0.2214] | [0.0950, 0.1900] | 0.1400 / 0.1400 | 90.8 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `alphago` | 100 | 0.8600 | 633.628 | 611.647 | 1051.0 | - | - | 30 | 0 / 0 |
| `rl_policy` | 100 | 0.1400 | 6.355 | 3.05 | 5.446 | - | - | - | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 76 | 76.00% |
| `threefold_repetition` | 17 | 17.00% |
| `stalemate` | 4 | 4.00% |
| `insufficient_material` | 3 | 3.00% |
