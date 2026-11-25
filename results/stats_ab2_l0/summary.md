# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 100
- **Tổng số cặp:** 1
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\ablation_lambda_0` | fingerprint: `e26c07d378aaa11b41f9bba865b770f65e92f96ad086fcfa7be59b1961b1e2ce` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `alphago_l0__alphazero_v2` | 100 | 66-34-0 | 0.8300 | [0.7445, 0.8911] | [0.7800, 0.8750] | 0.8600 / 0.8000 | 65.3 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `alphago_l0` | 100 | 0.8300 | 470.334 | 476.538 | 663.089 | - | - | 30 | 0 / 0 |
| `alphazero_v2` | 100 | 0.1700 | 515.007 | 509.145 | 797.197 | - | - | 30 | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 66 | 66.00% |
| `threefold_repetition` | 31 | 31.00% |
| `stalemate` | 3 | 3.00% |
