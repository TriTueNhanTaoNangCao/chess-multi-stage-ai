# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 100
- **Tổng số cặp:** 1
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\final_A_ag_vs_az2_s123` | fingerprint: `1d24bde1be5b99deb24e5b9c1161db10c6c54eb1b5eef25f2b0e894a6deee558` | seed=123, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `alphago__alphazero_v2` | 100 | 87-12-1 | 0.9300 | [0.8625, 0.9657] | [0.8950, 0.9600] | 0.9500 / 0.9100 | 71.89 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `alphago` | 100 | 0.9300 | 1035.418 | 980.88 | 1716.697 | - | - | 30 | 0 / 0 |
| `alphazero_v2` | 100 | 0.0700 | 898.702 | 853.448 | 1570.36 | - | - | 30 | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 88 | 88.00% |
| `threefold_repetition` | 11 | 11.00% |
| `insufficient_material` | 1 | 1.00% |
