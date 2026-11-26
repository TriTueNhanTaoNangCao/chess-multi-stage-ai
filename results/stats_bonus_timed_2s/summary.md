# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 300
- **Tổng số cặp:** 3
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\bonus_s42\bonus_timed_2s_ab_vs_mcts` | fingerprint: `75b578818018aad32594c2ad94421baa7eaec38d52670979c4c3ea1789fcc2ae` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\bonus_s42\bonus_timed_2s_ab_vs_ag` | fingerprint: `4f3c9419e2cf02651e4f9eaacb26aa9c308e23d5381f9925d9e37ef81ca7e069` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\bonus_s42\bonus_timed_2s_ab_vs_az2` | fingerprint: `565618c089d83f8a0c605d18dc477acdafcd1b7ebc429adef03a6b456c048a29` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.13.16`, chess `1.11.2`, torch `2.11.0+cu130`, numpy `2.1.3`
- CPU: `x86_64` (2 physical / 4 logical), RAM: `31.3 GB`
- GPU: `Tesla T4`, CUDA: `13.0`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `alphabeta__alphago` | 100 | 83-7-10 | 0.8650 | [0.7844, 0.9186] | [0.8050, 0.9200] | 0.9200 / 0.8100 | 64.61 |
| `alphabeta__alphazero_v2` | 100 | 100-0-0 | 1.0000 | [0.9630, 1.0000] | [1.0000, 1.0000] | 1.0000 / 1.0000 | 44.08 |
| `alphabeta__mcts_pure` | 100 | 100-0-0 | 1.0000 | [0.9630, 1.0000] | [1.0000, 1.0000] | 1.0000 / 1.0000 | 31.72 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `alphabeta` | 300 | 0.9550 | 1981.967 | 1981.841 | 1983.034 | 6130.446 | 955.337 | - | 0 / 0 |
| `alphago` | 100 | 0.1350 | 1928.784 | 1986.62 | 2185.489 | - | - | 5000 | 0 / 0 |
| `alphazero_v2` | 100 | 0.0000 | 1888.789 | 1986.845 | 2181.436 | - | - | 5000 | 0 / 0 |
| `mcts_pure` | 100 | 0.0000 | 1833.337 | 1981.434 | 1982.634 | - | - | 5000 | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 293 | 97.67% |
| `threefold_repetition` | 7 | 2.33% |
