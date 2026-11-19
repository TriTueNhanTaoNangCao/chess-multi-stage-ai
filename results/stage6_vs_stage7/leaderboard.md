# Local Elo Leaderboard: Stage 6 (AlphaGo) vs Stage 7 (AlphaZero)

- Tổng số ván hoàn thành: 200/200
- Giới hạn nước đi mỗi ván: 300 plies (Vượt giới hạn xử hòa: max_plies_adjudication)
- Số mô phỏng MCTS: 30 sims/nước
- Thiết bị tính toán: cpu | Luồng CPU: 2
- Nhiễu Dirichlet AlphaZero: Tắt (Đối xứng điều kiện)
- Elo khởi điểm: 1200.0 | K-factor: 32

### Bảng Xếp Hạng Hiệu Năng

| Agent | Elo | W / D / L | Score | Score % (Wilson 95% CI) | Win % | Draw % | ms/move |
| :--- | ---: | :---: | ---: | :---: | ---: | ---: | ---: |
| AlphaGo (30 sims) | 1354.5 | 153 / 47 / 0 | 176.5 | 88.2% [0.831, 0.920] | 76.5% | 23.5% | 903.33 |
| AlphaZero (30 sims) | 1045.5 | 0 / 47 / 153 | 23.5 | 11.8% [0.080, 0.169] | 0.0% | 23.5% | 587.94 |

## Ma trận đối đầu chi tiết (Trắng / Đen)

| Tình huống | Số ván | Thắng | Hòa | Thua | Tỷ lệ thắng |
| :--- | ---: | ---: | ---: | ---: | ---: |
| AlphaGo (30 sims) cầm Trắng | 100 | 69 | 31 | 0 | 69.0% |
| AlphaZero (30 sims) cầm Trắng | 100 | 0 | 16 | 84 | 0.0% |

## Danh mục Checkpoint sử dụng (Audit SHA256)

| Thành phần | Đường dẫn | SHA256 |
| :--- | :--- | :--- |
| Stage 6 Policy | `checkpoints/...` | `4d70697035064bfc4be2e505959bb070d4a3afd0c697390b78908605016e1832` |
| Stage 6 Value | `checkpoints/...` | `93dd82049962b4e8f9c94927879a766ba10fc914f9474dda6a794b3f88ca07ca` |
| Stage 7 AlphaZero | `checkpoints/...` | `6c8e5302898c2eb6889890aacf09845215a1d1f79814b91a14267302f74b4ca8` |
