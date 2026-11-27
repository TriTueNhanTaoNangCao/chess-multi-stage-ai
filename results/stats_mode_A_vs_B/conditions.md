# So sánh các điều kiện thực nghiệm

- Bootstrap: `10000` lần (seed `20261124`)

| Nhãn | Cặp | Seed | N | W-D-L | Score | Wilson 95% | Bootstrap 95% | ms TB (A) | ms TB (B) |
| :--- | :--- | ---: | ---: | :---: | ---: | :---: | :---: | ---: | ---: |
| `ModeA_30sims` | `alphago__alphazero_v2` | 42 | 200 | 164-35-1 | 0.9075 | [0.8593, 0.9403] | [0.8800, 0.9325] | 812.218 | 674.832 |
| `ModeB_100sims` | `alphago__alphazero_v2` | 42 | 200 | 175-23-2 | 0.9325 | [0.8890, 0.9597] | [0.9075, 0.9550] | 1988.65 | 1595.222 |
