# All-model chess benchmark

Mode: reduced_time_budget; completed 130/550 games.
Error games: 0; timeout games: 0.
Percentages use 0..100. Elo is local, schedule-dependent. Paired games are not independent.

| Agent | W / D / L | Win % | Score % | White win % | Black win % | Elo | ms/move |
|---|---:|---:|---:|---:|---:|---:|---:|
| minimax | 30/0/0 | 100.0 | 100.0 | 100.0 | 100.0 | 1455.24 | 418.324 |
| think3 | 20/6/4 | 66.6667 | 76.6667 | 73.3333 | 60.0 | 1316.03 | 8.289 |
| alphabeta | 20/0/0 | 100.0 | 100.0 | 100.0 | 100.0 | 1302.33 | 300.771 |
| mcts_pure | 20/0/0 | 100.0 | 100.0 | 100.0 | 100.0 | 1284.83 | 1472.751 |
| think2 | 21/8/11 | 52.5 | 62.5 | 55.0 | 50.0 | 1260.95 | 11.644 |
| alphago | 0/0/0 | None | None | None | None | 1200.0 | None |
| alphazero | 0/0/0 | None | None | None | None | 1200.0 | None |
| rl_policy | 0/0/0 | None | None | None | None | 1200.0 | None |
| supervised | 0/0/0 | None | None | None | None | 1200.0 | None |
| think1 | 6/5/49 | 10.0 | 14.1667 | 10.0 | 10.0 | 924.65 | 0.661 |
| random | 1/5/54 | 1.6667 | 5.8333 | 3.3333 | 0.0 | 855.99 | 0.329 |
