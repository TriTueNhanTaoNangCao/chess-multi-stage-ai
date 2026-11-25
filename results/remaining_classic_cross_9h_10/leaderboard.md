# Classic internal + classic vs neural: reduced benchmark

Completed 490/490 games; covered 49/49 pairs.
Classic internal: 210; cross-group: 280; reused: 130.
Neural-vs-neural games excluded. Elo belongs only to this selected schedule.
Percentages use 0..100; small sample, paired games are not independent.

| Agent | W / D / L | Win % | Score % | White win % | Black win % | ms/move |
|---|---:|---:|---:|---:|---:|---:|
| alphabeta | 99/0/1 | 99.0 | 99.0 | 98.0 | 100.0 | 363.431 |
| minimax | 89/1/10 | 89.0 | 89.5 | 88.0 | 90.0 | 397.707 |
| alphago | 50/2/18 | 71.4286 | 72.8571 | 71.4286 | 71.4286 | 3530.941 |
| think2 | 36/32/32 | 36.0 | 52.0 | 36.0 | 36.0 | 10.29 |
| think3 | 36/25/39 | 36.0 | 48.5 | 38.0 | 34.0 | 8.55 |
| supervised | 12/36/22 | 17.1429 | 42.8571 | 14.2857 | 20.0 | 5.137 |
| rl_policy | 12/36/22 | 17.1429 | 42.8571 | 14.2857 | 20.0 | 4.585 |
| think1 | 17/20/63 | 17.0 | 27.0 | 18.0 | 16.0 | 0.676 |
| mcts_pure | 31/22/47 | 31.0 | 42.0 | 30.0 | 32.0 | 1247.614 |
| random | 2/27/71 | 2.0 | 15.5 | 2.0 | 2.0 | 0.336 |
| alphazero | 0/11/59 | 0.0 | 7.8571 | 0.0 | 0.0 | 2746.902 |
