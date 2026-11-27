# Tổng hợp Thống kê Arena

> **Lưu ý:** hai ván cùng thế mở đầu không độc lập; không tính Elo.

- **Tổng số ván:** 11000
- **Tổng số cặp:** 55
- **Bootstrap:** 10000 lần lấy mẫu có hoàn lại theo `opening_index` (seed `20261124`)
- **Thư mục nguồn & Fingerprint:**
  - `results\final_A_ag_vs_az2_s42` | fingerprint: `f311ec23936bf579177e745762188806e5ff31ea708dd7d0e6aeb0256f4448fa` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphabeta__alphazero_v2` | fingerprint: `6494390aded25fea46c064a7c0291a2dde700a927d6643bb6fa69e5aef0c05c0` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphabeta__rl_policy` | fingerprint: `e3d4395f695121e9ac28200ccbca209d3a8014a835ca1355644c92a78a161427` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphabeta__supervised` | fingerprint: `8351d358d2c2b14013ab7ae6471d5588b3e9fba18992c023df2468a01697dee0` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphago__alphabeta` | fingerprint: `8e7075ec49f075831ac341b81992269e55de4aecd75ea0e6e4dd07bf5944c8a1` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphago__mcts_pure` | fingerprint: `a09a415eb51a6488ba658eb3280166efb518004c784b774f9906e4b39c232b72` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphago__minimax` | fingerprint: `b00686bb547d2e28da0a37befbd39158c1b5f1f31be0f572ab6cfc69f7619ce1` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphago__rl_policy` | fingerprint: `0c7b882166586aa0cdee77edf6a9ae440728806fcfa8c5d1cd2508fe54a4e27b` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphago__supervised` | fingerprint: `740608d9fc04f074057549ea33a5fe9d9fb10010d4ec0c325eeff718ce98cc9f` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\alphazero_v2__mcts_pure` | fingerprint: `7cf3dfcfa8d97179a04e3597e293348a7981d8f1f91e4c6e3068b067dd863e83` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\mcts_pure__alphabeta` | fingerprint: `14610d293877957cc9a698308fce4fa46deee989cb208b6a6ca69be3ac4f2ab7` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\mcts_pure__minimax` | fingerprint: `59b7def2f0d42d12c6502b2c61504871d871d6845a9145049a72d04316829f10` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\mcts_pure__rl_policy` | fingerprint: `5760ea741c70688ca788a423b4c960ca3e3fe19490fa73baacbf5a3ef9742e10` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\mcts_pure__supervised` | fingerprint: `ff708ebbae42620f29120d40554a240618fed5b0c79c01aa20e8c9fb2b10469a` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\minimax__alphabeta` | fingerprint: `0beb13cf283f396f308f6cd6b77642cc26abfb65cdd0334109c1f6f6a9d4efee` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\minimax__alphazero_v2` | fingerprint: `9665beea7064cdca283192dbbbaa78fcb82bbb42ca0dabcc293b286733446139` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\minimax__rl_policy` | fingerprint: `0ab52310c5f6dc64fe7f1590f682226f2d5a652402d5cf961c04333b50003097` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\minimax__supervised` | fingerprint: `baa37979125afd7160d2a1f8986d864903a4b309a346f5bb2600aead4e44b436` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\rl_policy__alphazero_v2` | fingerprint: `46ec86dec1b4c16265e17d15b2dfb62ec64bc8d0765cab43b0cf478491b42fcd` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\supervised__alphazero_v2` | fingerprint: `84871a20e8b2c5b7601958f42168ce52913d8942ec96d263bf6f1e31e3ea7954` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_pairs_s42\supervised__rl_policy` | fingerprint: `f2f85da3078195c548f13dcc6f85e3774223e332eb8ef7ae0167379d942d1715` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__alphabeta` | fingerprint: `bb7f7b2d76b5471fab1481a6e62d3dec9a03567479c21d46957711ed2c7f3373` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__alphago` | fingerprint: `81f94087d9ab28fe0503fc6511af29887c99002978971e27dc529a72ef3aaffe` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__alphazero_v2` | fingerprint: `28833c21e397c09e85baf3adf2520deb2bf5ab4371854c76139ad8d7d126f752` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__mcts_pure` | fingerprint: `01709d639bef0d502abfa9bc9abf29acc059940768637de3c335b69a0dab3b65` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__minimax` | fingerprint: `d93ed6b8a52b150f99090567660bbc03f5b6b5f3790c6de2d4d860572e3dd665` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__rl_policy` | fingerprint: `2f9fa928b69f702d6bdd6566d23286c211215388140a220205fccbacce495da2` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__supervised` | fingerprint: `50405c7e95d65a4b8eda088a189622686e712110d2525ed2bba1b48316a05401` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__think1` | fingerprint: `0b440858f6acc339138f5e69289480f1a8a36ff0a78897b1c4a14a177c66b6c0` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__think2` | fingerprint: `3fd90b30e7b44e3be1c189f9b79811700b1e5c6b5b786e33bd85489adf2b3003` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\random__think3` | fingerprint: `debabb33e6b78aef8c86c393b1f192ebfef21e532f09cb2f25fff64e3d23e9f7` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__alphabeta` | fingerprint: `c739aa23f5e6b05a1450a973bf1f9a5ed6ae9f7c8546e6fb7842c8898cba11f8` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__alphago` | fingerprint: `21735e89418ac08e26b88f6ce02bb22f1c4cf8e1fb3cafec29c93a3843785678` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__alphazero_v2` | fingerprint: `532eaf2cac465140b557f9daac1c469c3753cbe59b9b14314f61636d8ea83e7a` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__mcts_pure` | fingerprint: `758b89e23a608bbebe9c89ded4b7bdf67e3b212d4f1822b08f6152185bdccada` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__minimax` | fingerprint: `216e0e84f3fdbb2ed8fee15c5da0e97b76ec985c0096c10edb4bb0f9f7879b77` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__rl_policy` | fingerprint: `e78d292f4363004c5c4d331125cd398b4068355f635cf502b125c2786c686184` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__supervised` | fingerprint: `fe484b41f1ddd3cb2561b810735b7fd6dacfc153b9098bec6b6e0c7553b91445` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__think2` | fingerprint: `7e0fb899fd6a50d33ccaaa70689b383e8e0de85182d55b8c4428723938e64815` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think1__think3` | fingerprint: `e496689598040ce89370ce88ef0bd560f839727dfbf8f30c3957d53e6510e3e9` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__alphabeta` | fingerprint: `420e6084bea3a0f17bd9f3527a663ad4bece10f108f1870a731cae3c8b8c9b25` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__alphago` | fingerprint: `87100f3d087d14b2049d6fd1bbdb1758db21936d495071f698329c0cb5dd1c23` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__alphazero_v2` | fingerprint: `db470fb57018a3d8ed9f26a4554354043bd34e74aa9cb08924e8d05044514e97` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__mcts_pure` | fingerprint: `c9f12ba0b94e40f8a35681fe5154612cea253549f8954c9b8e60346a644ec32f` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__minimax` | fingerprint: `0f05ccb47bc53a573f538c77279a4e6823a88ddaba0ec02e33aee9309edf4a24` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__rl_policy` | fingerprint: `af993a140c9793dc44935b5d40418c7d2e5ee06e3b5469543329f979fd9ea7b6` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__supervised` | fingerprint: `7ceddd796b3b48aa095d8dc31d22a2530d4d022d07e67f92638d41916dd626d8` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think2__think3` | fingerprint: `92aa143e18f8b16a9c73d332203c4294d56a16ba0bc0768d55d006154640154a` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think3__alphabeta` | fingerprint: `cd9a8f7b38913ab26ff7f49884b051faf89e4d02623ec73c488d4313dc8552b0` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think3__alphago` | fingerprint: `9f52659d045ff69ffb21e5d99a6d8490d95d9108ac4dc42a134634ef6bd7370e` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think3__alphazero_v2` | fingerprint: `89bb0fe5c846543f7d310dc9cc903536f0fab885f0544f621512f5a2ba137f24` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think3__mcts_pure` | fingerprint: `c432640a581bbd4f2651fcf08a9cada8052408134050f864814d8232233f05c8` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think3__minimax` | fingerprint: `3a60b69b6c19743dae8e74e29b8e4e8bc88b9d5b3b6acf295229ef6344b034bc` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think3__rl_policy` | fingerprint: `9c9e4fce2cdfb46236314cf2ebed0706d7cfbfaba8007136f946fad9a70ec0ea` | seed=42, max_plies=400, claim_draw=True, device=cuda
  - `results\final_A_stage1_s42\think3__supervised` | fingerprint: `eb4b712bbe50d5f60123f08c12f37866e81fa70e42e2da8144ca338d8442204c` | seed=42, max_plies=400, claim_draw=True, device=cuda

## Cấu hình môi trường (manifest đầu)
- Python `3.11.9`, chess `1.11.2`, torch `2.6.0+cu124`, numpy `2.4.6`
- CPU: `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel` (8 physical / 12 logical), RAM: `23.7 GB`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, CUDA: `12.4`

## Kết quả theo cặp

| Cặp | N | W-D-L (A) | Score | Wilson 95% | Bootstrap 95% | Trắng / Đen (A) | Độ dài TB (ply) |
| :--- | ---: | :---: | ---: | :---: | :---: | :---: | ---: |
| `alphabeta__alphago` | 200 | 104-42-54 | 0.6250 | [0.5561, 0.6891] | [0.5650, 0.6850] | 0.6650 / 0.5850 | 82.56 |
| `alphabeta__alphazero_v2` | 200 | 188-12-0 | 0.9700 | [0.9361, 0.9862] | [0.9525, 0.9850] | 0.9850 / 0.9550 | 54.12 |
| `alphabeta__mcts_pure` | 200 | 199-1-0 | 0.9975 | [0.9765, 0.9997] | [0.9925, 1.0000] | 1.0000 / 0.9950 | 36.97 |
| `alphabeta__rl_policy` | 200 | 138-49-13 | 0.8125 | [0.7527, 0.8605] | [0.7700, 0.8525] | 0.8300 / 0.7950 | 73.81 |
| `alphabeta__supervised` | 200 | 129-52-19 | 0.7750 | [0.7123, 0.8274] | [0.7275, 0.8200] | 0.7950 / 0.7550 | 78.28 |
| `alphago__alphazero_v2` | 200 | 164-35-1 | 0.9075 | [0.8593, 0.9403] | [0.8800, 0.9325] | 0.9200 / 0.8950 | 77.6 |
| `mcts_pure__alphago` | 200 | 1-4-195 | 0.0150 | [0.0051, 0.0432] | [0.0025, 0.0300] | 0.0150 / 0.0150 | 58.16 |
| `mcts_pure__alphazero_v2` | 200 | 1-132-67 | 0.3350 | [0.2732, 0.4030] | [0.3050, 0.3650] | 0.2850 / 0.3850 | 110.47 |
| `mcts_pure__rl_policy` | 200 | 0-113-87 | 0.2825 | [0.2247, 0.3485] | [0.2500, 0.3175] | 0.3200 / 0.2450 | 62.98 |
| `mcts_pure__supervised` | 200 | 3-94-103 | 0.2500 | [0.1951, 0.3143] | [0.2150, 0.2850] | 0.2850 / 0.2150 | 61.23 |
| `minimax__alphabeta` | 200 | 69-65-66 | 0.5075 | [0.4387, 0.5760] | [0.4975, 0.5225] | 0.6100 / 0.4050 | 62.13 |
| `minimax__alphago` | 200 | 104-42-54 | 0.6250 | [0.5561, 0.6891] | [0.5650, 0.6850] | 0.6650 / 0.5850 | 82.83 |
| `minimax__alphazero_v2` | 200 | 188-12-0 | 0.9700 | [0.9361, 0.9862] | [0.9525, 0.9850] | 0.9850 / 0.9550 | 54.2 |
| `minimax__mcts_pure` | 200 | 199-1-0 | 0.9975 | [0.9765, 0.9997] | [0.9925, 1.0000] | 1.0000 / 0.9950 | 36.82 |
| `minimax__rl_policy` | 200 | 137-49-14 | 0.8075 | [0.7473, 0.8561] | [0.7625, 0.8475] | 0.8250 / 0.7900 | 73.5 |
| `minimax__supervised` | 200 | 129-54-17 | 0.7800 | [0.7176, 0.8318] | [0.7325, 0.8225] | 0.7950 / 0.7650 | 78.56 |
| `random__alphabeta` | 200 | 0-0-200 | 0.0000 | [0.0000, 0.0188] | [0.0000, 0.0000] | 0.0000 / 0.0000 | 33.91 |
| `random__alphago` | 200 | 0-4-196 | 0.0100 | [0.0027, 0.0357] | [0.0025, 0.0200] | 0.0150 / 0.0050 | 54.27 |
| `random__alphazero_v2` | 200 | 0-112-88 | 0.2800 | [0.2224, 0.3459] | [0.2475, 0.3125] | 0.2050 / 0.3550 | 117.09 |
| `random__mcts_pure` | 200 | 1-10-189 | 0.0300 | [0.0138, 0.0639] | [0.0150, 0.0475] | 0.0300 / 0.0300 | 119.11 |
| `random__minimax` | 200 | 0-0-200 | 0.0000 | [0.0000, 0.0188] | [0.0000, 0.0000] | 0.0000 / 0.0000 | 34.2 |
| `random__rl_policy` | 200 | 0-72-128 | 0.1800 | [0.1329, 0.2391] | [0.1500, 0.2100] | 0.2150 / 0.1450 | 58.7 |
| `random__supervised` | 200 | 0-98-102 | 0.2450 | [0.1906, 0.3090] | [0.2100, 0.2800] | 0.2750 / 0.2150 | 62.91 |
| `random__think1` | 200 | 11-85-104 | 0.2675 | [0.2110, 0.3328] | [0.2250, 0.3100] | 0.2450 / 0.2900 | 201.54 |
| `random__think2` | 200 | 0-21-179 | 0.0525 | [0.0292, 0.0927] | [0.0300, 0.0775] | 0.0650 / 0.0400 | 83.88 |
| `random__think3` | 200 | 0-13-187 | 0.0325 | [0.0154, 0.0672] | [0.0150, 0.0525] | 0.0350 / 0.0300 | 47.37 |
| `rl_policy__alphago` | 200 | 1-49-150 | 0.1275 | [0.0882, 0.1808] | [0.0975, 0.1575] | 0.1250 / 0.1300 | 83.91 |
| `rl_policy__alphazero_v2` | 200 | 69-125-6 | 0.6575 | [0.5893, 0.7197] | [0.6225, 0.6950] | 0.7100 / 0.6050 | 76.8 |
| `supervised__alphago` | 200 | 4-50-146 | 0.1450 | [0.1029, 0.2005] | [0.1125, 0.1775] | 0.1550 / 0.1350 | 85.02 |
| `supervised__alphazero_v2` | 200 | 47-149-4 | 0.6075 | [0.5384, 0.6725] | [0.5750, 0.6376] | 0.5950 / 0.6200 | 84.85 |
| `supervised__rl_policy` | 200 | 33-137-30 | 0.5075 | [0.4387, 0.5760] | [0.4675, 0.5475] | 0.5150 / 0.5000 | 86.97 |
| `think1__alphabeta` | 200 | 0-0-200 | 0.0000 | [0.0000, 0.0188] | [0.0000, 0.0000] | 0.0000 / 0.0000 | 33.53 |
| `think1__alphago` | 200 | 1-3-196 | 0.0125 | [0.0039, 0.0395] | [0.0025, 0.0275] | 0.0200 / 0.0050 | 56.29 |
| `think1__alphazero_v2` | 200 | 3-98-99 | 0.2600 | [0.2041, 0.3249] | [0.2225, 0.2950] | 0.1750 / 0.3450 | 102.28 |
| `think1__mcts_pure` | 200 | 26-5-169 | 0.1425 | [0.1008, 0.1977] | [0.0975, 0.1925] | 0.1250 / 0.1600 | 109.95 |
| `think1__minimax` | 200 | 0-0-200 | 0.0000 | [0.0000, 0.0188] | [0.0000, 0.0000] | 0.0000 / 0.0000 | 33.57 |
| `think1__rl_policy` | 200 | 2-87-111 | 0.2275 | [0.1749, 0.2904] | [0.1925, 0.2625] | 0.2700 / 0.1850 | 59.1 |
| `think1__supervised` | 200 | 1-91-108 | 0.2325 | [0.1793, 0.2958] | [0.2000, 0.2650] | 0.2550 / 0.2100 | 60.19 |
| `think1__think2` | 200 | 0-20-180 | 0.0500 | [0.0274, 0.0896] | [0.0300, 0.0725] | 0.0550 / 0.0450 | 82.17 |
| `think1__think3` | 200 | 2-8-190 | 0.0300 | [0.0138, 0.0639] | [0.0125, 0.0500] | 0.0350 / 0.0250 | 45.7 |
| `think2__alphabeta` | 200 | 1-0-199 | 0.0050 | [0.0009, 0.0278] | [0.0000, 0.0150] | 0.0100 / 0.0000 | 45.33 |
| `think2__alphago` | 200 | 1-17-182 | 0.0475 | [0.0256, 0.0864] | [0.0275, 0.0700] | 0.0550 / 0.0400 | 93.47 |
| `think2__alphazero_v2` | 200 | 2-184-14 | 0.4700 | [0.4020, 0.5391] | [0.4500, 0.4875] | 0.4350 / 0.5050 | 167.44 |
| `think2__mcts_pure` | 200 | 132-67-1 | 0.8275 | [0.7691, 0.8736] | [0.7975, 0.8575] | 0.8250 / 0.8300 | 95.57 |
| `think2__minimax` | 200 | 1-0-199 | 0.0050 | [0.0009, 0.0278] | [0.0000, 0.0150] | 0.0100 / 0.0000 | 44.91 |
| `think2__rl_policy` | 200 | 3-136-61 | 0.3550 | [0.2920, 0.4235] | [0.3200, 0.3900] | 0.3600 / 0.3500 | 78.28 |
| `think2__supervised` | 200 | 1-146-53 | 0.3700 | [0.3061, 0.4388] | [0.3400, 0.4000] | 0.3950 / 0.3450 | 80.59 |
| `think2__think3` | 200 | 54-121-25 | 0.5725 | [0.5032, 0.6391] | [0.5300, 0.6150] | 0.6000 / 0.5450 | 87.79 |
| `think3__alphabeta` | 200 | 1-0-199 | 0.0050 | [0.0009, 0.0278] | [0.0000, 0.0150] | 0.0000 / 0.0100 | 36.05 |
| `think3__alphago` | 200 | 0-17-183 | 0.0425 | [0.0221, 0.0801] | [0.0250, 0.0625] | 0.0600 / 0.0250 | 51.52 |
| `think3__alphazero_v2` | 200 | 10-154-36 | 0.4350 | [0.3682, 0.5043] | [0.4025, 0.4675] | 0.4300 / 0.4400 | 97.16 |
| `think3__mcts_pure` | 200 | 182-15-3 | 0.9475 | [0.9073, 0.9708] | [0.9225, 0.9700] | 0.9500 / 0.9450 | 51.72 |
| `think3__minimax` | 200 | 1-0-199 | 0.0050 | [0.0009, 0.0278] | [0.0000, 0.0150] | 0.0000 / 0.0100 | 35.91 |
| `think3__rl_policy` | 200 | 1-102-97 | 0.2600 | [0.2041, 0.3249] | [0.2250, 0.2950] | 0.3100 / 0.2100 | 54.47 |
| `think3__supervised` | 200 | 0-117-83 | 0.2925 | [0.2338, 0.3590] | [0.2575, 0.3275] | 0.3450 / 0.2400 | 53.17 |

## Hiệu năng từng Agent (Telemetry)

| Agent | N | Score | ms TB | ms Trung vị | ms p95 | Nodes TB | Cutoffs TB | Sims | Timeout / Error |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `minimax` | 2000 | 0.8678 | 414.442 | 406.495 | 869.775 | 890.767 | - | - | 0 / 0 |
| `alphabeta` | 2000 | 0.8662 | 62.565 | 51.858 | 140.639 | 145.847 | 30.918 | - | 0 / 0 |
| `alphago` | 2000 | 0.8257 | 991.573 | 973.115 | 1602.664 | - | - | 30 | 0 / 0 |
| `rl_policy` | 2000 | 0.5353 | 11.668 | 3.227 | 5.023 | - | - | - | 0 / 0 |
| `supervised` | 2000 | 0.5315 | 14.539 | 4.059 | 6.811 | - | - | - | 0 / 0 |
| `think2` | 2000 | 0.4550 | 10.445 | 8.751 | 23.608 | - | - | - | 0 / 0 |
| `think3` | 2000 | 0.4353 | 8.435 | 8.863 | 16.887 | - | - | - | 0 / 0 |
| `alphazero_v2` | 2000 | 0.4108 | 972.151 | 880.759 | 1876.043 | - | - | 30 | 0 / 0 |
| `mcts_pure` | 2000 | 0.2940 | 409.344 | 431.146 | 526.673 | - | - | 30 | 0 / 0 |
| `think1` | 2000 | 0.1688 | 0.694 | 0.65 | 1.237 | - | - | - | 0 / 0 |
| `random` | 2000 | 0.1098 | 0.351 | 0.338 | 0.595 | - | - | - | 0 / 0 |

## Phân bố lý do kết thúc

| Lý do | Số ván | Tỷ lệ |
| :--- | ---: | ---: |
| `checkmate` | 7930 | 72.09% |
| `threefold_repetition` | 2190 | 19.91% |
| `stalemate` | 664 | 6.04% |
| `insufficient_material` | 115 | 1.05% |
| `fifty_moves` | 77 | 0.70% |
| `max_plies_adjudication` | 24 | 0.22% |
