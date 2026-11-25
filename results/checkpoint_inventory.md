# Danh Mục Kiểm Kê Checkpoint Mô Hình (Task T1)

- So sánh SHA256 `stage6/policy.pt` vs `stage5/paper_frozen_2073/last.pt`: **TRÙNG**

| Đường dẫn | Tồn tại | MB | SHA256 | Model Class | Metadata | Thông số huấn luyện |
| :--- | :---: | ---: | :--- | :--- | :--- | :--- |
| `checkpoints/stage4/best_lichess.pt` | Có | 79.03 | `99ee0a6bcc2a2d80fa794bb90464ed2a68b57151622bdfff5388da976595073b` | `CNNPolicyNetwork` | HỢP LỆ | seed=42, iters/step/epoch=803694, lr=0.001 |
| `checkpoints/stage5/paper_frozen_2073/last.pt` | Có | 184.93 | `629e06baa336acd3b2f9cdb0cb87590524dd54ccb217b3a496967be7e4ddf7e4` | `CNNPolicyNetwork` | HỢP LỆ | seed=2073, iters/step/epoch=80 |
| `checkpoints/stage6/policy.pt` | Có | 184.93 | `629e06baa336acd3b2f9cdb0cb87590524dd54ccb217b3a496967be7e4ddf7e4` | `CNNPolicyNetwork` | HỢP LỆ | seed=2073, iters/step/epoch=80 |
| `checkpoints/stage6/value.pt` | Có | 0.48 | `6731b88954be429f9bec0b2213227e8d9ef9c75637db8f9c4bf539435d518d0b` | `ValueNetwork` | HỢP LỆ | seed=2072, iters/step/epoch=2, lr=0.0001, val_mse=0.2429 |
| `checkpoints/stage7/best.pt` | Có | 22.91 | `6c8e5302898c2eb6889890aacf09845215a1d1f79814b91a14267302f74b4ca8` | `DualHeadResNet` | HỢP LỆ | seed=42, iters/step/epoch=12, lr=0.001 |
| `checkpoints/stage7_v2/best.pt` | Có | 25.17 | `8aec76fe7b2cd7aac4c42cd55daee17754d54cea92565712bab302efe8948d13` | `DualHeadResNet` | HỢP LỆ | seed=42, iters/step/epoch=300, lr=0.02 |
