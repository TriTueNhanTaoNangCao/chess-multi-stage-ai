"""Kiểm tra và lập danh mục toàn bộ checkpoint (T1 - R1, R5).

Chạy từ gốc repo:
    python analysis/inspect_checkpoints.py
"""
import hashlib
import os
from pathlib import Path
import sys
import torch

DEFAULT_PATHS = [
    "checkpoints/stage4/best_lichess.pt",
    "checkpoints/stage5/paper_frozen_2073/last.pt",
    "checkpoints/stage6/policy.pt",
    "checkpoints/stage6/value.pt",
    "checkpoints/stage7/best.pt",
    "checkpoints/stage7_v2/best.pt",
]

EXPECTED_METADATA = {
    "CNNPolicyNetwork": {
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": 4672,
        "claim_draw": True,
    },
    "ValueNetwork": {
        "encoder_version": "hwc14-white-v1",
        "value_perspective": "side_to_move",
        "claim_draw": True,
    },
    "DualHeadResNet": {
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": 4672,
        "value_perspective": "side_to_move",
        "claim_draw": True,
    },
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_file(path_str: str) -> dict:
    p = Path(path_str)
    info = {
        "path": path_str,
        "exists": p.is_file(),
        "size_mb": 0.0,
        "sha256": "THIẾU",
        "model_class": "THIẾU",
        "metadata_status": "THIẾU",
        "training_info": "THIẾU",
        "errors": [],
    }
    if not p.is_file():
        info["errors"].append("File không tồn tại")
        return info

    size = p.stat().st_size
    info["size_mb"] = round(size / (1024 * 1024), 2)
    info["sha256"] = compute_sha256(p)

    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        info["metadata_status"] = f"LỖI LOAD ({type(exc).__name__})"
        info["errors"].append(str(exc))
        return info

    if not isinstance(payload, dict):
        info["metadata_status"] = "KHÔNG PHẢI DICT"
        info["errors"].append("Payload không phải là dict")
        return info

    model_class = payload.get("model_class")
    info["model_class"] = model_class if model_class else "THIẾU"

    meta_issues = []
    if "state_dict" not in payload:
        meta_issues.append("thiếu state_dict")

    if model_class in EXPECTED_METADATA:
        req = EXPECTED_METADATA[model_class]
        for key, expected_val in req.items():
            actual_val = payload.get(key)
            if actual_val is None:
                meta_issues.append(f"thiếu {key}")
            elif actual_val != expected_val:
                meta_issues.append(f"sai {key} (nhận {actual_val!r}, mong muốn {expected_val!r})")
    else:
        meta_issues.append(f"model_class không xác định: {model_class!r}")

    info["metadata_status"] = "HỢP LỆ" if not meta_issues else f"CẢNH BÁO: {', '.join(meta_issues)}"
    if meta_issues:
        info["errors"].extend(meta_issues)

    # Training config & info
    tc = payload.get("training_config") or {}
    t_parts = []
    seed = payload.get("seed") or tc.get("seed")
    if seed is not None:
        t_parts.append(f"seed={seed}")
    iters = payload.get("completed_iteration") or payload.get("step") or payload.get("epoch") or tc.get("iterations") or tc.get("num_iterations") or tc.get("epochs")
    if iters is not None:
        t_parts.append(f"iters/step/epoch={iters}")
    lr = tc.get("lr") or tc.get("learning_rate")
    if lr is not None:
        t_parts.append(f"lr={lr}")
    metrics = payload.get("metrics") or {}
    if "val_loss" in metrics:
        t_parts.append(f"val_loss={metrics['val_loss']:.4f}")
    if "val_mse" in payload:
        t_parts.append(f"val_mse={payload['val_mse']:.4f}")

    info["training_info"] = ", ".join(t_parts) if t_parts else "Không có training_config"
    return info


def main():
    target_paths = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PATHS
    results = []

    print("=" * 80)
    print("  KIỂM TRA DANH MỤC CHECKPOINT (TASK T1)")
    print("=" * 80)

    for path_str in target_paths:
        res = inspect_file(path_str)
        results.append(res)
        status_symbol = "✓" if res["exists"] and "HỢP LỆ" in res["metadata_status"] else ("✗" if not res["exists"] else "⚠")
        print(f"[{status_symbol}] {res['path']}")
        if res["exists"]:
            print(f"    - Dung lượng: {res['size_mb']} MB | SHA256: {res['sha256']}")
            print(f"    - Kiểu mạng: {res['model_class']} | Metadata: {res['metadata_status']}")
            print(f"    - Huấn luyện: {res['training_info']}")
        else:
            print("    - TRẠNG THÁI: THIẾU (Không tìm thấy file)")
        print("-" * 80)

    # So sánh SHA256 giữa stage6/policy.pt và stage5/.../last.pt
    sha_st6_policy = None
    sha_st5_frozen = None
    for r in results:
        if r["path"].endswith("stage6/policy.pt"):
            sha_st6_policy = r["sha256"]
        if "paper_frozen_2073/last.pt" in r["path"]:
            sha_st5_frozen = r["sha256"]

    match_result = "KHÔNG XÁC ĐỊNH (thiếu file)"
    if sha_st6_policy and sha_st5_frozen and sha_st6_policy != "THIẾU" and sha_st5_frozen != "THIẾU":
        match_result = "TRÙNG" if sha_st6_policy == sha_st5_frozen else "KHÁC"

    print(f"SO SÁNH SHA256: stage6/policy.pt vs stage5/paper_frozen_2073/last.pt -> {match_result}")
    print("=" * 80)

    # Xuất file Markdown results/checkpoint_inventory.md
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "checkpoint_inventory.md"

    md_lines = [
        "# Danh Mục Kiểm Kê Checkpoint Mô Hình (Task T1)",
        "",
        f"- So sánh SHA256 `stage6/policy.pt` vs `stage5/paper_frozen_2073/last.pt`: **{match_result}**",
        "",
        "| Đường dẫn | Tồn tại | MB | SHA256 | Model Class | Metadata | Thông số huấn luyện |",
        "| :--- | :---: | ---: | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        exists_str = "Có" if r["exists"] else "**THIẾU**"
        sha_str = f"`{r['sha256']}`" if r["sha256"] != "THIẾU" else "**THIẾU**"
        md_lines.append(
            f"| `{r['path']}` | {exists_str} | {r['size_mb']} | {sha_str} | `{r['model_class']}` | {r['metadata_status']} | {r['training_info']} |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nĐã xuất kết quả kiểm kê vào: {out_md.resolve()}")


if __name__ == "__main__":
    main()
