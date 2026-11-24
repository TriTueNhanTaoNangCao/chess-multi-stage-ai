"""Fresh Stage 4 300k training using the existing SL trainer and NPY memmaps.

Only a new output directory is accepted. The trainer writes stage4_cnn_best.pt
and stage4_cnn_last.pt there; no existing stage4_policy.pt is replaced.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.dataset import load_dataset
from training.stage4_train_sl import Stage4TrainConfig, train


def inspect_npy_dataset(dataset_dir: Path) -> dict:
    """Inspect metadata only, without converting or scanning full sample arrays.

load_dataset prioritizes NPZ over NPY, so refuse NPZ to guarantee the memmap
branch. ChessDataset copies/transposes one sample in __getitem__; the existing
trainer batches those samples with DataLoader(batch_size=64, num_workers=0).
Shuffle index allocation and OS file caching are separate from sample loading.
"""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Dataset manifest is not complete")
    status_path = dataset_dir / "build_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError("Dataset build_status is not complete")
    for key, expected in (("encoder_version", "hwc14-white-v1"),
                          ("action_version", "az73-rankflip-v1"), ("num_actions", 4672)):
        if manifest.get(key) != expected:
            raise ValueError(f"Incompatible dataset metadata: {key}")
    report = {}
    for split in ("train", "val"):
        if (dataset_dir / f"{split}.npz").exists():
            raise ValueError(f"{split}.npz would bypass NPY memmaps; use an NPY-only directory")
        dataset = load_dataset(dataset_dir, split=split, to_torch=False)
        try:
            arrays = {"states": dataset.states, "targets": dataset.targets,
                      "legal_masks": dataset.legal_masks}
            if not all(isinstance(array, np.memmap) for array in arrays.values()):
                raise ValueError(f"{split} arrays are not NPY memmaps")
            count = len(dataset)
            if count == 0:
                raise ValueError(f"{split} is empty")
            if dataset.states.shape != (count, 8, 8, 14) or dataset.states.dtype != np.float32:
                raise ValueError(f"{split} states must be float32 HWC (N,8,8,14)")
            if dataset.targets.shape != (count,) or dataset.targets.dtype.kind not in "iu":
                raise ValueError(f"{split} targets must be integer (N,)")
            if dataset.legal_masks.shape != (count, 4672) or dataset.legal_masks.dtype != np.bool_:
                raise ValueError(f"{split} legal masks must be bool (N,4672)")
            expected_count = manifest.get("num_samples", {}).get(split)
            if expected_count is not None and expected_count != count:
                raise ValueError(f"{split} sample count differs from manifest")
            report[split] = {
                "samples": count, "mmap_mode": "r",
                "files": {key: str(array.filename) for key, array in arrays.items()},
            }
        finally:
            # Release preflight mappings before the trainer opens its own.
            for array in vars(dataset).values():
                if isinstance(array, np.memmap):
                    array._mmap.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Must be a NEW directory, not checkpoints/stage4 itself")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"Refusing to overwrite existing output: {output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    report = inspect_npy_dataset(dataset_dir)
    config = Stage4TrainConfig(
        dataset_dir=str(dataset_dir), output_dir=str(output_dir),
        device=args.device, lr=1e-3, batch_size=64, seed=42, epochs=20,
        early_stopping_patience=3, early_stopping_min_delta=0.0,
        model_config={"in_channels": 14, "num_actions": 4672,
                      "hidden_dim": 128, "fc_dim": 512, "dropout": 0.2},
        resume_checkpoint=None, evaluate_test=False,
    )
    # exist_ok=False also prevents racing with another fresh run.
    output_dir.mkdir(parents=True)
    run_config = {
        "training_config": asdict(config), "npy_preflight": report,
        "dataset_manifest_sha256": hashlib.sha256(
            (dataset_dir / "manifest.json").read_bytes()).hexdigest(),
        "checkpoint_selection": "lowest validation loss; min_delta=0",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_config, indent=2), flush=True)
    summary = train(config)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
