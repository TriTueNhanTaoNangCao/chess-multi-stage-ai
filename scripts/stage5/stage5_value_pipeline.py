"""Reproduce the Stage 5 three-phase Value workflow on Kaggle, one step at a time.

This wrapper was created after the experiment for reproducibility; it is not the
original Kaggle notebook and does not guarantee a byte-identical checkpoint.
It delegates to the existing collection and Value training/evaluation CLIs and
never chains collect, train, and test automatically.

Examples::

  python -u scripts/stage5/stage5_value_pipeline.py collect --bundle-root /kaggle/input/stage5-job --output-dir /kaggle/working/value-2071
  python -u scripts/stage5/stage5_value_pipeline.py train --bundle-root /kaggle/input/stage5-job --output-dir /kaggle/working/value-model --dataset /kaggle/input/value-data/dataset.npz --manifest /kaggle/input/value-data/manifest.json
  python -u scripts/stage5/stage5_value_pipeline.py evaluate-test --bundle-root /kaggle/input/stage5-job --output-dir /kaggle/working/value-test --dataset /kaggle/input/value-data/dataset.npz --manifest /kaggle/input/value-data/manifest.json --checkpoint /kaggle/input/value-model/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COLLECT_RUNNER = HERE / "stage5_alphago_value_job.py"
REQUIRED_BUNDLE_FILES = (
    "source/scripts/stage5/__init__.py",
    "source/scripts/stage5/stage5_alphago_value_data.py",
    "source/scripts/stage5/stage5_alphago_value_train.py",
    "source/scripts/stage5/stage5_value_io.py",
    "source/training/stage5/alphago_value_data.py",
    "source/training/stage5/checkpoint.py",
    "source/training/stage5/updates.py",
    "source/training/stage5/value_training.py",
    "source/models/cnn_policy.py",
    "source/models/value_net.py",
    "checkpoints/stage4/best_lichess.pt",
    "checkpoints/stage5/batch_comparison/b16/best.pt",
)


def validate_bundle(bundle_root: Path) -> Path:
    root = bundle_root.resolve()
    missing = [relative for relative in REQUIRED_BUNDLE_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Bundle is missing required files: {missing}")
    return root


def run(command: list[str], *, cwd: Path) -> None:
    print("COMMAND " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd)


def collect(args: argparse.Namespace, bundle_root: Path, output_dir: Path) -> None:
    if (output_dir / "dataset.npz").exists() or (output_dir / "manifest.json").exists():
        raise FileExistsError("Collection output is already complete or partially finalized")
    command = [
        sys.executable, "-u", str(COLLECT_RUNNER),
        "--bundle-root", str(bundle_root), "--output-dir", str(output_dir),
    ]
    if args.resume_shards is not None:
        command.extend(("--resume-shards", str(args.resume_shards.resolve())))
    run(command, cwd=REPO_ROOT)


def train(args: argparse.Namespace, bundle_root: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output_dir}")
    cli = bundle_root / "source/scripts/stage5/stage5_alphago_value_train.py"
    command = [
        sys.executable, "-u", str(cli), "train",
        "--dataset", str(args.dataset.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--checkpoint-dir", str(output_dir), "--device", args.device,
        "--value-seed", "2072", "--value-lr", "1e-4",
        "--batch-size", "128", "--max-epochs", "100", "--patience", "10",
    ]
    run(command, cwd=bundle_root / "source")


def evaluate_test(args: argparse.Namespace, bundle_root: Path, output_dir: Path) -> None:
    result_path = output_dir / "test_metrics.json"
    if result_path.exists():
        raise FileExistsError(f"Test evaluation already exists: {result_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {output_dir}")
    cli = bundle_root / "source/scripts/stage5/stage5_alphago_value_train.py"
    command = [
        sys.executable, "-u", str(cli), "evaluate-test",
        "--dataset", str(args.dataset.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--output", str(result_path), "--device", args.device,
        "--batch-size", "128",
    ]
    run(command, cwd=bundle_root / "source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bundle-root", required=True, type=Path)
    common.add_argument("--output-dir", required=True, type=Path)
    collect_parser = subparsers.add_parser("collect", parents=[common])
    collect_parser.add_argument("--resume-shards", type=Path)
    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--dataset", required=True, type=Path)
    train_parser.add_argument("--manifest", required=True, type=Path)
    train_parser.add_argument("--device", default="cuda")
    test_parser = subparsers.add_parser("evaluate-test", parents=[common])
    test_parser.add_argument("--dataset", required=True, type=Path)
    test_parser.add_argument("--manifest", required=True, type=Path)
    test_parser.add_argument("--checkpoint", required=True, type=Path)
    test_parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = validate_bundle(args.bundle_root)
    output_dir = args.output_dir.resolve()
    if args.command == "collect":
        collect(args, bundle_root, output_dir)
    elif args.command == "train":
        train(args, bundle_root, output_dir)
    else:
        evaluate_test(args, bundle_root, output_dir)


if __name__ == "__main__":
    main()
