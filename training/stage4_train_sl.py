"""
Stage 4: Supervised Learning Training for CNNPolicyNetwork.
Trains policy network on PGN dataset using Cross-Entropy loss over legal moves,
validates with top-1/top-5 metrics, and serializes checkpoints conforming to docs/HANDOFF.md.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    F = None
    DataLoader = None

from games.chess.encoder import NUM_ACTIONS
from models.cnn_policy import CNNPolicyNetwork
from training.dataset import dataset_split_exists, load_dataset, ChessDataset


@dataclass
class Stage4TrainConfig:
    """Configuration for Stage 4 Policy SL training."""
    dataset_dir: str
    output_dir: str = "checkpoints"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    device: str = "cpu"
    evaluate_test: bool = False
    model_config: Optional[Dict[str, Any]] = None
    # Resume
    resume_checkpoint: Optional[str] = None
    # Early stopping
    early_stopping_patience: int = 0   # 0 = disabled
    early_stopping_min_delta: float = 0.0


def validate_batch(batch: Dict[str, Any]) -> None:
    """
    Ensure every sample in the batch has at least one legal action
    and target belongs to the legal mask.
    """
    legal_mask = batch["legal_mask"]  # (B, 4672) bool
    target = batch["target"]          # (B,) long

    # 1. At least one legal move per position
    if not legal_mask.any(dim=-1).all():
        raise ValueError("Invalid batch: found position with no legal actions in legal_mask")

    # 2. Target must be marked legal
    target_in_mask = legal_mask.gather(1, target.unsqueeze(1)).squeeze(1)
    if not target_in_mask.all():
        raise ValueError("Invalid batch: target action is not marked as legal in legal_mask")


def train_one_epoch(
    model: CNNPolicyNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    Returns weighted sample-average metrics: loss, top-1 accuracy, top-5 accuracy.
    """
    model.train()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for batch in loader:
        state = batch["state"].to(device)        # (B, 14, 8, 8)
        target = batch["target"].to(device)      # (B,)
        legal_mask = batch["legal_mask"].to(device) # (B, 4672)

        validate_batch(batch)

        optimizer.zero_grad()
        logits = model(state)  # raw logits (B, 4672)

        # Mask illegal actions with large negative value before softmax
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        loss = F.cross_entropy(masked_logits, target)
        loss.backward()
        optimizer.step()

        b_size = target.size(0)
        total_loss += loss.item() * b_size

        # Top-1 accuracy over legal moves
        pred_top1 = masked_logits.argmax(dim=-1)
        total_top1 += (pred_top1 == target).sum().item()

        # Top-5 accuracy over legal moves
        k = min(5, masked_logits.size(-1))
        _, pred_topk = masked_logits.topk(k, dim=-1)
        total_top5 += (pred_topk == target.unsqueeze(1)).any(dim=-1).sum().item()

        total_samples += b_size

    if total_samples == 0:
        return {"loss": 0.0, "top1": 0.0, "top5": 0.0, "samples": 0}

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
        "samples": total_samples,
    }


def evaluate(
    model: CNNPolicyNetwork,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset split in eval mode with torch.no_grad().
    Returns sample-weighted metrics: loss, top-1 accuracy, top-5 accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            state = batch["state"].to(device)
            target = batch["target"].to(device)
            legal_mask = batch["legal_mask"].to(device)

            validate_batch(batch)

            logits = model(state)
            masked_logits = logits.masked_fill(~legal_mask, -1e9)
            loss = F.cross_entropy(masked_logits, target)

            b_size = target.size(0)
            total_loss += loss.item() * b_size

            pred_top1 = masked_logits.argmax(dim=-1)
            total_top1 += (pred_top1 == target).sum().item()

            k = min(5, masked_logits.size(-1))
            _, pred_topk = masked_logits.topk(k, dim=-1)
            total_top5 += (pred_topk == target.unsqueeze(1)).any(dim=-1).sum().item()

            total_samples += b_size

    if total_samples == 0:
        return {"loss": 0.0, "top1": 0.0, "top5": 0.0, "samples": 0}

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
        "samples": total_samples,
    }


def save_checkpoint(
    path: Union[str, Path],
    model: CNNPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    training_config: Dict[str, Any],
    metrics: Dict[str, Any],
    seed: int,
    best_val_loss: Optional[float] = None,
    save_rng_state: bool = True,
) -> None:
    """Save model checkpoint according to docs/HANDOFF.md schema.

    New optional fields (backward-compatible):
      best_val_loss   – best validation loss seen so far, for resume.
      rng_state       – CPU RNG state for exact reproducibility.
      cuda_rng_state  – CUDA RNG state (when CUDA is available).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "format_version": 1,
        "stage": 4,
        "model_class": "CNNPolicyNetwork",
        "model_config": model.get_config(),
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": NUM_ACTIONS,
        "value_perspective": "side_to_move",
        "claim_draw": True,
        "seed": seed,
        "epoch": epoch,
        "step": step,
        "training_config": training_config,
        "metrics": metrics,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss

    if save_rng_state:
        payload["rng_state"] = torch.get_rng_state()
        payload["numpy_rng_state"] = np.random.get_state()
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state()

    torch.save(payload, p)


def load_checkpoint(
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[CNNPolicyNetwork, Dict[str, Any]]:
    """Load and validate checkpoint; returns initialized model in eval mode and payload."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    payload = torch.load(p, map_location=device, weights_only=False)

    # Validate schema compatibility
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported format_version: {payload.get('format_version')}")
    if payload.get("stage") != 4:
        raise ValueError(f"Expected stage 4 checkpoint, got stage {payload.get('stage')}")
    if payload.get("model_class") != "CNNPolicyNetwork":
        raise ValueError(f"Expected CNNPolicyNetwork, got {payload.get('model_class')}")
    if payload.get("encoder_version") != "hwc14-white-v1":
        raise ValueError(f"Incompatible encoder_version: {payload.get('encoder_version')}")
    if payload.get("action_version") != "az73-rankflip-v1":
        raise ValueError(f"Incompatible action_version: {payload.get('action_version')}")

    model_config = payload.get("model_config", {})
    model = CNNPolicyNetwork.from_config(model_config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, payload


def train(config: Union[Dict[str, Any], Stage4TrainConfig]) -> Dict[str, Any]:
    """
    Execute Stage 4 Supervised Learning Training for CNNPolicyNetwork.

    Supports:
      - Fresh training (resume_checkpoint=None)
      - Resume from a checkpoint (resume_checkpoint=path)
        * Restores model weights, optimizer state, epoch counter, best_val_loss.
        * Restores CPU/CUDA RNG state when present in checkpoint.
          If the checkpoint lacks rng_state the resume proceeds, but training
          will NOT be bit-for-bit identical to a continuous run; this is
          noted in the returned summary under 'rng_state_restored'.
      - Early stopping (early_stopping_patience > 0)
        * Stops when validation loss has not improved by more than
          early_stopping_min_delta for <patience> consecutive epochs.

    1. Validates dataset manifest compatibility.
    2. Loads train and validation splits.
    3. Runs epochs with Adam optimizer (resumed from checkpoint if provided).
    4. Computes sample-weighted metrics and saves best and last checkpoints.
    5. Optionally evaluates test split with best checkpoint.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for Stage 4 training but is not installed.")

    cfg = config if isinstance(config, Stage4TrainConfig) else Stage4TrainConfig(**config)

    # 1. Validate dataset metadata
    dataset_dir = Path(cfg.dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("encoder_version") != "hwc14-white-v1":
        raise ValueError(f"Incompatible dataset encoder version: {manifest.get('encoder_version')}")
    if manifest.get("action_version") != "az73-rankflip-v1":
        raise ValueError(f"Incompatible dataset action version: {manifest.get('action_version')}")
    if manifest.get("num_actions") != NUM_ACTIONS:
        raise ValueError(f"Incompatible dataset num_actions: {manifest.get('num_actions')}")

    # 2. Load dataset splits
    train_ds = load_dataset(dataset_dir, split="train", to_torch=True)
    val_ds = load_dataset(dataset_dir, split="val", to_torch=True)

    if len(train_ds) == 0:
        raise ValueError(f"Train split is empty (0 samples) in {dataset_dir}")
    if len(val_ds) == 0:
        raise ValueError(f"Validation split is empty (0 samples) in {dataset_dir}")

    # 3. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # 4. Seed, device, model, optimizer
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device_obj = torch.device(cfg.device)
    model_kwargs = cfg.model_config or {}
    model = CNNPolicyNetwork(**model_kwargs).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss: float = float("inf")
    best_epoch: int = 0
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []
    start_epoch: int = 1
    global_step: int = 0
    rng_state_restored: bool = False
    resume_from: Optional[str] = None

    # ------------------------------------------------------------------ #
    # 5. Optional resume                                                   #
    # ------------------------------------------------------------------ #
    if cfg.resume_checkpoint:
        resume_path = Path(cfg.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"resume_checkpoint not found: {resume_path}. "
                "Refusing to start from scratch instead of resuming."
            )

        # load_checkpoint validates format_version, stage, model_class, encoder/action versions
        resume_model, resume_payload = load_checkpoint(resume_path, device=device_obj)

        # Verify model_config is compatible with what we want to train
        ckpt_model_config = resume_payload.get("model_config", {})
        our_config = CNNPolicyNetwork(**model_kwargs).get_config()
        for key in ("in_channels", "num_actions", "hidden_dim", "fc_dim"):
            if ckpt_model_config.get(key) != our_config.get(key):
                raise ValueError(
                    f"Checkpoint model_config mismatch at '{key}': "
                    f"checkpoint={ckpt_model_config.get(key)!r}, "
                    f"requested={our_config.get(key)!r}. "
                    "Cannot resume — checkpoint is incompatible."
                )

        # Restore model & optimizer (optimizer must be on same device)
        model.load_state_dict(resume_payload["state_dict"], strict=True)
        model.train()  # switch back to train mode (load_checkpoint leaves it in eval)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_obj)

        start_epoch = resume_payload.get("epoch", 0) + 1
        global_step = resume_payload.get("step", 0)
        resume_from = str(resume_path)

        # Restore best_val_loss from checkpoint; refuse to silently reset to inf
        if "best_val_loss" in resume_payload:
            best_val_loss = float(resume_payload["best_val_loss"])
        else:
            # Old checkpoint: derive from metrics if available
            ckpt_metrics = resume_payload.get("metrics", {})
            if "val_loss" in ckpt_metrics:
                best_val_loss = float(ckpt_metrics["val_loss"])
            # else leave as inf — acknowledged in summary

        # Restore RNG state when available
        if "rng_state" in resume_payload:
            torch.set_rng_state(resume_payload["rng_state"])
            if "numpy_rng_state" in resume_payload:
                np.random.set_state(resume_payload["numpy_rng_state"])
            if "cuda_rng_state" in resume_payload and torch.cuda.is_available():
                torch.cuda.set_rng_state(resume_payload["cuda_rng_state"])
            rng_state_restored = True
        # else: rng_state absent in old checkpoint — proceed, document in summary

    # ------------------------------------------------------------------ #
    # 6. Checkpoint paths                                                  #
    # ------------------------------------------------------------------ #
    best_ckpt_path = out_dir / "stage4_cnn_best.pt"
    last_ckpt_path = out_dir / "stage4_cnn_last.pt"

    training_cfg_dict = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "device": cfg.device,
        "dataset_dir": str(dataset_dir),
        "resume_from": resume_from,
        "early_stopping_patience": cfg.early_stopping_patience,
        "early_stopping_min_delta": cfg.early_stopping_min_delta,
    }

    # ------------------------------------------------------------------ #
    # 7. Early stopping state                                              #
    # ------------------------------------------------------------------ #
    patience = cfg.early_stopping_patience
    min_delta = cfg.early_stopping_min_delta
    no_improve_count: int = 0
    stop_reason: str = "max_epochs_reached"

    # ------------------------------------------------------------------ #
    # 8. Training loop                                                     #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device_obj)
        val_metrics = evaluate(model, val_loader, device_obj)
        global_step += len(train_loader)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        history.append(epoch_record)

        # Save last checkpoint (includes RNG state for future resumes)
        save_checkpoint(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            training_config=training_cfg_dict,
            metrics=epoch_record,
            seed=cfg.seed,
            best_val_loss=best_val_loss,
            save_rng_state=True,
        )

        # Update best checkpoint when val loss strictly decreases
        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = dict(epoch_record)
            no_improve_count = 0
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                training_config=training_cfg_dict,
                metrics=best_metrics,
                seed=cfg.seed,
                best_val_loss=best_val_loss,
                save_rng_state=True,
            )
        else:
            no_improve_count += 1

        # Early stopping check
        if patience > 0 and no_improve_count >= patience:
            stop_reason = f"early_stopping_patience_{patience}_epochs_no_improvement"
            break

    else:
        # Loop completed without break
        if patience > 0 and no_improve_count < patience:
            stop_reason = "max_epochs_reached"

    # ------------------------------------------------------------------ #
    # 9. Optional test split evaluation using best checkpoint              #
    # ------------------------------------------------------------------ #
    test_metrics = None
    if cfg.evaluate_test:
        if dataset_split_exists(dataset_dir, "test"):
            test_ds = load_dataset(dataset_dir, split="test", to_torch=True)
            if len(test_ds) > 0:
                best_model, _ = load_checkpoint(best_ckpt_path, device=device_obj)
                test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
                test_metrics = evaluate(best_model, test_loader, device_obj)

    summary: Dict[str, Any] = {
        "stage": 4,
        "model_class": "CNNPolicyNetwork",
        "start_epoch": start_epoch,
        "epochs_trained": cfg.epochs,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "history": history,
        "resume_from": resume_from,
        "rng_state_restored": rng_state_restored,
        "checkpoints": {
            "best": str(best_ckpt_path),
            "last": str(last_ckpt_path),
        },
        "test_metrics": test_metrics,
    }

    summary_file = out_dir / "training_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Stage 4 Policy SL Network")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing manifest.json and NPZ or NPY-array splits",
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--evaluate_test", action="store_true", help="Evaluate test split with best checkpoint")

    args = parser.parse_args()
    config = Stage4TrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        evaluate_test=args.evaluate_test,
    )
    result = train(config)
    print(f"Training completed. Best epoch: {result['best_epoch']}, Best val loss: {result['best_val_loss']:.4f}")
    if result["test_metrics"]:
        print(f"Test metrics: {result['test_metrics']}")

