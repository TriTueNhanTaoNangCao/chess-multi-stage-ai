"""Fresh Stage 5 B16 -> three-phase collection -> Value -> Paper Policy.

Uses the existing trainers/collector and validates each output before advancing.
Writes b16/last.pt, value_data/{dataset.npz,manifest.json}, value/best.pt,
and paper_policy/last.pt. No Arena or Policy best selection. Existing run roots
are refused; Paper Policy is fresh-init from B16 with a frozen Value baseline.
"""
import argparse
import contextlib
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COLLECTION = {"total_games": 24576, "shard_size": 512, "u_min": 1,
              "u_max": 100, "max_plies": 300, "seed": 2071}
VALUE_TRAINING = {"seed": 2072, "learning_rate": 1e-4, "batch_size": 128,
                  "max_epochs": 100, "early_stopping_patience": 10}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_cli(module, arguments, log_path):
    """Run exactly one existing CLI, tee its output, and stop on failure."""
    command = [sys.executable, "-u", "-m", module, *arguments]
    print("COMMAND " + subprocess.list2cmdline(command), flush=True)
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        with subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def check_b16(path):
    from training.stage5.checkpoint import load_stage5_policy_checkpoint

    model, provenance = load_stage5_policy_checkpoint(path, device="cpu")
    expected = {"completed_iteration": 400, "baseline": "zero",
                "value_mode": "disabled", "seed": 42, "claim_draw": True}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"B16 output mismatch: {key}")
    del model
    return {"path": str(path), "sha256": sha256_file(path), "provenance": provenance}


def collect_value_data(stage4, b16, output, device, expected_hashes):
    from scripts.stage5.stage5_alphago_value_data import existing_shards
    from scripts.stage5.stage5_alphago_value_train import load_three_phase_inputs

    data_dir = output / "value_data"
    arguments = ["--sl-checkpoint", str(stage4), "--rl-checkpoint", str(b16),
                 "--data-dir", str(data_dir), "--device", device]
    for key, value in COLLECTION.items():
        arguments.extend(("--" + key.replace("_", "-"), str(value)))
    run_cli("scripts.stage5.stage5_alphago_value_data", arguments, output / "collection.log")
    dataset, manifest_path = data_dir / "dataset.npz", data_dir / "manifest.json"
    sources, source_data = load_three_phase_inputs(dataset, manifest_path)
    for key, digest in expected_hashes.items():
        if sources[key]["sha256"] != digest:
            raise ValueError(f"Collector {key} checkpoint provenance mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in COLLECTION.items():
        if manifest["config"]["collection"].get(key) != value:
            raise ValueError(f"Collector output config mismatch: {key}")
    if manifest["collection"]["games"] != COLLECTION["total_games"]:
        raise ValueError("Collector did not complete all 24576 attempted games")
    if len(existing_shards(data_dir / "shards")) != COLLECTION["total_games"] // COLLECTION["shard_size"]:
        raise ValueError("Collector did not write exactly 48 shards")
    return {"source_data": source_data, "source_policies": sources,
            "collection": manifest["collection"], "split": manifest["split"]}


def train_value(output, device, data_check):
    import torch
    from training.stage5.checkpoint import load_value_only_checkpoint

    data_dir, checkpoint_dir = output / "value_data", output / "value"
    run_cli("scripts.stage5.stage5_alphago_value_train", [
        "train", "--dataset", str(data_dir / "dataset.npz"),
        "--manifest", str(data_dir / "manifest.json"),
        "--checkpoint-dir", str(checkpoint_dir), "--device", device,
        "--value-seed", str(VALUE_TRAINING["seed"]),
        "--value-lr", str(VALUE_TRAINING["learning_rate"]),
        "--batch-size", str(VALUE_TRAINING["batch_size"]),
        "--max-epochs", str(VALUE_TRAINING["max_epochs"]),
        "--patience", str(VALUE_TRAINING["early_stopping_patience"]),
    ], output / "value_training.log")
    best = checkpoint_dir / "best.pt"
    model, provenance = load_value_only_checkpoint(best, device="cpu", claim_draw=True)
    del model
    payload = torch.load(best, map_location="cpu", weights_only=False)
    for key, value in {**VALUE_TRAINING, "device": device}.items():
        if payload["training_config"].get(key) != value:
            raise ValueError(f"Value training config mismatch: {key}")
    if payload.get("source_policies") != data_check["source_policies"]:
        raise ValueError("Value source policies differ from the collected data")
    for key in ("dataset_sha256", "manifest_sha256"):
        if payload["source_data"].get(key) != data_check["source_data"][key]:
            raise ValueError(f"Value source data mismatch: {key}")
    if (not isinstance(payload.get("epoch"), int)
            or not 1 <= payload["epoch"] <= VALUE_TRAINING["max_epochs"]
            or not math.isfinite(float(payload["val_mse"]))):
        raise ValueError("Invalid Value best epoch or validation MSE")
    summary = json.loads((checkpoint_dir / "training_summary.json").read_text(encoding="utf-8"))
    best_hash = sha256_file(best)
    if summary["best_checkpoint"]["sha256"] != best_hash:
        raise ValueError("Value best checkpoint differs from its training summary")
    return {"path": str(best), "sha256": best_hash, "provenance": provenance,
            "epoch": payload["epoch"], "val_mse": payload["val_mse"]}


def b16_config(stage4_checkpoint: Path, output_dir: Path, device: str):
    from training.stage5_train_rl import Stage5TrainConfig

    return Stage5TrainConfig(
        stage4_checkpoint=stage4_checkpoint,
        output_dir=output_dir / "b16",
        resume_checkpoint=None,
        policy_init_checkpoint=None,
        value_checkpoint=None,
        device=device,
        seed=42,
        iterations=400,
        games_per_iteration=16,
        max_plies=300,
        chunk_size=16,
        policy_lr=1e-5,
        value_lr=1e-4,  # Dormant in disabled mode; retained from the B16 config.
        baseline="zero",
        value_mode="disabled",
        opponent_mode="pool",
        opponent_pool_size=4,
        opponent_snapshot_interval=40,
        start_mode="standard",
        mixed_start_manifest=None,
    )


def paper_config(stage4_checkpoint: Path, output_dir: Path, device: str):
    from training.stage5_train_rl import Stage5TrainConfig

    return Stage5TrainConfig(
        stage4_checkpoint=stage4_checkpoint,
        policy_init_checkpoint=output_dir / "b16/last.pt",
        value_checkpoint=output_dir / "value/best.pt",
        output_dir=output_dir / "paper_policy",
        resume_checkpoint=None,
        device=device,
        seed=2073,
        iterations=80,
        games_per_iteration=32,
        max_plies=300,
        chunk_size=8,
        policy_lr=1e-5,
        value_lr=1e-3,
        baseline="value",
        value_mode="frozen",
        opponent_mode="pool",
        opponent_pool_size=4,
        opponent_snapshot_interval=5,
        start_mode="standard",
        mixed_start_manifest=None,
    )


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def train_paper_policy(stage4, output, device, expected_hashes):
    import torch
    from training.stage5.checkpoint import load_stage5_policy_checkpoint, load_value_only_checkpoint
    from training.stage5_train_rl import train

    config = paper_config(stage4, output, device)
    config.validate()
    sources = {"stage4": stage4, "b16": config.policy_init_checkpoint,
               "value": config.value_checkpoint}
    for key, path in sources.items():
        if sha256_file(path) != expected_hashes[key]:
            raise ValueError(f"Paper Policy source hash changed before training: {key}")
    directory = Path(config.output_dir)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite Paper Policy run: {directory}")
    directory.mkdir()
    save_json(directory / "run_config.json", asdict(config))
    log_path = directory / "training.log"
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
             contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                print("PAPER_POLICY_CONFIG " + json.dumps(asdict(config), default=str), flush=True)
                result = train(config)
                if (len(result.history) != 80 or result.history[-1]["iteration"] != 80
                        or any(row["games"] != 32 for row in result.history)):
                    raise ValueError("Paper Policy did not complete 80 iterations of 32 games")
                if result.value_model.training or any(p.requires_grad for p in result.value_model.parameters()):
                    raise ValueError("Paper Value baseline is not frozen/eval")
                frozen, _ = load_value_only_checkpoint(config.value_checkpoint, device="cpu", claim_draw=True)
                actual = result.value_model.state_dict()
                if any(not torch.equal(actual[key].detach().cpu(), tensor)
                       for key, tensor in frozen.state_dict().items()):
                    raise ValueError("Frozen Value weights changed during Paper Policy training")
                del frozen
                save_json(directory / "history.json", result.history)
                for row in result.history:
                    print(json.dumps(row), flush=True)
                del result
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                last = directory / "last.pt"
                model, provenance = load_stage5_policy_checkpoint(last, device="cpu")
                del model
                for key, value in {"completed_iteration": 80, "seed": 2073,
                                   "baseline": "value", "value_mode": "frozen",
                                   "claim_draw": True}.items():
                    if provenance.get(key) != value:
                        raise ValueError(f"Paper Policy output mismatch: {key}")
                for key, path in sources.items():
                    if sha256_file(path) != expected_hashes[key]:
                        raise ValueError(f"Paper Policy source checkpoint changed: {key}")
                check = {"path": str(last), "sha256": sha256_file(last),
                         "provenance": provenance, "training_config": asdict(config),
                         "source_sha256": expected_hashes, "frozen_value_unchanged": True,
                         "completed_iterations": 80, "attempted_games": 80 * 32,
                         "log": str(log_path)}
                print("PAPER_POLICY_VERIFIED " + json.dumps(check, default=str), flush=True)
                return check
            except BaseException:
                traceback.print_exc()
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="NEW run directory for b16/, value_data/, value/ and paper_policy/")
    parser.add_argument("--stage4-checkpoint", type=Path,
                        default=Path("checkpoints/stage4/stage4_policy.pt"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite an existing run: {output}")
    checkpoint = args.stage4_checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Stage 4 checkpoint not found: {checkpoint}")

    import torch
    from training.stage5_train_rl import train

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    config = b16_config(checkpoint, output, args.device)
    config.validate()
    stage4_hash = sha256_file(checkpoint)
    # Reserve a fresh root before handing its b16 subdirectory to the trainer.
    output.mkdir(parents=True)
    run_config = {
        "steps": ["b16", "value_data", "value", "paper_policy"], "attempted_games_target": 400 * 16,
        "training_config": asdict(config), "stage4_sha256": stage4_hash,
        "collection": {**COLLECTION, "device": args.device},
        "value_training": {**VALUE_TRAINING, "device": args.device},
        "paper_policy_training": asdict(paper_config(checkpoint, output, args.device)),
        "paper_policy_attempted_games_target": 80 * 32,
        "claim_draw": "inherited from Stage 4 checkpoint metadata",
        "checkpoint_selection": "B16/Paper Policy: last iteration; Value: lowest validation MSE",
    }
    save_json(output / "run_config.json", run_config)
    print(json.dumps(run_config, indent=2, default=str), flush=True)
    result = train(config)
    if (len(result.history) != 400 or result.history[-1]["iteration"] != 400
            or any(row["games"] != 16 for row in result.history)):
        raise ValueError("B16 training did not complete 400 iterations of 16 games")
    save_json(output / "b16/history.json", result.history)
    print(f"B16 complete: {len(result.history)} iterations; {output / 'b16/last.pt'}",
          flush=True)
    # Do not retain B16 models/pool on GPU while the next CLI uses that device.
    del result
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    b16 = output / "b16/last.pt"
    checks = {"b16": check_b16(b16)}
    if sha256_file(checkpoint) != stage4_hash:
        raise ValueError("Stage 4 checkpoint changed during B16 training")
    save_json(output / "steps.json", checks)
    checks["value_data"] = collect_value_data(
        checkpoint, b16, output, args.device,
        {"sl": stage4_hash, "rl": checks["b16"]["sha256"]},
    )
    save_json(output / "steps.json", checks)
    checks["value"] = train_value(output, args.device, checks["value_data"])
    if sha256_file(checkpoint) != stage4_hash or sha256_file(b16) != checks["b16"]["sha256"]:
        raise ValueError("Source checkpoints changed during Value collection/training")
    save_json(output / "steps.json", checks)
    print(f"Value complete and verified: {output / 'value/best.pt'}", flush=True)
    checks["paper_policy"] = train_paper_policy(checkpoint, output, args.device, {
        "stage4": stage4_hash, "b16": checks["b16"]["sha256"],
        "value": checks["value"]["sha256"],
    })
    save_json(output / "steps.json", checks)
    print(f"Paper Policy complete and verified: {output / 'paper_policy/last.pt'}", flush=True)
    return checks


if __name__ == "__main__":
    main()
