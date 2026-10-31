"""Fixture-only checks for the post-experiment three-phase Value entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.stage5.stage5_alphago_value_train import (
    COLLECTOR_PURPOSE,
    _git_provenance,
    evaluate_test,
    load_three_phase_inputs,
    run_training,
)
from scripts.stage5.stage5_value_io import _sha256_file
from models.value_net import ValueNetwork


def write_fixture(root: Path) -> tuple[Path, Path, dict]:
    dataset = root / "dataset.npz"
    manifest = root / "manifest.json"
    config_hash = "c" * 64
    states = np.zeros((2, 14, 8, 8), dtype=np.float32)
    targets = np.asarray([[-1.0], [1.0]], dtype=np.float32)
    np.savez_compressed(
        dataset,
        config_sha256=np.asarray(config_hash),
        train_states=states,
        train_targets=targets,
        val_states=states[:1],
        val_targets=np.zeros((1, 1), dtype=np.float32),
        test_states=states[:1],
        test_targets=targets[:1],
    )
    sources = {
        "claim_draw": True,
        "sl": {"stage": 4, "sha256": "s" * 64},
        "rl": {
            "stage": 5, "sha256": "r" * 64,
            "baseline": "zero", "value_mode": "disabled",
        },
    }
    manifest.write_text(
        json.dumps(
            {
                "purpose": COLLECTOR_PURPOSE,
                "config_sha256": config_hash,
                "config": {"sources": sources},
                "dataset": {"sha256": _sha256_file(dataset)},
            }
        ),
        encoding="utf-8",
    )
    return dataset, manifest, sources


def test_three_phase_schema_and_training_only_entrypoint():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset, manifest, expected_sources = write_fixture(root)
        sources, source_data = load_three_phase_inputs(dataset, manifest)
        assert sources == expected_sources
        assert source_data["dataset_sha256"] == _sha256_file(dataset)
        args = argparse.Namespace(
            dataset=dataset, manifest=manifest, checkpoint_dir=root / "out",
            device="cpu", value_seed=2072, value_lr=1e-4,
            batch_size=128, max_epochs=100, patience=10,
        )
        with patch("scripts.stage5.stage5_alphago_value_train.train_value") as train:
            run_training(args)
        train.assert_called_once()
        call = train.call_args
        assert call.kwargs["source_policies"] == expected_sources
        assert call.args[4:] == (dataset, manifest)
        assert call.args[1]["value_training"]["selection"] == "lowest_validation_mse"


def test_git_provenance_without_git_records_unavailable_and_source_hash():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "runner.py"
        source.write_text("pass\n", encoding="utf-8")
        with patch(
            "scripts.stage5.stage5_alphago_value_train.subprocess.check_output",
            side_effect=FileNotFoundError("git"),
        ):
            provenance = _git_provenance(source)
        assert provenance["status"] == "unavailable"
        assert provenance["head"] is None
        assert provenance["source_sha256"][str(source.resolve())] == _sha256_file(source)


def test_explicit_test_evaluation_strict_loads_fixture_and_records_hashes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset, manifest, sources = write_fixture(root)
        checkpoint = root / "value.pt"
        output = root / "test_metrics.json"
        model = ValueNetwork(hidden_dim=2, conv2_dim=2, fc1_dim=4, fc2_dim=2)
        torch.save(
            {
                "format_version": 1, "stage": 5,
                "checkpoint_type": "value_only", "model_class": "ValueNetwork",
                "model_config": model.get_config(), "state_dict": model.state_dict(),
                "encoder_version": "hwc14-white-v1",
                "value_perspective": "side_to_move", "claim_draw": True,
                "source_policies": sources,
                "source_data": {"dataset_sha256": _sha256_file(dataset)},
            },
            checkpoint,
        )
        result = evaluate_test(
            argparse.Namespace(
                dataset=dataset, manifest=manifest, checkpoint=checkpoint,
                output=output, device="cpu", batch_size=2,
            )
        )
        assert output.is_file()
        assert result["checkpoint_sha256"] == _sha256_file(checkpoint)
        assert set(("all", "label_-1", "label_0", "label_+1")).issubset(
            result["metrics"]["value"]
        )
        assert "always_zero" in result["metrics"]
        assert "always_train_target_mean" in result["metrics"]


def test_schema_rejects_missing_dual_provenance():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset, manifest, _ = write_fixture(root)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        del payload["config"]["sources"]["rl"]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_three_phase_inputs(dataset, manifest)
        except ValueError as exc:
            assert "RL provenance" in str(exc)
        else:
            raise AssertionError("missing RL provenance was accepted")


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
