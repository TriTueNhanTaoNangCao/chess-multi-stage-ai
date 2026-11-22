"""Fresh Stage 5 Policy initialization with a frozen Value baseline."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import torch

from models.cnn_policy import CNNPolicyNetwork
from models.value_net import ValueNetwork
from tests.stage5.helpers import (
    Stage5TrainConfig,
    load_stage5_checkpoint,
    save_stage5_checkpoint,
    train,
    write_small_stage4_checkpoint,
    write_value_only_checkpoint,
)
from training.stage4_train_sl import load_checkpoint as load_stage4_checkpoint
from training.stage5.checkpoint import load_stage5_policy_checkpoint


def _write_stage5_policy_source(path: Path) -> CNNPolicyNetwork:
    policy = CNNPolicyNetwork(hidden_dim=2, fc_dim=8, dropout=0.0)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.25)
    opponent = CNNPolicyNetwork.from_config(policy.get_config())
    opponent.load_state_dict(policy.state_dict(), strict=True)
    value = ValueNetwork(hidden_dim=4, conv2_dim=4, fc1_dim=8, fc2_dim=4)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    policy_optimizer.step()
    policy_optimizer.zero_grad(set_to_none=True)
    collector_rng = torch.Generator(device="cpu")
    collector_rng.manual_seed(19)
    save_stage5_checkpoint(
        path,
        policy_model=policy,
        value_model=value,
        opponent_model=opponent,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        config=Stage5TrainConfig(
            stage4_checkpoint="stage4-source.pt",
            output_dir=path.parent,
            iterations=3,
            policy_lr=3e-4,
            baseline="zero",
            value_mode="disabled",
        ),
        completed_iteration=3,
        history=[{"iteration": index, "games": 1} for index in range(1, 4)],
        source_checkpoint={"path": "stage4-source.pt", "stage": 4},
        collector_rng=collector_rng,
        value_rng=__import__("numpy").random.default_rng(19),
        claim_draw=True,
    )
    return policy


def test_fresh_policy_init_uses_new_state_and_preserves_sources():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stage4_path = root / "stage4.pt"
        policy_path = root / "stage5-policy.pt"
        value_path = root / "value.pt"
        output_dir = root / "fresh"
        write_small_stage4_checkpoint(stage4_path)
        source_policy = _write_stage5_policy_source(policy_path)
        source_value = write_value_only_checkpoint(value_path)
        source_payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        assert source_payload["policy_optimizer_state_dict"]["state"]

        result = train(
            Stage5TrainConfig(
                stage4_checkpoint=stage4_path,
                policy_init_checkpoint=policy_path,
                value_checkpoint=value_path,
                output_dir=output_dir,
                seed=2073,
                iterations=1,
                games_per_iteration=1,
                max_plies=0,
                policy_lr=1e-5,
                baseline="value",
                value_mode="frozen",
            )
        )

        for key, tensor in source_policy.state_dict().items():
            assert torch.equal(result.policy_model.state_dict()[key], tensor), key
        stage4_model, _ = load_stage4_checkpoint(stage4_path)
        for key, tensor in stage4_model.state_dict().items():
            assert torch.equal(result.opponent_model.state_dict()[key], tensor), key
        assert next(result.policy_model.parameters()).data_ptr() != next(
            result.opponent_model.parameters()
        ).data_ptr()
        assert not result.opponent_model.training
        assert not any(p.requires_grad for p in result.opponent_model.parameters())
        assert not result.value_model.training
        assert not any(p.requires_grad for p in result.value_model.parameters())
        for key, tensor in source_value.state_dict().items():
            assert torch.equal(result.value_model.state_dict()[key], tensor), key

        saved = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
        assert saved["policy_optimizer_state_dict"]["state"] == {}
        provenance = saved["source_checkpoint"]
        assert provenance["kind"] == "fresh_stage5_policy_init"
        assert provenance["policy_init"]["path"] == str(policy_path)
        assert len(provenance["policy_init"]["sha256"]) == 64
        assert provenance["policy_init"]["completed_iteration"] == 3
        assert provenance["stage4_opponent"]["path"] == str(stage4_path)
        restored = load_stage5_checkpoint(output_dir / "last.pt")
        assert restored.payload["source_checkpoint"] == provenance
        assert restored.payload["history"] != source_payload["history"]
        assert restored.payload["completed_iteration"] == 1


def test_policy_init_rejects_bad_metadata_and_resume_combination():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        policy_path = root / "stage5-policy.pt"
        _write_stage5_policy_source(policy_path)
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        payload["encoder_version"] = "wrong"
        bad_path = root / "bad.pt"
        torch.save(payload, bad_path)
        try:
            load_stage5_policy_checkpoint(bad_path)
        except ValueError as exc:
            assert "encoder_version" in str(exc)
        else:
            raise AssertionError("bad Policy init metadata was accepted")

        config = Stage5TrainConfig(
            policy_init_checkpoint=policy_path,
            resume_checkpoint=policy_path,
        )
        try:
            config.validate()
        except ValueError as exc:
            assert "cannot be used with resume_checkpoint" in str(exc)
        else:
            raise AssertionError("Policy init plus resume was accepted")
