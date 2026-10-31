"""Stage 5 Policy and Value update operations."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Dict, Optional, Sequence
import numpy as np
import torch
from torch import nn
from games.chess.encoder import NUM_ACTIONS, NUM_CHANNELS
from training.stage5._utils import _model_in_train, _models_in_eval
from training.stage5.types import GameTrajectory, ValueBatch

def build_value_batch(
    trajectories: Sequence[GameTrajectory],
    *,
    seed: int = 42,
    rng: Optional[np.random.Generator] = None,
) -> ValueBatch:
    """Select at most one learner decision from each terminal game."""
    if rng is None:
        rng = np.random.default_rng(seed)
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    states = []
    targets = []
    skipped = 0
    truncated_count = 0

    for trajectory in trajectories:
        if trajectory.truncated:
            truncated_count += 1
            skipped += 1
            continue
        eligible_steps = [step for step in trajectory.steps if step.target is not None]
        if not eligible_steps:
            skipped += 1
            continue

        step = eligible_steps[int(rng.integers(len(eligible_steps)))]
        state = np.asarray(step.state)
        if state.shape != (NUM_CHANNELS, 8, 8):
            raise ValueError(
                "Trajectory state must have shape "
                f"({NUM_CHANNELS}, 8, 8), got {state.shape}"
            )
        if state.dtype != np.float32:
            raise ValueError("Trajectory state must have dtype float32")
        target = float(step.target)
        if not np.isfinite(target) or target not in (-1.0, 0.0, 1.0):
            raise ValueError("Value target must be one of -1, 0, or 1")
        states.append(state)
        targets.append(target)

    if states:
        states_tensor = torch.from_numpy(np.stack(states)).to(dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    else:
        states_tensor = torch.empty(
            (0, NUM_CHANNELS, 8, 8), dtype=torch.float32
        )
        targets_tensor = torch.empty((0, 1), dtype=torch.float32)
    return ValueBatch(
        states=states_tensor,
        targets=targets_tensor,
        games_used=len(states),
        games_skipped=skipped,
        games_truncated=truncated_count,
    )

def update_value_network(
    value_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: ValueBatch,
    *,
    device: str | torch.device = "cpu",
) -> Dict[str, object]:
    """Apply one mean-squared-error update to a prepared value batch."""
    if not isinstance(value_model, nn.Module):
        raise TypeError("value_model must be a torch.nn.Module instance")
    if not isinstance(batch, ValueBatch):
        raise TypeError("batch must be a ValueBatch")
    if not isinstance(batch.states, torch.Tensor) or not isinstance(
        batch.targets, torch.Tensor
    ):
        raise TypeError("ValueBatch states and targets must be torch tensors")
    if batch.states.shape[1:] != (NUM_CHANNELS, 8, 8):
        raise ValueError(
            f"Value states must have shape (B, {NUM_CHANNELS}, 8, 8)"
        )
    if batch.states.dtype != torch.float32:
        raise ValueError("Value states must have dtype float32")
    if batch.targets.shape != (batch.states.shape[0], 1):
        raise ValueError("Value targets must have shape (B, 1)")
    if batch.targets.dtype != torch.float32:
        raise ValueError("Value targets must have dtype float32")
    if not torch.isfinite(batch.targets).all() or not torch.all(
        (batch.targets == -1) | (batch.targets == 0) | (batch.targets == 1)
    ):
        raise ValueError("Value targets must contain only -1, 0, or 1")

    sample_count = int(batch.states.shape[0])
    metrics: Dict[str, object] = {
        "value_loss": None,
        "samples_used": sample_count,
    }
    if sample_count == 0:
        return metrics

    states = batch.states.to(device=torch.device(device))
    targets = batch.targets.to(device=torch.device(device))
    optimizer.zero_grad(set_to_none=True)
    try:
        with _model_in_train(value_model):
            predictions = value_model(states)
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Value model must return shape {tuple(targets.shape)}, "
                    f"got {tuple(predictions.shape)}"
                )
            loss = torch.mean((predictions - targets) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite value loss; optimizer step aborted")
            if not loss.requires_grad:
                raise ValueError("Value loss has no gradient; optimizer step aborted")
            loss.backward()

        gradients = [
            parameter.grad
            for parameter in value_model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise ValueError("Value update produced no gradients; optimizer step aborted")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("Non-finite value gradient; optimizer step aborted")
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise

    metrics["value_loss"] = float(loss.detach().cpu())
    return metrics

def reinforce_policy_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[GameTrajectory],
    *,
    device: str | torch.device = "cpu",
    chunk_size: Optional[int] = None,
    value_model: Optional[nn.Module] = None,
) -> Dict[str, object]:
    """Apply one REINFORCE update, optionally using a fixed value baseline."""
    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be a torch.nn.Module instance")
    if value_model is not None and not isinstance(value_model, nn.Module):
        raise TypeError("value_model must be a torch.nn.Module instance or None")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")
    if value_model is not None:
        value_parameter_ids = {id(parameter) for parameter in value_model.parameters()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if value_parameter_ids & optimizer_parameter_ids:
            raise ValueError("Policy optimizer must not contain value_model parameters")

    usable = []
    skipped = 0
    truncated_count = 0
    for trajectory in trajectories:
        if trajectory.truncated:
            truncated_count += 1
        if (
            trajectory.truncated
            or not trajectory.steps
            or any(step.target is None for step in trajectory.steps)
        ):
            skipped += 1
        else:
            usable.append(trajectory)

    metrics: Dict[str, object] = {
        "policy_loss": None,
        "games_used": len(usable),
        "games_skipped": skipped,
        "games_truncated": truncated_count,
        "decisions": 0,
    }
    if not usable:
        return metrics

    device_obj = torch.device(device)
    games_per_chunk = chunk_size or len(usable)
    optimizer.zero_grad(set_to_none=True)
    total_loss_value = 0.0
    decision_count = 0

    try:
        value_context = (
            _models_in_eval(value_model) if value_model is not None else nullcontext()
        )
        with _models_in_eval(policy), value_context:
            for start in range(0, len(usable), games_per_chunk):
                game_losses = []
                for trajectory in usable[start : start + games_per_chunk]:
                    states = []
                    masks = []
                    actions = []
                    targets = []
                    for step in trajectory.steps:
                        state = np.asarray(step.state)
                        mask = np.asarray(step.legal_mask)
                        if state.shape != (NUM_CHANNELS, 8, 8):
                            raise ValueError(
                                "Trajectory state must have shape "
                                f"({NUM_CHANNELS}, 8, 8), got {state.shape}"
                            )
                        if state.dtype != np.float32:
                            raise ValueError("Trajectory state must have dtype float32")
                        if mask.shape != (NUM_ACTIONS,) or mask.dtype != np.bool_:
                            raise ValueError(
                                f"Legal mask must be bool with shape ({NUM_ACTIONS},)"
                            )
                        if not isinstance(step.action, (int, np.integer)):
                            raise ValueError("Trajectory action must be an integer")
                        action = int(step.action)
                        if action < 0 or action >= NUM_ACTIONS or not mask[action]:
                            raise ValueError(
                                "Trajectory action must be present in its legal mask"
                            )
                        target = float(step.target)
                        if not np.isfinite(target):
                            raise ValueError("Trajectory target must be finite")
                        states.append(state)
                        masks.append(mask)
                        actions.append(action)
                        targets.append(target)

                    states_tensor = torch.from_numpy(np.stack(states)).to(device_obj)
                    masks_tensor = torch.from_numpy(np.stack(masks)).to(device_obj)
                    actions_tensor = torch.tensor(
                        actions, dtype=torch.long, device=device_obj
                    )
                    targets_tensor = torch.tensor(
                        targets, dtype=torch.float32, device=device_obj
                    )
                    logits = policy(states_tensor)
                    expected_shape = (len(states), NUM_ACTIONS)
                    if logits.shape != expected_shape:
                        raise ValueError(
                            f"Policy must return logits with shape {expected_shape}, "
                            f"got {tuple(logits.shape)}"
                        )
                    if value_model is None:
                        advantages = targets_tensor
                    else:
                        with torch.no_grad():
                            baselines = value_model(states_tensor)
                        expected_value_shape = (len(states), 1)
                        if baselines.shape != expected_value_shape:
                            raise ValueError(
                                "Value baseline must return shape "
                                f"{expected_value_shape}, got {tuple(baselines.shape)}"
                            )
                        if not torch.isfinite(baselines).all():
                            raise ValueError(
                                "Non-finite value baseline; optimizer step aborted"
                            )
                        advantages = targets_tensor - baselines.squeeze(1)
                        if not torch.isfinite(advantages).all():
                            raise ValueError(
                                "Non-finite advantage; optimizer step aborted"
                            )
                    masked_logits = logits.masked_fill(~masks_tensor, -torch.inf)
                    log_probs = torch.log_softmax(masked_logits, dim=1)
                    selected_log_probs = log_probs.gather(
                        1, actions_tensor.unsqueeze(1)
                    ).squeeze(1)
                    game_loss = -(selected_log_probs * advantages).sum()
                    if not torch.isfinite(game_loss):
                        raise ValueError("Non-finite policy loss; optimizer step aborted")
                    game_losses.append(game_loss)
                    decision_count += len(states)

                chunk_loss = torch.stack(game_losses).sum() / len(usable)
                if not chunk_loss.requires_grad:
                    raise ValueError("Policy loss has no gradient; optimizer step aborted")
                chunk_loss.backward()
                total_loss_value += sum(
                    float(game_loss.detach().cpu()) for game_loss in game_losses
                )

        gradients = [
            parameter.grad
            for parameter in policy.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise ValueError("Policy update produced no gradients; optimizer step aborted")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("Non-finite policy gradient; optimizer step aborted")
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise

    metrics["policy_loss"] = total_loss_value / len(usable)
    metrics["decisions"] = decision_count
    return metrics
