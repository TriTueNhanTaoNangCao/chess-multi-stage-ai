"""
Stage 7: AlphaZero Self-Play and Training Pipeline.
Author: Phan Thanh Tung (Tung).

Architectural Reference:
- Silver et al., "A general reinforcement learning algorithm that masters chess,
  shogi, and Go through self-play", Science 362, 1140-1144 (2018).
- Mark Liu, "AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games", CRC Press.

Closed-Loop AlphaZero Pipeline:
1. Self-Play: DualHeadResNet + AlphaZeroMCTS generate game trajectories (s_t, pi_t, z_t).
2. Replay Buffer: Collects state tensors, MCTS visit distributions pi, and game outcomes z.
3. Loss Optimization: Joint loss L = (z - v)^2 - pi^T log(p) + c ||theta||^2.
4. Checkpointing: Saves model state and metadata according to Appendix B contract.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import chess
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from models.dual_head_resnet import DualHeadResNet
except ImportError:
    torch = None
    F = None
    DualHeadResNet = None

from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS, NUM_CHANNELS
from agents.stage7_alphazero.mcts import AlphaZeroMCTS


@dataclass
class AlphaZeroTrainConfig:
    """Configuration parameters for AlphaZero self-play and training."""
    num_iterations: int = 1
    games_per_iteration: int = 2
    mcts_simulations: int = 15
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temp_threshold_plies: int = 10
    max_plies_per_game: int = 80
    epochs_per_iteration: int = 1
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_dir: str = "checkpoints/stage7"
    device: str = "cpu"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlphaZeroTrainConfig":
        valid_keys = {k for k in cls.__dataclass_fields__.keys()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SelfPlaySample:
    """Single training sample from an AlphaZero self-play game."""
    state: np.ndarray  # Shape (14, 8, 8) float32
    pi: np.ndarray     # Shape (4672,) float32
    value: float       # Scalar outcome in [-1, 1] relative to current player


class AlphaZeroReplayBuffer:
    """Fixed-capacity experience replay buffer for self-play data."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.samples: List[SelfPlaySample] = []

    def push(self, sample: SelfPlaySample) -> None:
        if len(self.samples) >= self.capacity:
            self.samples.pop(0)
        self.samples.append(sample)

    def extend(self, samples: List[SelfPlaySample]) -> None:
        for s in samples:
            self.push(s)

    def sample_batch(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.samples:
            raise ValueError("Cannot sample from an empty replay buffer.")
        k = min(batch_size, len(self.samples))
        batch = random.sample(self.samples, k)

        states = torch.from_numpy(np.stack([s.state for s in batch])).to(device)
        pis = torch.from_numpy(np.stack([s.pi for s in batch])).to(device)
        values = torch.tensor([[s.value] for s in batch], dtype=torch.float32).to(device)

        return states, pis, values

    def __len__(self) -> int:
        return len(self.samples)


def play_self_play_game(
    network: Any,
    config: AlphaZeroTrainConfig,
    seed: Optional[int] = None,
) -> List[SelfPlaySample]:
    """
    Simulate a complete self-play game between AlphaZero and itself.

    Returns:
        List of SelfPlaySample tuples (s_t, pi_t, z_t) for each step of the game.
    """
    board = ChessBoard()
    mcts = AlphaZeroMCTS(
        network=network,
        simulations=config.mcts_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        device=config.device,
        seed=seed,
    )

    trajectory: List[Tuple[np.ndarray, np.ndarray, chess.Color]] = []
    ply = 0

    while not board.is_game_over() and ply < config.max_plies_per_game:
        # Encode state from side to move: HWC -> CHW (14, 8, 8)
        hwc_state = ChessEncoder.encode_board(board.board)
        chw_state = np.transpose(hwc_state, (2, 0, 1)).astype(np.float32)

        # Temperature schedule: exploration (tau=1.0) early, greedy (tau=0.1) later
        temp = 1.0 if ply < config.temp_threshold_plies else 0.1

        # MCTS search with root Dirichlet noise
        pi_dict = mcts.get_action_probs(board, add_noise=True, temperature=temp)
        if not pi_dict:
            break

        # Dense 4672 probability target vector
        pi_vector = np.zeros(NUM_ACTIONS, dtype=np.float32)
        turn = board.turn
        for move, prob in pi_dict.items():
            action_idx = ChessEncoder.encode_move(move, turn)
            pi_vector[action_idx] = prob

        # Record step
        trajectory.append((chw_state, pi_vector, turn))

        # Sample action from pi
        moves = list(pi_dict.keys())
        probs = list(pi_dict.values())
        chosen_move = random.choices(moves, weights=probs, k=1)[0]

        board.push(chosen_move)
        ply += 1

    # Determine terminal outcome from White's perspective: +1 = White win, -1 = Black win, 0 = Draw
    res = board.outcome()
    if res is None or res.winner is None:
        white_outcome = 0.0
    else:
        white_outcome = 1.0 if res.winner == chess.WHITE else -1.0

    # Assign value targets from the perspective of each step's player
    samples: List[SelfPlaySample] = []
    for state, pi_vector, player_turn in trajectory:
        step_val = white_outcome if player_turn == chess.WHITE else -white_outcome
        samples.append(SelfPlaySample(state=state, pi=pi_vector, value=float(step_val)))

    return samples


def compute_alphazero_loss(
    policy_logits: torch.Tensor,
    value_preds: torch.Tensor,
    target_pis: torch.Tensor,
    target_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the combined AlphaZero loss:
      L = (z - v)^2 - pi^T * log_softmax(logits)
    """
    # Policy cross-entropy
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.sum(target_pis * log_probs, dim=-1).mean()

    # Value MSE
    value_loss = F.mse_loss(value_preds, target_values)

    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss


def train_step(
    model: DualHeadResNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: AlphaZeroReplayBuffer,
    config: AlphaZeroTrainConfig,
) -> Dict[str, float]:
    """Execute one training epoch over replay buffer minibatches."""
    model.train()
    total_loss_accum = 0.0
    policy_loss_accum = 0.0
    value_loss_accum = 0.0
    num_batches = max(1, len(replay_buffer) // config.batch_size)

    for _ in range(num_batches):
        states, pis, values = replay_buffer.sample_batch(config.batch_size, device=config.device)
        optimizer.zero_grad()

        policy_logits, value_preds = model(states)
        total_loss, pol_loss, val_loss = compute_alphazero_loss(
            policy_logits, value_preds, pis, values
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_accum += total_loss.item()
        policy_loss_accum += pol_loss.item()
        value_loss_accum += val_loss.item()

    return {
        "loss": total_loss_accum / num_batches,
        "policy_loss": policy_loss_accum / num_batches,
        "value_loss": value_loss_accum / num_batches,
    }


def save_checkpoint(
    model: DualHeadResNet,
    config: AlphaZeroTrainConfig,
    step: int,
    metrics: Dict[str, float],
    filename: str = "best.pt",
) -> Path:
    """Save checkpoint complying with Appendix B contract."""
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / filename

    payload = {
        "format_version": 1,
        "stage": 7,
        "model_class": "DualHeadResNet",
        "model_config": model.get_config(),
        "encoder_version": "hwc14-white-v1",
        "action_version": "az73-rankflip-v1",
        "num_actions": NUM_ACTIONS,
        "value_perspective": "side_to_move",
        "claim_draw": True,
        "seed": config.seed,
        "step": step,
        "training_config": config.to_dict(),
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, save_path)
    return save_path


def train(config_input: Optional[Union[Dict[str, Any], AlphaZeroTrainConfig]] = None) -> Dict[str, Any]:
    """
    Main training entry point for AlphaZero closed-loop self-play and optimization.
    """
    if torch is None or DualHeadResNet is None:
        raise RuntimeError("PyTorch is required for AlphaZero training.")

    if config_input is None:
        config = AlphaZeroTrainConfig()
    elif isinstance(config_input, dict):
        config = AlphaZeroTrainConfig.from_dict(config_input)
    else:
        config = config_input

    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = DualHeadResNet()
    model.to(config.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    replay_buffer = AlphaZeroReplayBuffer()
    all_metrics: List[Dict[str, float]] = []

    for it in range(1, config.num_iterations + 1):
        # 1. Self-Play data generation
        model.eval()
        for g in range(config.games_per_iteration):
            samples = play_self_play_game(model, config, seed=config.seed + it * 100 + g)
            replay_buffer.extend(samples)

        # 2. Optimization step
        metrics = train_step(model, optimizer, replay_buffer, config)
        metrics["iteration"] = it
        metrics["buffer_size"] = len(replay_buffer)
        all_metrics.append(metrics)

    # Save final best checkpoint
    last_metrics = all_metrics[-1] if all_metrics else {}
    ckpt_path = save_checkpoint(model, config, step=config.num_iterations, metrics=last_metrics)

    return {
        "status": "success",
        "checkpoint_path": str(ckpt_path),
        "total_samples": len(replay_buffer),
        "iterations": config.num_iterations,
        "final_metrics": last_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Training Pipeline")
    parser.add_argument("--iterations", type=int, default=1, help="Number of self-play iterations")
    parser.add_argument("--games", type=int, default=2, help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=15, help="MCTS simulations per move")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage7", help="Checkpoint directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = AlphaZeroTrainConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        mcts_simulations=args.simulations,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    result = train(cfg)
    print("Training finished successfully:", result)
