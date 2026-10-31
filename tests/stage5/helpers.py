"""Shared Stage 5 test doubles and fixtures."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import chess
import numpy as np
import torch
from torch import nn

from agents import (
    AGENT_REGISTRY,
    CLASSIC_AGENTS,
    DEFAULT_TOURNAMENT_AGENTS,
    build_agent,
)
from agents.stage5_rl.rl_agent import RLPolicyAgent
from arena.tournament import build_cli_agents
from games.chess.board import ChessBoard
from games.chess.encoder import ChessEncoder, NUM_ACTIONS
from training.stage5_train_rl import (
    GameStart,
    GameTrajectory,
    OpponentSnapshot,
    Stage5TrainConfig,
    TrajectoryStep,
    ValueBatch,
    build_value_batch,
    build_mixed_start_batch,
    collect_trajectory,
    collect_trajectory_batch,
    load_stage5_checkpoint,
    load_value_only_checkpoint,
    load_mixed_start_manifest,
    reinforce_policy_update,
    save_stage5_checkpoint,
    train,
    update_value_network,
    _collect_trajectory_batch_with_opponent_pool,
    _collect_trajectory,
    _collect_trajectory_batch_from_starts_with_rng,
    _snapshot_policy,
)


class TinyPolicy(nn.Module):
    """Small policy with parameters and buffers but no global-RNG use."""

    def __init__(self, actions=None):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bn = nn.BatchNorm1d(1)
        self.register_buffer("marker", torch.tensor([3.0]))
        self.actions = list(actions or [])
        self.calls = 0

    def forward(self, state):
        logits = torch.zeros(
            (state.shape[0], NUM_ACTIONS), device=state.device, dtype=torch.float32
        )
        logits = logits + self.scale * 0.0
        if self.actions:
            action = self.actions[self.calls]
            self.calls += 1
            logits[:, action] = 100.0
        return logits


class LogitPolicy(nn.Module):
    """Trainable action biases plus BatchNorm state for update tests."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(NUM_ACTIONS))
        self.bn = nn.BatchNorm2d(14)

    def forward(self, state):
        features = self.bn(state)
        zero = features.mean(dim=(1, 2, 3)) * 0.0
        return self.logits.unsqueeze(0).expand(state.shape[0], -1) + zero.unsqueeze(1)


class ScalarValueModel(nn.Module):
    """Minimal scalar regressor used to verify MSE independently."""

    def __init__(self, initial_value=0.0):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(initial_value)))
        self.bn = nn.BatchNorm2d(14)

    def forward(self, state):
        features = self.bn(state)
        zero = features.mean(dim=(1, 2, 3), keepdim=False) * 0.0
        return (self.value + zero).reshape(-1, 1)


class NonFiniteValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, state):
        return torch.full(
            (state.shape[0], 1), float("nan"), device=state.device
        ) + self.anchor * 0.0


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def make_step(action=0, legal_actions=(0, 1), target=1.0, state_value=0.0):
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[list(legal_actions)] = True
    return TrajectoryStep(
        state=np.full((14, 8, 8), state_value, dtype=np.float32),
        action=action,
        legal_mask=mask,
        current_player=1,
        ply_index=0,
        target=target,
    )


def make_trajectory(steps, *, truncated=False):
    return GameTrajectory(
        steps=list(steps),
        learner_color=1,
        truncated=truncated,
        termination_reason="max_plies" if truncated else "checkmate",
        winner_white=None if truncated else 1,
        plies_played=len(steps),
    )


def write_small_stage4_checkpoint(path):
    from models.cnn_policy import CNNPolicyNetwork

    model = CNNPolicyNetwork(hidden_dim=2, fc_dim=8, dropout=0.0)
    torch.save(
        {
            "format_version": 1,
            "stage": 4,
            "model_class": "CNNPolicyNetwork",
            "model_config": model.get_config(),
            "encoder_version": "hwc14-white-v1",
            "action_version": "az73-rankflip-v1",
            "num_actions": NUM_ACTIONS,
            "value_perspective": "side_to_move",
            "claim_draw": True,
            "seed": 42,
            "epoch": 0,
            "step": 0,
            "training_config": {},
            "metrics": {},
            "state_dict": model.state_dict(),
        },
        path,
    )


def write_value_only_checkpoint(path, *, claim_draw=True, overrides=None):
    """Write a small standalone Value checkpoint matching the real schema."""
    from models.value_net import ValueNetwork

    model = ValueNetwork(hidden_dim=4, conv2_dim=4, fc1_dim=8, fc2_dim=4)
    payload = {
        "format_version": 1,
        "stage": 5,
        "checkpoint_type": "value_only",
        "model_class": "ValueNetwork",
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),
        "encoder_version": "hwc14-white-v1",
        "value_perspective": "side_to_move",
        "claim_draw": claim_draw,
        "source_policy": {"sha256": "1" * 64},
        "source_data": {"dataset_sha256": "2" * 64},
        "seed": 7,
        "epoch": 3,
    }
    if overrides:
        payload.update(overrides)
    torch.save(payload, path)
    return model


def write_mixed_start_manifest(path):
    """Write a minimal 12-PGN/4-mate manifest for isolated tests."""
    root = chess.STARTING_FEN
    pgn_entries = []
    first_moves = list(chess.Board().legal_moves)[:11]
    mate_fen = (
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/"
        "PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    )
    pgn_entries.append(
        {
            "type": "pgn",
            "root_fen": mate_fen,
            "prefix_uci": [],
            "current_fen": mate_fen,
            "game_result": "0-1",
            "law_state_sha256": "pgn-mate-source",
        }
    )
    for index, move in enumerate(first_moves, 1):
        board = chess.Board(root)
        board.push(move)
        pgn_entries.append(
            {
                "type": "pgn",
                "root_fen": root,
                "prefix_uci": [move.uci()],
                "current_fen": board.fen(en_passant="fen"),
                "game_result": "*",
                "law_state_sha256": f"pgn-{index}",
            }
        )
    mate_fens = [
        "8/8/8/8/8/8/8/k1KQ4 w - - 0 1",
        "8/8/8/8/8/8/8/k1K1Q3 w - - 0 1",
        "8/8/8/8/8/8/8/K1kq4 b - - 0 1",
        "8/8/8/8/8/8/8/K1k1q3 b - - 0 1",
    ]
    mate_entries = [
        {
            "type": "mate_in_one",
            "root_fen": fen,
            "prefix_uci": [],
            "current_fen": fen,
            "law_state_sha256": f"mate-{index}",
        }
        for index, fen in enumerate(mate_fens)
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "stage5_mixed_start_test",
                "positions": pgn_entries + mate_entries,
            }
        ),
        encoding="utf-8",
    )


def scripted_indices(moves):
    board = chess.Board()
    white, black = [], []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        target = white if board.turn == chess.WHITE else black
        target.append(ChessEncoder.encode_move(move, board.turn))
        board.push(move)
    return white, black
