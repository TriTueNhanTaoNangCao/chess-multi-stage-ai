"""Registry and factory for chess agents."""
from pathlib import Path
from .base import BaseAgent
from .human_agent import HumanAgent
from .random_agent import RandomAgent
from .stage1_heuristics import Think1Agent, Think2Agent, Think3Agent
from .stage2_tree_search import MinimaxAgent, AlphaBetaAgent
from .stage3_mcts_pure import PureMCTSAgent, MCTSNode
from .stage4_supervised.supervised_agent import SupervisedAgent
from .stage5_rl.rl_agent import RLPolicyAgent
from .stage6_alphago import AlphaGoAgent, AlphaGoNode, AlphaGoMCTS
from .stage7_alphazero import AlphaZeroAgent, AlphaZeroNode, AlphaZeroMCTS

# Agents that can be instantiated with default (depth, seed) without external checkpoints
CLASSIC_AGENTS = [
    "random",
    "think1",
    "think2",
    "think3",
    "minimax",
    "alphabeta",
    "mcts_pure",
    "alphago",
    "alphazero",
]

DEFAULT_TOURNAMENT_AGENTS = list(CLASSIC_AGENTS)
CHECKPOINT_REQUIRED_AGENTS = {"supervised", "rl_policy"}

AGENT_REGISTRY = {
    "random": RandomAgent,
    "think1": Think1Agent,
    "think2": Think2Agent,
    "think3": Think3Agent,
    "minimax": MinimaxAgent,
    "alphabeta": AlphaBetaAgent,
    "mcts_pure": PureMCTSAgent,
    "supervised": SupervisedAgent,
    "rl_policy": RLPolicyAgent,
    "alphago": AlphaGoAgent,
    "alphazero": AlphaZeroAgent,
}


def build_agent(key, depth=None, seed=None, *, checkpoint=None, device="cpu", move_time_limit_seconds=None):
    if key not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent key: {key}. Available: {list(AGENT_REGISTRY.keys())}")
    if move_time_limit_seconds is not None:
        from services.search_worker import TimedAgent
        agent = TimedAgent(("factory", dict(key=key, depth=depth, seed=seed, checkpoint=checkpoint, device=device)),
                           move_time_limit_seconds, name=f"{key}" + (f":{depth}" if key in ("minimax", "alphabeta") else "") + f" · ≤{move_time_limit_seconds:g}s")
        try:
            agent.prepare()
        except BaseException:
            agent.close()
            raise
        return agent
    cls = AGENT_REGISTRY[key]
    if key in CHECKPOINT_REQUIRED_AGENTS:
        if not checkpoint:
            raise ValueError(
                f"{key} agent requires a valid checkpoint path (use --checkpoint)."
            )
        return cls(checkpoint=checkpoint, device=device, seed=seed)
    if issubclass(cls, MinimaxAgent):
        return cls(**({"depth": depth} if depth is not None else {}))
    if issubclass(cls, PureMCTSAgent):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        return cls(**kwargs)
    if issubclass(cls, AlphaGoAgent):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        else:
            default_p = Path("checkpoints/stage6/policy.pt")
            default_v = Path("checkpoints/stage6/value.pt")
            default_best = Path("checkpoints/stage6/best.pt")
            if default_p.is_file():
                kwargs["checkpoint_policy"] = str(default_p)
            if default_v.is_file():
                kwargs["checkpoint_value"] = str(default_v)
            elif default_best.is_file():
                kwargs["checkpoint"] = str(default_best)
        if device is not None:
            kwargs["device"] = device
        if depth is not None:
            kwargs["simulations"] = max(10, depth * 15)
        return cls(**kwargs)
    if issubclass(cls, AlphaZeroAgent):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if checkpoint is not None:
            kwargs["checkpoint"] = checkpoint
        else:
            default_ckpt = Path("checkpoints/stage7/best.pt")
            if default_ckpt.is_file():
                kwargs["checkpoint"] = str(default_ckpt)
        if device is not None:
            kwargs["device"] = device
        if depth is not None:
            kwargs["simulations"] = max(10, depth * 15)
        return cls(**kwargs)
    return cls(seed=seed)


__all__ = [
    "BaseAgent", "HumanAgent", "RandomAgent", "Think1Agent", "Think2Agent",
    "Think3Agent", "MinimaxAgent", "AlphaBetaAgent", "PureMCTSAgent", "MCTSNode",
    "SupervisedAgent", "RLPolicyAgent", "AlphaGoAgent", "AlphaGoNode", "AlphaGoMCTS",
    "AlphaZeroAgent", "AlphaZeroNode", "AlphaZeroMCTS",
    "AGENT_REGISTRY", "build_agent", "CLASSIC_AGENTS", "DEFAULT_TOURNAMENT_AGENTS",
    "CHECKPOINT_REQUIRED_AGENTS",
]
