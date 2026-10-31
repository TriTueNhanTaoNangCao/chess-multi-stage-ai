"""Registry and factory for chess agents."""
from .base import BaseAgent
from .human_agent import HumanAgent
from .random_agent import RandomAgent
from .stage1_heuristics import Think1Agent, Think2Agent, Think3Agent
from .stage2_tree_search import MinimaxAgent, AlphaBetaAgent
from .stage3_mcts_pure import PureMCTSAgent, MCTSNode
from .stage4_supervised.supervised_agent import SupervisedAgent
from .stage5_rl.rl_agent import RLPolicyAgent

# Agents that can be instantiated with default (depth, seed) without external checkpoints
CLASSIC_AGENTS = [
    "random",
    "think1",
    "think2",
    "think3",
    "minimax",
    "alphabeta",
    "mcts_pure",
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
}


def build_agent(key, depth=None, seed=None, *, checkpoint=None, device="cpu"):
    if key not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent key: {key}. Available: {list(AGENT_REGISTRY.keys())}")
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
    return cls(seed=seed)


__all__ = [
    "BaseAgent", "HumanAgent", "RandomAgent", "Think1Agent", "Think2Agent",
    "Think3Agent", "MinimaxAgent", "AlphaBetaAgent", "PureMCTSAgent", "MCTSNode",
    "SupervisedAgent", "RLPolicyAgent", "AGENT_REGISTRY", "build_agent",
    "CLASSIC_AGENTS", "DEFAULT_TOURNAMENT_AGENTS", "CHECKPOINT_REQUIRED_AGENTS",
]
