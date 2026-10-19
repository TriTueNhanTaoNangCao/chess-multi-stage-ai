"""Only completed agents are registered for GUI, CLI and arena."""
from .base import BaseAgent
from .human_agent import HumanAgent
from .random_agent import RandomAgent
from .stage1_heuristics import Think1Agent, Think2Agent, Think3Agent
from .stage2_tree_search import MinimaxAgent, AlphaBetaAgent
from .stage3_mcts_pure import PureMCTSAgent, MCTSNode

AGENT_REGISTRY = {
    "random": RandomAgent, "think1": Think1Agent, "think2": Think2Agent,
    "think3": Think3Agent, "minimax": MinimaxAgent, "alphabeta": AlphaBetaAgent,
    "mcts_pure": PureMCTSAgent,
}

def build_agent(key, depth=None, seed=None):
    cls = AGENT_REGISTRY[key]
    if issubclass(cls, MinimaxAgent):
        return cls(**({"depth": depth} if depth is not None else {}))
    if issubclass(cls, PureMCTSAgent):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        return cls(**kwargs)
    return cls(seed=seed)

__all__ = ["BaseAgent", "HumanAgent", "RandomAgent", "Think1Agent", "Think2Agent",
           "Think3Agent", "MinimaxAgent", "AlphaBetaAgent", "PureMCTSAgent", "MCTSNode",
           "AGENT_REGISTRY", "build_agent"]
