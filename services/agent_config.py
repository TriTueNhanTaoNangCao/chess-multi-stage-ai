"""Explicit, serializable bot configurations for matches and historical comparisons."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from agents import AGENT_REGISTRY, build_agent, PureMCTSAgent, AlphaGoAgent, AlphaZeroAgent


@dataclass(frozen=True)
class AgentConfig:
    key: str = "alphabeta"
    depth: int = 2
    simulations: int = 100
    seed: int = 42
    checkpoint: str = ""
    checkpoint_value: str = ""
    device: str = "cpu"

    @property
    def label(self):
        if self.key == "human":
            return "Người chơi"
        detail = f" · depth {self.depth}" if self.key in ("minimax", "alphabeta") else ""
        if self.key in ("mcts_pure", "alphago", "alphazero"):
            detail += f" · {self.simulations} sims"
        if self.checkpoint:
            detail += f" · {Path(self.checkpoint).name}"
        elif self.key in ("alphago", "alphazero"):
            detail += " · chưa huấn luyện"
        if self.key == "alphago" and self.checkpoint and not self.checkpoint_value:
            detail += " · value chưa huấn luyện"
        return f"{self.key}{detail}"

    def metadata(self):
        data = asdict(self)
        # Record hashes at game creation, never on every UI refresh.
        for field in ("checkpoint", "checkpoint_value"):
            if data[field]:
                path = Path(data[field]).expanduser().resolve()
                data[field] = str(path)
                digest = sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                data[field + "_sha256"] = digest.hexdigest()
        # Irrelevant form defaults must not split identical agents into variants.
        identity = {"key": self.key}
        if self.key in ("minimax", "alphabeta"):
            identity["depth"] = self.depth
            identity.pop("seed", None)
        if self.key in ("mcts_pure", "alphago", "alphazero"):
            identity["simulations"] = self.simulations
        for field in ("checkpoint_sha256", "checkpoint_value_sha256"):
            if field in data:
                identity[field] = data[field]
        data["id"] = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        data["label"] = self.label
        return data

    def build(self):
        if self.key == "human":
            return None
        if self.key not in AGENT_REGISTRY:
            raise ValueError(f"Bot không tồn tại: {self.key}")
        if self.depth < 1 or self.simulations < 1:
            raise ValueError("Depth và simulations phải lớn hơn 0")
        for path in (self.checkpoint, self.checkpoint_value):
            if path and not Path(path).is_file():
                raise ValueError(f"Không tìm thấy checkpoint: {path}")
        if self.key == "mcts_pure":
            return PureMCTSAgent(simulations=self.simulations, seed=self.seed)
        if self.key in ("alphago", "alphazero"):
            kwargs: dict[str, Any] = dict(simulations=self.simulations, seed=self.seed, device=self.device)
            # Explicit checkpoints must never silently become randomly initialized models.
            if self.checkpoint or self.checkpoint_value:
                import torch
                if self.key == "alphago":
                    for path, expected in ((self.checkpoint, "CNNPolicyNetwork"),
                                           (self.checkpoint_value, "ValueNetwork")):
                        if path:
                            payload = torch.load(path, map_location="cpu", weights_only=False)
                            if not isinstance(payload, dict) or payload.get("model_class") != expected:
                                raise ValueError(f"Checkpoint phải có model_class={expected}")
                    kwargs.update(checkpoint_policy=self.checkpoint or None,
                                  checkpoint_value=self.checkpoint_value or None)
                else:
                    from models.dual_head_resnet import DualHeadResNet
                    payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
                    if not isinstance(payload, dict) or "state_dict" not in payload:
                        raise ValueError("Checkpoint AlphaZero phải chứa state_dict và model_config")
                    network = DualHeadResNet.from_config(payload.get("model_config", {}))
                    network.load_state_dict(payload["state_dict"], strict=True)
                    network.to(self.device).eval()
                    kwargs["network"] = network
            return AlphaGoAgent(**kwargs) if self.key == "alphago" else AlphaZeroAgent(**kwargs)
        return build_agent(self.key, depth=self.depth, seed=self.seed,
                           checkpoint=self.checkpoint or None, device=self.device)
