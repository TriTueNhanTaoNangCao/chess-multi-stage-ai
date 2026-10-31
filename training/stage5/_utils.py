"""Shared model-mode context managers for Stage 5."""
from contextlib import contextmanager
from typing import Iterator
from torch import nn

@contextmanager
def _models_in_eval(*models: nn.Module) -> Iterator[None]:
    """Temporarily use eval mode while preserving every submodule mode flag."""
    modules = []
    seen = set()
    for model in models:
        if not isinstance(model, nn.Module):
            raise TypeError("policies must be torch.nn.Module instances")
        for module in model.modules():
            if id(module) not in seen:
                seen.add(id(module))
                modules.append((module, module.training))
        model.eval()
    try:
        yield
    finally:
        for module, was_training in modules:
            module.training = was_training

@contextmanager
def _model_in_train(model: nn.Module) -> Iterator[None]:
    """Temporarily enable training while preserving every mode flag."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module instance")
    modes = [(module, module.training) for module in model.modules()]
    model.train()
    try:
        yield
    finally:
        for module, was_training in modes:
            module.training = was_training
