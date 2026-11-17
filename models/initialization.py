"""Seed CPU model construction without reseeding other agents or training runs."""
from threading import Lock
from typing import Callable, TypeVar

import torch


_T = TypeVar("_T")
_INITIALIZATION_LOCK = Lock()


def initialize_model(factory: Callable[[], _T], seed: int | None) -> _T:
    # These factories create CPU weights; moving to CUDA happens afterwards.
    # Serialize our constructors because fork_rng temporarily swaps the CPU RNG.
    with _INITIALIZATION_LOCK:
        if seed is None:
            return factory()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return factory()
