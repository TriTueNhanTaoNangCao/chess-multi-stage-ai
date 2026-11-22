"""Seeded fallback models must be reproducible without changing global RNGs."""
import numpy as np
import pytest
import torch

from agents import AlphaGoAgent, AlphaZeroAgent


@pytest.mark.parametrize("agent_class", [AlphaGoAgent, AlphaZeroAgent])
def test_seeded_fallback_models_are_reproducible_and_isolated(agent_class):
    torch_before = torch.random.get_rng_state()
    numpy_before = np.random.get_state()
    first = agent_class(seed=42, simulations=1)
    second = agent_class(seed=42, simulations=1)
    other = agent_class(seed=43, simulations=1)
    names = ["network"] if agent_class is AlphaZeroAgent else ["policy_net", "value_net"]
    for name in names:
        a, b, c = [getattr(agent, name).state_dict() for agent in (first, second, other)]
        assert all(torch.equal(a[key], b[key]) for key in a)
        assert any(not torch.equal(a[key], c[key]) for key in a)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
