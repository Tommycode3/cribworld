import numpy as np
import torch

from cribworld.agents import (AgentConfig, RNDAgent, expanded_bounds, icm_reward,
                              module_probabilities, progress)
from cribworld.metrics import coverage_curve, discovered_curve, first_discovery


def test_icm_reward_is_half_squared_error():
    pred = np.array([[1.0, 2.0], [0.0, 0.0]])
    true = np.array([[1.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(icm_reward(pred, true), [2.0, 12.5])


def test_rnd_reward_zero_when_predictor_copies_target():
    agent = RNDAgent("state", 10, seed=0, cfg=AgentConfig())
    agent.pred.load_state_dict(agent.target.state_dict())
    b = torch.randn(5, 8)
    x = torch.randn(5, 8)
    from cribworld.agents import rnd_reward
    with torch.no_grad():
        r = rnd_reward(agent.pred(b, x).numpy(), agent.target(b, x).numpy())
    np.testing.assert_allclose(r, 0.0, atol=1e-12)


def test_progress_and_module_probabilities():
    assert progress(0.5, 0.2) == 0.3 or np.isclose(progress(0.5, 0.2), 0.3)
    assert np.isclose(progress(0.2, 0.5), 0.3)  # absolute value: regress counts too
    p = module_probabilities(np.array([0.0, 1.0, 3.0]), eps=0.2)
    np.testing.assert_allclose(p.sum(), 1.0)
    np.testing.assert_allclose(p, [0.2 / 3, 0.2 / 3 + 0.8 * 0.25, 0.2 / 3 + 0.8 * 0.75])
    np.testing.assert_allclose(module_probabilities(np.zeros(4), 0.2), 0.25)


def test_expanded_bounds():
    lo, hi = expanded_bounds(np.array([[0.0, 1.0], [1.0, 1.0]]), margin=0.1)
    np.testing.assert_allclose(lo, [-0.1, 1.0 - 1e-4])
    np.testing.assert_allclose(hi, [1.1, 1.0 + 1e-4])


def test_coverage_and_discovery():
    xy = np.array([[0.0, 0.0], [0.001, 0.001], [0.1, 0.1], [0.0, 0.0]])
    np.testing.assert_array_equal(coverage_curve(xy, [1, 2, 3, 4]), [1, 1, 2, 2])
    ev = np.array([[0, 0], [1, 0], [0, 0], [1, 1]], bool)
    np.testing.assert_array_equal(first_discovery(ev), [1, 3])
    np.testing.assert_array_equal(discovered_curve(ev, [1, 2, 4]), [0, 1, 2])
