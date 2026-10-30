import numpy as np

from cribworld.env import EVENT_NAMES, OUTCOME_DIM, Crib, EnvConfig


def test_same_seed_same_theta_same_outcome():
    theta = np.array([0.3, -0.2, 0.8, 0.1, -0.5, 0.6])
    a = Crib(seed=7).run(theta)
    b = Crib(seed=7).run(theta)
    np.testing.assert_array_equal(a.outcome, b.outcome)
    np.testing.assert_array_equal(a.context, b.context)
    np.testing.assert_array_equal(a.events, b.events)


def test_reset_restores_scene():
    env = Crib(EnvConfig(context_noise=0.0), seed=0)
    theta = np.array([0.9, -0.9, 0.4, 0.4, -0.2, 0.7])
    first = env.run(theta)
    env.run(-theta)  # disturb the scene
    again = env.run(theta)
    np.testing.assert_allclose(first.outcome, again.outcome, atol=1e-9)


def test_shapes_and_bounds():
    env = Crib(seed=1)
    ep = env.run(np.zeros(6))
    assert ep.outcome.shape == (OUTCOME_DIM,)
    assert ep.events.shape == (len(EVENT_NAMES),)
    assert ep.obs1["body"].shape == (8,)
    assert ep.obs1["state"].shape == (8,)
    # every object stays inside the walls
    xy = ep.outcome[2:8].reshape(3, 2)
    assert np.all(np.abs(xy[:, 0]) < 0.5) and np.all((xy[:, 1] > -0.14) & (xy[:, 1] < 0.6))


def test_motionless_arm_moves_nothing():
    env = Crib(EnvConfig(context_noise=0.0), seed=0)
    # home pose expressed as theta: the arm never leaves its folded start
    from cribworld.env import HOME_Q
    lo, hi = env._jnt_lo, env._jnt_hi
    th = np.tile(2 * (HOME_Q - lo) / (hi - lo) - 1, 3)
    ep = env.run(th)
    assert not ep.events[EVENT_NAMES.index("move_light_box")]
    assert not ep.events[EVENT_NAMES.index("move_ball")]
    assert not ep.events[EVENT_NAMES.index("toy_light_on")]
