"""Evaluation. Everything here uses simulator ground truth, never the agent's own beliefs."""
from __future__ import annotations

import numpy as np
import torch

from .env import EVENT_NAMES, OBJECTS, THETA_DIM, Crib, EnvConfig
from .nets import fit_regressor, mlp

TABLE_X = (-0.5, 0.5)
TABLE_Y = (-0.14, 0.6)
CELL = 0.02  # 2 cm grid cells

OBJ_SLICES = {"light_box": slice(2, 4), "ball": slice(4, 6), "heavy_box": slice(6, 8)}


def cells(xy: np.ndarray, cell: float = CELL) -> np.ndarray:
    ix = np.floor((xy[:, 0] - TABLE_X[0]) / cell).astype(np.int64)
    iy = np.floor((xy[:, 1] - TABLE_Y[0]) / cell).astype(np.int64)
    return ix * 10_000 + iy


def coverage_curve(xy: np.ndarray, checkpoints, cell: float = CELL) -> np.ndarray:
    """Number of distinct grid cells reached by the first t outcomes, for t in checkpoints."""
    c = cells(xy, cell)
    seen = set()
    out = []
    j = 0
    for t in checkpoints:
        while j < t:
            seen.add(int(c[j]))
            j += 1
        out.append(len(seen))
    return np.array(out)


def object_coverage(outcome: np.ndarray, checkpoints) -> dict:
    res = {o: coverage_curve(outcome[:, s], checkpoints) for o, s in OBJ_SLICES.items()}
    res["objects_total"] = sum(res[o] for o in OBJ_SLICES)
    res["hand"] = coverage_curve(outcome[:, 0:2], checkpoints)
    return res


def toy_coverage(outcome: np.ndarray, checkpoints, bins: int = 20) -> np.ndarray:
    """Distinct 0.07 rad bins of the maximal lever deflection."""
    b = np.minimum((outcome[:, 8] / 1.4 * bins).astype(int), bins - 1)
    seen, out, j = set(), [], 0
    for t in checkpoints:
        while j < t:
            seen.add(int(b[j]))
            j += 1
        out.append(len(seen))
    return np.array(out)


def first_discovery(events: np.ndarray) -> np.ndarray:
    """Episode index of the first occurrence of each interaction type (nan if never)."""
    out = np.full(events.shape[1], np.nan)
    for k in range(events.shape[1]):
        idx = np.flatnonzero(events[:, k])
        if len(idx):
            out[k] = idx[0]
    return out


def discovered_curve(events: np.ndarray, checkpoints, min_count: int = 1) -> np.ndarray:
    cs = np.cumsum(events, axis=0)
    return np.array([(cs[t - 1] >= min_count).sum() for t in checkpoints])


def focus(events: np.ndarray, window: int = 250) -> dict:
    """Fraction of episodes per window in which each object was touched / the toy lit."""
    names = {"light_box": "touch_light_box", "ball": "touch_ball", "heavy_box": "touch_heavy_box",
             "toy": "touch_toy", "toy_light": "toy_light_on"}
    n = len(events) // window
    out = {}
    for k, ev in names.items():
        col = events[: n * window, EVENT_NAMES.index(ev)].reshape(n, window)
        out[k] = col.mean(1)
    no_obj = ~events[:, [EVENT_NAMES.index(names[k]) for k in ("light_box", "ball", "heavy_box", "toy")]].any(1)
    out["body_only"] = no_obj[: n * window].reshape(n, window).mean(1)
    return out


# ------------------------------------------------- held-out forward model


def fm_inputs(context: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.concatenate([context, theta], axis=1).astype(np.float32)


def fm_targets(context: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Object displacements (6) in cm and lever deflection (rad)."""
    disp = outcome[:, 2:8] - context
    return np.concatenate([disp * 100.0, outcome[:, 8:9]], axis=1).astype(np.float32)


def heldout_forward_error(train: dict, test: dict, steps: int = 3000, seed: int = 0) -> dict:
    """Train a fresh, fixed-size forward model on an agent's free-play data; test on held-out interactions.

    Returns the mean squared error of object displacement (cm^2) per object.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = fm_inputs(train["context"], train["theta"])
    y = fm_targets(train["context"], train["outcome"])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    ymu, ysd = y.mean(0), y.std(0) + 1e-3
    model = mlp(x.shape[1], y.shape[1], hidden=128, layers=2)
    fit_regressor(model, (x - mu) / sd, (y - ymu) / ysd, steps=steps, lr=1e-3, batch=256, rng=rng)
    xt = fm_inputs(test["context"], test["theta"])
    yt = fm_targets(test["context"], test["outcome"])
    with torch.no_grad():
        pred = model(torch.as_tensor((xt - mu) / sd)).numpy() * ysd + ymu
    err = (pred - yt) ** 2
    res = {o: float(err[:, 2 * i:2 * i + 2].sum(1).mean()) for i, o in enumerate(OBJECTS)}
    res["objects_mean"] = float(np.mean([res[o] for o in OBJECTS]))
    res["toy"] = float(err[:, 6].mean())
    return res


# ------------------------------------------------- downstream push tasks


def solve_push(env: Crib, obj: str, goal: np.ndarray, trials: int, rng: np.random.Generator,
               prior_theta: np.ndarray | None = None, prior_outcome: np.ndarray | None = None,
               sigma0: float = 0.15) -> np.ndarray:
    """(1+1)-ES with the 1/5th success rule on real trials.

    With a prior (free-play memory), the first command is the stored one whose
    outcome for ``obj`` was closest to the goal. Without it, the first command
    is uniform random. Returns the observed distance (m) at every trial.
    """
    sl = OBJ_SLICES[obj]
    if prior_theta is not None and len(prior_theta):
        j = int(np.argmin(np.linalg.norm(prior_outcome[:, sl] - goal, axis=1)))
        best = prior_theta[j].astype(np.float64).copy()
    else:
        best = rng.uniform(-1, 1, THETA_DIM)
    sigma = sigma0
    dists = np.zeros(trials)
    best_d = np.inf
    theta = best
    for t in range(trials):
        if t > 0:
            theta = np.clip(best + rng.normal(0, sigma, THETA_DIM), -1, 1)
        ep = env.run(theta)
        d = float(np.linalg.norm(ep.outcome[sl] - goal))
        dists[t] = d
        if d < best_d:
            best_d, best = d, theta
            if t > 0:
                sigma *= 1.5
        elif t > 0:
            sigma *= 1.5 ** -0.25
        sigma = float(np.clip(sigma, 0.01, 1.0))
    return dists


def make_env(seed: int) -> Crib:
    return Crib(EnvConfig(perception="state"), seed=seed)
