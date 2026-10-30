"""Small networks and helpers. Everything here runs on CPU in seconds."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(inp: int, out: int, hidden: int = 64, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ELU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class Encoder(nn.Module):
    """Maps a sensed observation to a feature vector.

    ``state`` perception: MLP over [body, object state].
    ``pixels`` perception: MLP over the flattened 32x32 RGB image, concatenated
    with the body vector (proprioception + touch).
    """

    def __init__(self, perception: str, body_dim: int, state_dim: int, out: int = 32, image_size: int = 32):
        super().__init__()
        self.perception = perception
        if perception == "pixels":
            # A flat MLP over the 32x32x3 image. A conv net is the natural choice, but
            # PyTorch's CPU convolutions were ~20x slower here, so we keep it flat.
            self.pix = nn.Sequential(nn.Flatten(), nn.Linear(3 * image_size * image_size, 128), nn.ELU())
            self.head = mlp(128 + body_dim, out, hidden=64, layers=1)
        else:
            self.head = mlp(body_dim + state_dim, out, hidden=64, layers=2)

    def forward(self, body: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.perception == "pixels":
            return self.head(torch.cat([self.pix(x), body], dim=1))
        return self.head(torch.cat([body, x], dim=1))


class RunningNorm:
    """Mean/std normaliser fitted on all data seen so far (Welford update)."""

    def __init__(self, dim: int, eps: float = 1e-4):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.eps = eps

    def update(self, x: np.ndarray) -> None:
        x = np.atleast_2d(x)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (row - self.mean)

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.m2 / (self.n - 1)) + self.eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def fit_regressor(model: nn.Module, x: np.ndarray, y: np.ndarray, steps: int, lr: float = 1e-3,
                  batch: int = 128, rng: np.random.Generator | None = None, weight_decay: float = 0.0) -> float:
    """Plain minibatch Adam regression; returns the final full-batch MSE."""
    rng = rng or np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    n = len(xt)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, size=min(batch, n)))
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((model(xt) - yt) ** 2).mean())
