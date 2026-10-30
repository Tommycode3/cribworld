import numpy as np
import torch

from cribworld.nets import fit_regressor, mlp


def test_forward_model_learns_known_linear_system():
    """s' = A s + B a: the same MLP + optimiser used everywhere must recover it."""
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    A = rng.normal(0, 0.5, (4, 4))
    B = rng.normal(0, 0.5, (4, 2))
    s = rng.uniform(-1, 1, (2000, 4))
    a = rng.uniform(-1, 1, (2000, 2))
    y = s @ A.T + a @ B.T
    x = np.concatenate([s, a], 1)
    model = mlp(6, 4, hidden=64, layers=2)
    fit_regressor(model, x[:1500], y[:1500], steps=3000, lr=3e-3, batch=128, rng=rng)
    with torch.no_grad():
        pred = model(torch.as_tensor(x[1500:], dtype=torch.float32)).numpy()
    mse = np.mean((pred - y[1500:]) ** 2)
    assert mse < 0.01 * np.var(y), mse
