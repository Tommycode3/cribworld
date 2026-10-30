"""Free-play agents.

All agents act in the same space: one motor command theta in [-1, 1]^6 per
episode. They differ only in how they pick the next theta.

* ``random``    motor babbling, theta ~ U([-1, 1]^6).
* ``icm``       prediction-error curiosity in a learned feature space
                (forward + inverse model, Pathak et al. 2017).
* ``rnd``       novelty as the error of predicting a fixed random network
                (Burda et al. 2019).
* ``imgep_rmb`` modular goal exploration, module chosen uniformly at random.
* ``imgep_lp``  modular goal exploration, module chosen by learning progress
                (Baranes & Oudeyer 2013; Forestier et al. 2017).
* ``imgep_pca`` goal exploration in a PCA latent space learned from the camera
                images, for the pixel setting (cf. Pere et al. 2018).

The curiosity agents share one actor: a critic Q(theta) regresses the
intrinsic reward of recent episodes and the agent plays the best of K random
candidates (epsilon-greedy). The episode is the decision unit, so this is a
one-step (bandit) version of the RL formulation in the original papers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .env import MODULES, THETA_DIM, Episode
from .nets import Encoder, fit_regressor, mlp

# ---------------------------------------------------------------- pure maths


def icm_reward(phi1_pred: np.ndarray, phi1: np.ndarray) -> np.ndarray:
    """r = 1/2 * || f(phi(s0), a) - phi(s1) ||^2, one value per row."""
    return 0.5 * np.sum((np.asarray(phi1_pred) - np.asarray(phi1)) ** 2, axis=-1)


def rnd_reward(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """r = || f_hat(s1) - f(s1) ||^2, one value per row."""
    return np.sum((np.asarray(pred) - np.asarray(target)) ** 2, axis=-1)


def progress(dist_ref: float, dist_now: float) -> float:
    """Absolute competence progress for one goal.

    dist_ref is the distance between the goal and the outcome of the stored
    command we started from (what we could already do); dist_now is the
    distance after executing the perturbed command.
    """
    return abs(dist_ref - dist_now)


def module_probabilities(interest: np.ndarray, eps: float) -> np.ndarray:
    """p(m) = eps / M + (1 - eps) * LP_m / sum LP; uniform if all LP are zero."""
    interest = np.maximum(np.asarray(interest, dtype=np.float64), 0.0)
    m = len(interest)
    s = interest.sum()
    if s <= 0:
        return np.full(m, 1.0 / m)
    return eps / m + (1.0 - eps) * interest / s


def expanded_bounds(x: np.ndarray, margin: float = 0.1):
    lo, hi = x.min(axis=0), x.max(axis=0)
    r = np.maximum(hi - lo, 1e-3)
    return lo - margin * r, hi + margin * r


# ------------------------------------------------------------------- memory


class Memory:
    def __init__(self, perception: str, capacity: int, image_size: int = 32):
        self.perception = perception
        self.n = 0
        self.theta = np.zeros((capacity, THETA_DIM), np.float32)
        self.outcome = np.zeros((capacity, 9), np.float32)
        self.body0 = np.zeros((capacity, 8), np.float32)
        self.body1 = np.zeros((capacity, 8), np.float32)
        if perception == "pixels":
            self.x0 = np.zeros((capacity, image_size, image_size, 3), np.uint8)
            self.x1 = np.zeros((capacity, image_size, image_size, 3), np.uint8)
        else:
            self.x0 = np.zeros((capacity, 8), np.float32)
            self.x1 = np.zeros((capacity, 8), np.float32)

    def add(self, ep: Episode):
        i = self.n
        self.theta[i] = ep.theta
        self.outcome[i] = ep.outcome
        self.body0[i] = ep.obs0["body"]
        self.body1[i] = ep.obs1["body"]
        key = "image" if self.perception == "pixels" else "state"
        self.x0[i] = ep.obs0[key]
        self.x1[i] = ep.obs1[key]
        self.n += 1


@dataclass
class AgentConfig:
    bootstrap: int = 100        # random episodes before any agent-specific choice
    # curiosity agents
    update_every: int = 50
    model_steps: int = 50
    critic_steps: int = 50
    reward_window: int = 1000
    candidates: int = 64
    epsilon: float = 0.1
    feature_dim: int = 32
    icm_beta: float = 0.2
    lr: float = 1e-3
    batch: int = 128
    # goal exploration
    sigma: float = 0.05
    lp_window: int = 50
    lp_eps: float = 0.2
    pca_dim: int = 8
    pca_refit: int = 500


class Agent:
    name = "base"

    def __init__(self, perception: str, budget: int, seed: int, cfg: AgentConfig | None = None):
        self.cfg = cfg or AgentConfig()
        self.perception = perception
        self.rng = np.random.default_rng(seed + 12345)
        torch.manual_seed(seed)
        self.mem = Memory(perception, budget)
        self.log_module = []
        self.log_reward = []

    def random_theta(self):
        return self.rng.uniform(-1, 1, THETA_DIM)

    def propose(self) -> np.ndarray:
        return self.random_theta()

    def observe(self, ep: Episode) -> None:
        self.mem.add(ep)
        self.log_module.append(-1)
        self.log_reward.append(np.nan)


class RandomAgent(Agent):
    name = "random"


# --------------------------------------------------------- curiosity agents


class CuriosityAgent(Agent):
    """Shared actor for ICM and RND; subclasses define the intrinsic reward."""

    def __init__(self, perception, budget, seed, cfg=None):
        super().__init__(perception, budget, seed, cfg)
        self.critic = mlp(THETA_DIM, 1, hidden=64, layers=2)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self._norm = None
        self._build()

    # --- observation tensors
    def _fit_norm(self):
        n = self.mem.n
        if self.perception == "pixels":
            x = self.mem.x1[:n].astype(np.float32) / 255.0
            self._norm = (x.mean(0), x.std(0) + 0.05)
        else:
            x = np.concatenate([self.mem.x0[:n], self.mem.x1[:n]])
            self._norm = (x.mean(0), x.std(0) + 1e-2)
        b = np.concatenate([self.mem.body0[:n], self.mem.body1[:n]])
        self._bnorm = (b.mean(0), b.std(0) + 1e-2)

    def _tensors(self, idx, which: int):
        body = self.mem.body0[idx] if which == 0 else self.mem.body1[idx]
        x = self.mem.x0[idx] if which == 0 else self.mem.x1[idx]
        body = (body - self._bnorm[0]) / self._bnorm[1]
        if self.perception == "pixels":
            x = (x.astype(np.float32) / 255.0 - self._norm[0]) / self._norm[1]
            x = np.clip(x, -5, 5).transpose(0, 3, 1, 2)
        else:
            x = np.clip((x - self._norm[0]) / self._norm[1], -5, 5)
        return (torch.as_tensor(body, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32))

    def _build(self):
        raise NotImplementedError

    def _train_models(self, steps):
        raise NotImplementedError

    def rewards(self, idx) -> np.ndarray:
        raise NotImplementedError

    # --- actor
    def propose(self):
        if self.mem.n < self.cfg.bootstrap or self.rng.random() < self.cfg.epsilon:
            return self.random_theta()
        cand = self.rng.uniform(-1, 1, (self.cfg.candidates, THETA_DIM))
        with torch.no_grad():
            q = self.critic(torch.as_tensor(cand, dtype=torch.float32)).numpy().ravel()
        return cand[int(np.argmax(q))]

    def observe(self, ep):
        self.mem.add(ep)
        n = self.mem.n
        self.log_module.append(-1)
        if n == self.cfg.bootstrap:
            self._fit_norm()
        if n >= self.cfg.bootstrap and self._norm is not None:
            self.log_reward.append(float(self.rewards(np.array([n - 1]))[0]))
        else:
            self.log_reward.append(np.nan)
        if n >= self.cfg.bootstrap and n % self.cfg.update_every == 0:
            self._train_models(self.cfg.model_steps)
            lo = max(0, n - self.cfg.reward_window)
            idx = np.arange(lo, n)
            r = self.rewards(idx)
            r = (r - r.mean()) / (r.std() + 1e-8)
            fit_regressor_inplace(self.critic, self.critic_opt, self.mem.theta[idx], r[:, None],
                                  self.cfg.critic_steps, self.cfg.batch, self.rng)


def fit_regressor_inplace(model, opt, x, y, steps, batch, rng):
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    for _ in range(steps):
        i = torch.as_tensor(rng.integers(0, len(xt), size=min(batch, len(xt))))
        loss = ((model(xt[i]) - yt[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


class ICMAgent(CuriosityAgent):
    name = "icm"

    def _build(self):
        f = self.cfg.feature_dim
        self.enc = Encoder(self.perception, 8, 8, out=f)
        self.inv = mlp(2 * f, THETA_DIM, hidden=64, layers=2)
        self.fwd = mlp(f + THETA_DIM, f, hidden=64, layers=2)
        params = list(self.enc.parameters()) + list(self.inv.parameters()) + list(self.fwd.parameters())
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr)

    def _train_models(self, steps):
        beta = self.cfg.icm_beta
        for _ in range(steps):
            idx = self.rng.integers(0, self.mem.n, size=min(self.cfg.batch, self.mem.n))
            phi0 = self.enc(*self._tensors(idx, 0))
            phi1 = self.enc(*self._tensors(idx, 1))
            a = torch.as_tensor(self.mem.theta[idx], dtype=torch.float32)
            l_inv = ((self.inv(torch.cat([phi0, phi1], 1)) - a) ** 2).sum(1).mean()
            l_fwd = 0.5 * ((self.fwd(torch.cat([phi0, a], 1)) - phi1.detach()) ** 2).sum(1).mean()
            loss = (1 - beta) * l_inv + beta * l_fwd
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    @torch.no_grad()
    def rewards(self, idx):
        phi0 = self.enc(*self._tensors(idx, 0))
        phi1 = self.enc(*self._tensors(idx, 1))
        a = torch.as_tensor(self.mem.theta[idx], dtype=torch.float32)
        return icm_reward(self.fwd(torch.cat([phi0, a], 1)).numpy(), phi1.numpy())


class RNDAgent(CuriosityAgent):
    name = "rnd"

    def _build(self):
        f = self.cfg.feature_dim
        self.target = Encoder(self.perception, 8, 8, out=f)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred = Encoder(self.perception, 8, 8, out=f)
        self.opt = torch.optim.Adam(self.pred.parameters(), lr=self.cfg.lr)

    def _train_models(self, steps):
        for _ in range(steps):
            idx = self.rng.integers(0, self.mem.n, size=min(self.cfg.batch, self.mem.n))
            b, x = self._tensors(idx, 1)
            loss = ((self.pred(b, x) - self.target(b, x)) ** 2).sum(1).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    @torch.no_grad()
    def rewards(self, idx):
        b, x = self._tensors(idx, 1)
        return rnd_reward(self.pred(b, x).numpy(), self.target(b, x).numpy())


# ------------------------------------------------------ goal exploration


class GoalExplorer(Agent):
    """IMGEP with a nearest-neighbour inverse model.

    Each episode: pick a module m, sample a goal g uniformly in the (observed,
    slightly expanded) range of that module, find the stored command whose
    outcome in module m is closest to g, perturb it with Gaussian noise and
    execute it. The outcome of *every* module is stored, whatever the goal.
    """

    def __init__(self, perception, budget, seed, cfg=None, selection: str = "lp"):
        super().__init__(perception, budget, seed, cfg)
        self.selection = selection
        self.modules = list(MODULES.keys())
        self.slices = [MODULES[m] for m in self.modules]
        self.recent = [deque(maxlen=self.cfg.lp_window) for _ in self.modules]
        self.interest = np.zeros(len(self.modules))
        self._pending = None

    @property
    def name(self):
        return f"imgep_{self.selection}"

    def outcomes(self, n, refit: bool = True):
        return self.mem.outcome[:n]

    def choose_module(self) -> int:
        if self.selection == "rmb":
            return int(self.rng.integers(len(self.modules)))
        p = module_probabilities(self.interest, self.cfg.lp_eps)
        return int(self.rng.choice(len(self.modules), p=p))

    def propose(self):
        n = self.mem.n
        if n < self.cfg.bootstrap:
            self._pending = None
            return self.random_theta()
        m = self.choose_module()
        sl = self.slices[m]
        out = self.outcomes(n)[:, sl]
        lo, hi = expanded_bounds(out)
        goal = self.rng.uniform(lo, hi)
        scale = np.maximum(hi - lo, 1e-3)
        d = np.linalg.norm((out - goal) / scale, axis=1)
        j = int(np.argmin(d))
        theta = np.clip(self.mem.theta[j] + self.rng.normal(0, self.cfg.sigma, THETA_DIM), -1, 1)
        self._pending = (m, goal, scale, float(d[j]))
        return theta

    def observe(self, ep):
        self.mem.add(ep)
        if self._pending is None:
            self.log_module.append(-1)
            self.log_reward.append(np.nan)
            return
        m, goal, scale, d_ref = self._pending
        o = self.outcomes(self.mem.n, refit=False)[-1, self.slices[m]]
        d_now = float(np.linalg.norm((o - goal) / scale))
        p = progress(d_ref, d_now)
        self.recent[m].append(p)
        self.interest[m] = float(np.mean(self.recent[m]))
        self.log_module.append(m)
        self.log_reward.append(p)


class PCAGoalExplorer(GoalExplorer):
    """Goal exploration in a PCA latent space of the final camera image."""

    def __init__(self, perception, budget, seed, cfg=None):
        assert perception == "pixels"
        super().__init__(perception, budget, seed, cfg, selection="pca")
        self.modules = ["latent"]
        self.slices = [slice(0, self.cfg.pca_dim)]
        self.recent = [deque(maxlen=self.cfg.lp_window)]
        self.interest = np.zeros(1)
        self._latent = None
        self._fit_at = -1

    def _refit(self, n):
        x = self.mem.x1[:n].reshape(n, -1).astype(np.float32) / 255.0
        self._mu = x.mean(0)
        _, _, vt = np.linalg.svd(x - self._mu, full_matrices=False)
        self._w = vt[: self.cfg.pca_dim].T
        self._latent = (x - self._mu) @ self._w
        self._fit_at = n

    def outcomes(self, n, refit: bool = True):
        if self._latent is None or (refit and n - self._fit_at >= self.cfg.pca_refit and n >= self.cfg.bootstrap):
            self._refit(n)
        if len(self._latent) < n:
            k = len(self._latent)
            x = self.mem.x1[k:n].reshape(n - k, -1).astype(np.float32) / 255.0
            self._latent = np.concatenate([self._latent, (x - self._mu) @ self._w])
        return self._latent[:n]

    def choose_module(self):
        return 0


def make_agent(name: str, perception: str, budget: int, seed: int, cfg: AgentConfig | None = None) -> Agent:
    if name == "random":
        return RandomAgent(perception, budget, seed, cfg)
    if name == "icm":
        return ICMAgent(perception, budget, seed, cfg)
    if name == "rnd":
        return RNDAgent(perception, budget, seed, cfg)
    if name == "imgep_lp":
        return GoalExplorer(perception, budget, seed, cfg, selection="lp")
    if name == "imgep_rmb":
        return GoalExplorer(perception, budget, seed, cfg, selection="rmb")
    if name == "imgep_pca":
        return PCAGoalExplorer(perception, budget, seed, cfg)
    raise ValueError(name)
