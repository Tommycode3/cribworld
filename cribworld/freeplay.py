"""Run one free-play phase and save everything the evaluation needs."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .agents import AgentConfig, make_agent
from .env import Crib, EnvConfig, set_headless_gl


def run_freeplay(agent_name: str, perception: str, seed: int, episodes: int,
                 agent_cfg: AgentConfig | None = None, env_cfg: EnvConfig | None = None,
                 out: str | Path | None = None, verbose: bool = True) -> dict:
    set_headless_gl()
    torch.set_num_threads(2)
    env_cfg = env_cfg or EnvConfig()
    env_cfg.perception = perception
    env = Crib(env_cfg, seed=seed)
    agent = make_agent(agent_name, perception, episodes, seed, agent_cfg)
    n_ev = None
    log = {"context": [], "events": []}
    t0 = time.time()
    for i in range(episodes):
        theta = agent.propose()
        ep = env.run(theta)
        agent.observe(ep)
        log["context"].append(ep.context)
        log["events"].append(ep.events)
        if verbose and (i + 1) % 1000 == 0:
            ev = np.array(log["events"])
            n_ev = int(ev.any(0).sum())
            print(f"  [{agent_name}/{perception}/s{seed}] {i + 1}/{episodes} "
                  f"types={n_ev} {time.time() - t0:.0f}s", flush=True)
    env.close()
    res = {
        "agent": agent_name, "perception": perception, "seed": seed,
        "theta": agent.mem.theta[:episodes].copy(),
        "outcome": agent.mem.outcome[:episodes].copy(),
        "context": np.array(log["context"], np.float32),
        "events": np.array(log["events"], bool),
        "module": np.array(agent.log_module, np.int16),
        "intrinsic": np.array(agent.log_reward, np.float32),
        "wallclock": time.time() - t0,
    }
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res)
    return res
