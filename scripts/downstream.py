"""After free play: push an object to a given target. Prior = the agent's own free-play memory.

Every condition uses the same solver, (1+1)-ES on real trials; only the
starting command differs. 'scratch' starts from a random command.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cribworld.env import set_headless_gl  # noqa: E402
from cribworld.metrics import make_env, solve_push  # noqa: E402
from cribworld.utils import load_config, run_path, wait_for_memory  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
args = p.parse_args()
cfg = load_config(args.config)
set_headless_gl()
out = Path(cfg["out_dir"]) / "downstream.pkl"
done = pickle.load(open(out, "rb")) if out.exists() else {}
hold = np.load(Path(cfg["out_dir"]) / "heldout.npz")
d = cfg["downstream"]
conds = [("state", a) for a in cfg["state_agents"]] + [("pixels", a) for a in cfg["pixel_agents"]] + [("none", "scratch")]
for perception, agent in conds:
    for seed in (cfg["pixel_seeds"] if perception == "pixels" else cfg["seeds"]):
        key = (perception, agent, seed)
        if key in done:
            continue
        wait_for_memory(cfg["min_free_memory_pct"])
        if agent == "scratch":
            prior_t = prior_o = None
        else:
            r = np.load(run_path(cfg["out_dir"], perception, agent, seed))
            prior_t, prior_o = r["theta"], r["outcome"]
        env = make_env(seed=5000 + seed)
        rng = np.random.default_rng(7000 + seed)
        res = {}
        for obj in d["objects"]:
            res[obj] = np.stack([solve_push(env, obj, g, d["trials"], rng, prior_t, prior_o)
                                 for g in hold[f"target_{obj}"]])
        done[key] = res
        print(key, {o: round(float(np.minimum.accumulate(v, 1)[:, -1].mean() * 100), 2) for o, v in res.items()},
              "cm", flush=True)
        pickle.dump(done, open(out, "wb"))
