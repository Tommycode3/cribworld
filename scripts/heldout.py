"""Build the held-out interaction set and the downstream push targets.

The pool mixes episodes produced by every state agent on a seed never used for
training, so no single exploration strategy defines what "held-out" means.
Only episodes in which at least one object moved more than 2 cm are kept.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cribworld.freeplay import run_freeplay  # noqa: E402
from cribworld.metrics import OBJ_SLICES  # noqa: E402
from cribworld.utils import load_config, wait_for_memory  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
args = p.parse_args()
cfg = load_config(args.config)
h = cfg["heldout"]
out = Path(cfg["out_dir"]) / "heldout.npz"
if out.exists():
    print("exists:", out)
    sys.exit(0)

pool = {k: [] for k in ("theta", "context", "outcome", "events")}
for agent in cfg["state_agents"]:
    wait_for_memory(cfg["min_free_memory_pct"])
    print("pool:", agent, flush=True)
    r = run_freeplay(agent, "state", h["seed"], h["episodes_per_agent"], verbose=False)
    for k in pool:
        pool[k].append(r[k])
pool = {k: np.concatenate(v) for k, v in pool.items()}

disp = np.stack([np.linalg.norm(pool["outcome"][:, s] - pool["context"][:, 2 * i:2 * i + 2], axis=1)
                 for i, s in enumerate(OBJ_SLICES.values())], 1)
moved = disp.max(1) > 0.02
rng = np.random.default_rng(h["seed"])
cand = np.flatnonzero(moved)
test = rng.choice(cand, size=min(h["test_size"], len(cand)), replace=False)

d = cfg["downstream"]
targets = {}
for i, obj in enumerate(OBJ_SLICES):
    if obj not in d["objects"]:
        continue
    ok = np.flatnonzero(disp[:, i] > d["min_displacement"])
    pick = rng.choice(ok, size=min(d["targets_per_object"], len(ok)), replace=False)
    targets[obj] = pool["outcome"][pick][:, OBJ_SLICES[obj]]

np.savez_compressed(out, **{f"test_{k}": v[test] for k, v in pool.items()},
                    **{f"target_{k}": v for k, v in targets.items()},
                    pool_moved_fraction=moved.mean())
print(f"test set: {len(test)} interactions; moved fraction in pool {moved.mean():.2f}")
