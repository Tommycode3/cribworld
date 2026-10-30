"""Run every free-play condition in the config, one at a time. Existing results are skipped."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cribworld.freeplay import run_freeplay  # noqa: E402
from cribworld.utils import load_config, run_path, wait_for_memory  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--only", default=None, help="run a single agent name")
args = p.parse_args()
cfg = load_config(args.config)

jobs = [("state", a, s, cfg["episodes"]) for s in cfg["seeds"] for a in cfg["state_agents"]]
jobs += [("pixels", a, s, cfg["pixel_episodes"]) for s in cfg["pixel_seeds"] for a in cfg["pixel_agents"]]
for perception, agent, seed, episodes in jobs:
    if True:
        if args.only and agent != args.only:
            continue
        out = run_path(cfg["out_dir"], perception, agent, seed)
        if out.exists():
            continue
        wait_for_memory(cfg["min_free_memory_pct"])
        print(f"run {perception}/{agent}/seed {seed}", flush=True)
        r = run_freeplay(agent, perception, seed, episodes, out=out)
        print(f"  done in {r['wallclock']:.0f}s", flush=True)
