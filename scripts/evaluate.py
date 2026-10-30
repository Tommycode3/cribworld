"""Compute every free-play metric from the saved runs -> <out_dir>/metrics.pkl"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cribworld import metrics as M  # noqa: E402
from cribworld.utils import load_config, run_path  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
args = p.parse_args()
cfg = load_config(args.config)
hold = np.load(Path(cfg["out_dir"]) / "heldout.npz")
test = {k[5:]: hold[k] for k in hold.files if k.startswith("test_")}

res = {}
conds = [("state", a) for a in cfg["state_agents"]] + [("pixels", a) for a in cfg["pixel_agents"]]
for perception, agent in conds:
    for seed in (cfg["seeds"] if perception == "state" else cfg["pixel_seeds"]):
        f = run_path(cfg["out_dir"], perception, agent, seed)
        if not f.exists():
            print("missing", f)
            continue
        r = dict(np.load(f))
        cps = [c for c in cfg["checkpoints"] if c <= len(r["theta"])]
        m = {"coverage": M.object_coverage(r["outcome"], cps),
             "toy_coverage": M.toy_coverage(r["outcome"], cps),
             "discovered": M.discovered_curve(r["events"], cps),
             "discovered5": M.discovered_curve(r["events"], cps, min_count=5),
             "first": M.first_discovery(r["events"]),
             "event_rate": r["events"].mean(0),
             "focus": M.focus(r["events"]),
             "module": r["module"], "intrinsic": r["intrinsic"],
             "wallclock": float(r["wallclock"]), "fm": {}, "cps": np.array(cps)}
        for t in [t for t in cfg["fm_checkpoints"] if t <= len(r["theta"])]:
            sub = {k: r[k][:t] for k in ("context", "theta", "outcome")}
            m["fm"][t] = M.heldout_forward_error(sub, test, seed=seed)
        res[(perception, agent, seed)] = m
        print(perception, agent, seed, "cov", m["coverage"]["objects_total"][-1],
              "types", m["discovered"][-1], "fm", round(m["fm"][max(m["fm"])]["objects_mean"], 2),
              flush=True)
with open(Path(cfg["out_dir"]) / "metrics.pkl", "wb") as f:
    pickle.dump(res, f)
