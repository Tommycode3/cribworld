"""Render a GIF of one agent's free play: logged episodes replayed exactly (same command, same start)."""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cribworld.env import EVENT_NAMES, Crib, EnvConfig, set_headless_gl  # noqa: E402
from cribworld.utils import load_config, run_path  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--agent", default="imgep_lp")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
args = p.parse_args()
set_headless_gl()
cfg = load_config(args.config)
out_path = args.out or str(Path(cfg.get("fig_dir", "figures")) / "freeplay.gif")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
r = np.load(run_path(cfg["out_dir"], "state", args.agent, args.seed))
ev = r["events"]
n = len(ev)
light = EVENT_NAMES.index("toy_light_on")
far = [EVENT_NAMES.index("push_light_box_far"), EVENT_NAMES.index("roll_ball_far")]


def first(lo, hi, cols):
    idx = [i for i in range(lo, hi) if ev[i, cols].all()]
    return idx[0] if idx else None


E = EVENT_NAMES.index
chosen = [
    (5, "random babbling (bootstrap)"),
    (first(0, n, [E("toy_light_on")]), "first time the toy lights up"),
    (first(n // 3, n, [E("push_heavy_box_15cm")]), "heavy box pushed 15 cm"),
    (first(n // 3, n, [E("light_box_to_back")]), "light box pushed to the back"),
    (first(2 * n // 3, n, [E("toy_light_on"), E("move_light_box")]), "late: box moved and toy lit"),
]
chosen = sorted((i, c) for i, c in chosen if i is not None)
env = Crib(EnvConfig(), seed=0)
frames = []
for i, caption in chosen:
    ep, fr = env.run(r["theta"][i], record=True, record_every=3, context=r["context"][i])
    # float32 logs + contact chaos: replay matches the log to within millimetres, not bit for bit
    err = float(np.abs(ep.outcome - r["outcome"][i]).max())
    print(f"episode {i}: max outcome difference vs log {err * 1000:.2f} mm; events equal: "
          f"{bool((ep.events == ev[i]).all())}")
    for f in fr:
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 18], fill=(252, 252, 251))
        d.text((6, 3), f"{args.agent}  episode {i}: {caption}", fill=(11, 11, 11))
        frames.append(np.asarray(im))
    frames += [frames[-1]] * 10
imageio.mimsave(out_path, frames, duration=0.05, loop=0)
print("wrote", out_path, len(frames), "frames")
