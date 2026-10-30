"""Figures + summary table from metrics.pkl and downstream.pkl."""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cribworld.env import EVENT_NAMES, MODULES  # noqa: E402
from cribworld.utils import load_config  # noqa: E402

COLORS = {"random": "#52514e", "icm": "#2a78d6", "rnd": "#eb6834", "imgep_rmb": "#1baf7a",
          "imgep_lp": "#6250d6", "imgep_pca": "#e87ba4", "scratch": "#eda100"}
LABEL = {"random": "random babbling", "icm": "ICM curiosity", "rnd": "RND novelty",
         "imgep_rmb": "goal babbling, random module", "imgep_lp": "goal babbling, learning progress",
         "imgep_pca": "goal babbling, PCA of pixels", "scratch": "from scratch (no free play)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "legend.frameon": False})


def ci(x, axis=0):
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    m = x.mean(axis)
    if n < 2:
        return m, np.zeros_like(m)
    h = stats.t.ppf(0.975, n - 1) * x.std(axis, ddof=1) / np.sqrt(n)
    return m, h


def fmt(x):
    m, h = ci(x)
    return f"{m:.2f} ± {h:.2f}" if abs(m) < 10 else f"{m:.0f} ± {h:.0f}"


p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--figdir", default=None)
args = p.parse_args()
cfg = load_config(args.config)
out = Path(cfg["out_dir"])
fig_dir = Path(args.figdir or cfg.get("fig_dir", "figures"))
fig_dir.mkdir(exist_ok=True)
met = pickle.load(open(out / "metrics.pkl", "rb"))
down = pickle.load(open(out / "downstream.pkl", "rb")) if (out / "downstream.pkl").exists() else {}
cps = np.array(cfg["checkpoints"])
conds = [("state", a) for a in cfg["state_agents"]] + [("pixels", a) for a in cfg["pixel_agents"]]
conds = [c for c in conds if any(k[:2] == c for k in met)]


def runs(c):
    return [met[k] for k in sorted(met) if k[:2] == c]


def style(c):
    return dict(color=COLORS[c[1]], ls="-" if c[0] == "state" else "--", lw=2)


def name(c):
    return LABEL[c[1]] + (" (pixels)" if c[0] == "pixels" else "")


# ---------------------------------------------------------- 1. exploration
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
panels = [("objects_total", "object-position coverage\n(2 cm cells reached, 3 objects summed)"),
          ("toy", "toy coverage\n(distinct lever-deflection bins)"),
          ("types", "interaction types seen ≥ 5 times\n(of %d)" % len(EVENT_NAMES))]
for ax, (key, title) in zip(axes, panels):
    for c in conds:
        R = runs(c)
        if key == "objects_total":
            y = np.array([r["coverage"]["objects_total"] for r in R])
        elif key == "toy":
            y = np.array([r["toy_coverage"] for r in R])
        else:
            y = np.array([r["discovered5"] for r in R])
        m, h = ci(y)
        x = R[0]["cps"]
        ax.plot(x, m, label=name(c), **style(c))
        ax.fill_between(x, m - h, m + h, color=COLORS[c[1]], alpha=0.12, lw=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("free-play episodes")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(fig_dir / "exploration.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------- 2. developmental trajectory
focus_keys = [("body_only", "no object touched", "#8a8984"), ("light_box", "light box", "#2a78d6"),
              ("ball", "ball", "#eb6834"), ("heavy_box", "heavy box", "#1baf7a"),
              ("toy", "toy touched", "#6250d6"), ("toy_light", "toy light on", "#eda100")]
fig, axes = plt.subplots(2, len(conds) // 2 + len(conds) % 2 + 1, figsize=(16, 6.4), sharey=False)
axes = axes.ravel()
for ax, c in zip(axes, conds):
    R = runs(c)
    for k, lab, col in focus_keys:
        y = np.array([r["focus"][k] for r in R])
        m, h = ci(y)
        x = (np.arange(len(m)) + 0.5) * 250
        ax.plot(x, m, color=col, lw=2, label=lab)
        ax.fill_between(x, m - h, m + h, color=col, alpha=0.12, lw=0)
    ax.set_ylim(0, 1)
    ax.set_title(name(c), loc="left", fontsize=9)
    ax.set_xlabel("episode")
axes[0].set_ylabel("fraction of episodes")
axes[0].legend(fontsize=7, loc="upper right")
# module choice of LP agent
lp = ("state", "imgep_lp")
axm = axes[len(conds)]
if lp in conds:
    mods = list(MODULES)
    mcols = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    W = 100
    for i, (mname, col) in enumerate(zip(mods, mcols)):
        ys = []
        for r in runs(lp):
            mod = r["module"]
            n = len(mod) // W
            ys.append((mod[: n * W].reshape(n, W) == i).mean(1))
        m, h = ci(np.array(ys))
        x = (np.arange(len(m)) + 0.5) * W
        axm.plot(x, m, color=col, lw=2, label=mname)
        axm.fill_between(x, m - h, m + h, color=col, alpha=0.12, lw=0)
    axm.set_title("learning-progress agent:\nwhich goal space it chooses", loc="left", fontsize=9)
    axm.set_xlabel("episode")
    axm.legend(fontsize=7)
for ax in axes[len(conds) + 1:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(fig_dir / "trajectory.png", dpi=150)
plt.close(fig)

# --------------------------------------------------- 3. held-out forward model
fcp = cfg["fm_checkpoints"]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
for ax, key, title in zip(axes, ["objects_mean", "light_box", "ball", "heavy_box"],
                          ["all objects", "light box", "ball", "heavy box"]):
    for c in conds:
        tt = sorted(runs(c)[0]["fm"])
        y = np.array([[r["fm"][t][key] for t in tt] for r in runs(c)])
        m, h = ci(y)
        ax.errorbar(tt, m, yerr=h, label=name(c), capsize=2, marker="o", ms=4, **style(c))
    ax.set_xscale("log")
    ax.set_xticks(fcp)
    ax.set_xticklabels([str(t) for t in fcp])
    ax.minorticks_off()
    ax.set_title(f"held-out displacement error, {title}", loc="left", fontsize=9)
    ax.set_xlabel("free-play episodes used for training")
axes[0].set_ylabel("MSE (cm²)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(fig_dir / "forward_model.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------- 4. downstream
dconds = conds + [("none", "scratch")]
dconds = [c for c in dconds if any(k[:2] == c for k in down)]
if down:
    objs = cfg["downstream"]["objects"]
    fig, axes = plt.subplots(1, len(objs), figsize=(6 * len(objs), 3.8))
    for ax, obj in zip(np.atleast_1d(axes), objs):
        for c in dconds:
            per_seed = [np.minimum.accumulate(down[k][obj], 1).mean(0) * 100 for k in sorted(down) if k[:2] == c]
            m, h = ci(np.array(per_seed))
            x = np.arange(1, len(m) + 1)
            st = style(c) if c[0] != "none" else dict(color=COLORS["scratch"], ls=":", lw=2.5)
            ax.plot(x, m, label=name(c) if c[0] != "none" else LABEL["scratch"], **st)
            ax.fill_between(x, np.maximum(m - h, 0), m + h, color=st["color"], alpha=0.10, lw=0)
        ax.set_ylim(0, None)
        ax.set_title(f"push the {obj.replace('_', ' ')} to a target: best distance so far", loc="left", fontsize=9)
        ax.set_xlabel("real trials on the task")
        ax.set_ylabel("distance to target (cm)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "downstream.png", dpi=150)
    plt.close(fig)

# ------------------------------------------------------------- summary table
T = cps[-1]
# matched budget: the largest forward-model checkpoint every condition reached
MB = max(t for t in cfg["fm_checkpoints"] if t <= min(cfg["episodes"], cfg.get("pixel_episodes", cfg["episodes"])))
lines = ["Free play (values at the end of each run's budget; mean ± 95% CI over seeds).", "",
         f"| condition | episodes × seeds | object coverage @{MB} | held-out FM error @{MB} (cm²) | object coverage (cells) | toy bins | types ≥5 | toy lit (% episodes) | held-out FM error (cm²) |",
         "|---|---|---|---|---|---|---|---|---|"]
for c in conds:
    R = runs(c)
    i2k = int(np.argmin(np.abs(np.array(R[0]["cps"]) - MB)))
    lines.append("| %s | %d × %d | %s | %s | %s | %s | %s | %s | %s |" % (
        name(c), R[0]["cps"][-1], len(R), fmt([r["coverage"]["objects_total"][i2k] for r in R]),
        fmt([r["fm"][MB]["objects_mean"] for r in R]), fmt([r["coverage"]["objects_total"][-1] for r in R]), fmt([r["toy_coverage"][-1] for r in R]),
        fmt([r["discovered5"][-1] for r in R]),
        fmt([100 * r["event_rate"][EVENT_NAMES.index("toy_light_on")] for r in R]),
        fmt([r["fm"][max(r["fm"])]["objects_mean"] for r in R])))
lines.append("")
if down:
    succ = 3.0
    ntr = cfg["downstream"]["trials"]
    hdr = "| condition | " + " | ".join(f"{o}: best dist after 1 / 10 / {ntr} trials (cm) | {o}: success ≤{succ:.0f} cm after {ntr}" for o in objs) + " |"
    lines += [hdr, "|---|" + "---|---|" * len(objs)]
    for c in dconds:
        row = [name(c) if c[0] != "none" else LABEL["scratch"]]
        for obj in objs:
            ks = [k for k in sorted(down) if k[:2] == c]
            best = [np.minimum.accumulate(down[k][obj], 1) * 100 for k in ks]
            tr = best[0].shape[1]
            vals = [fmt([b[:, min(t, tr) - 1].mean() for b in best]) for t in (1, 10, tr)]
            row.append(" / ".join(vals))
            row.append(fmt([100 * (b[:, -1] <= succ).mean() for b in best]) + " %")
        lines.append("| " + " | ".join(row) + " |")
lines.append("")
lines.append("Discovery episode (mean over seeds that found it; n found / n seeds):")
lines.append("")
lines.append("| event | " + " | ".join(name(c) for c in conds) + " |")
lines.append("|---|" + "---|" * len(conds))
for i, ev in enumerate(EVENT_NAMES):
    row = [ev]
    for c in conds:
        f = np.array([r["first"][i] for r in runs(c)])
        ok = ~np.isnan(f)
        row.append(f"{np.mean(f[ok]):.0f} ({ok.sum()}/{len(f)})" if ok.any() else f"never (0/{len(f)})")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")
lines.append("Wall-clock per free-play run (s): " + ", ".join(
    f"{name(c)} {np.mean([r['wallclock'] for r in runs(c)]):.0f}" for c in conds))
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
