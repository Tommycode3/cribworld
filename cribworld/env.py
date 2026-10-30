"""The crib environment.

One *episode* is one motor primitive: the arm starts folded at the home pose,
visits three joint-space via-points (each held for ``hold`` seconds), then rests
while the scene settles. The motor command is a vector theta in [-1, 1]^6.

The simulator exposes three kinds of signals:

* what the agent senses (``obs``): proprioception, touch and either a 32x32
  egocentric camera image (``perception="pixels"``) or object states
  (``perception="state"``, the oracle ablation);
* the outcome vector used by goal-exploration agents (state perception only);
* ground-truth bookkeeping used only by the evaluation code (``info``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .assets import CRIB_XML

OBJECTS = ("light_box", "ball", "heavy_box")
THETA_DIM = 6
N_VIA = 3
# Interaction types detected from ground truth (evaluation only).
EVENT_NAMES = (
    "touch_light_box", "touch_ball", "touch_heavy_box", "touch_toy",
    "move_light_box", "move_ball", "move_heavy_box",
    "push_light_box_far", "roll_ball_far", "push_heavy_box_far",
    "toy_light_on", "object_hits_object", "object_hits_wall",
    # rarer outcomes (about 0-5% of random-babbling episodes)
    "push_heavy_box_15cm", "toy_full_swing", "light_box_to_back", "light_box_to_right_half",
)
# Outcome-space modules used by modular goal exploration (state perception).
MODULES = {
    "hand": slice(0, 2),
    "light_box": slice(2, 4),
    "ball": slice(4, 6),
    "heavy_box": slice(6, 8),
    "toy": slice(8, 9),
}
OUTCOME_DIM = 9
CONTEXT_DIM = 6
LIGHT_THRESHOLD = 0.5  # |lever angle| (rad) that switches the toy light on
HOME_Q = np.array([0.35, 2.5])  # folded arm, hand close to the body


@dataclass
class EnvConfig:
    hold: float = 0.3          # seconds per via-point
    settle: float = 0.3        # seconds of rest after the last via-point
    frame_skip: int = 2         # physics steps per control step
    timestep: float = 0.004
    context_noise: float = 0.015  # +- uniform jitter of object start xy (m)
    perception: str = "state"   # "state" or "pixels"
    image_size: int = 32
    camera: str = "eye"


@dataclass
class Episode:
    theta: np.ndarray
    context: np.ndarray            # true start xy of the 3 objects (6,)
    outcome: np.ndarray            # (9,) hand xy, 3 objects xy, max lever angle
    obs0: np.ndarray               # sensed observation at start
    obs1: np.ndarray               # sensed observation at end
    events: np.ndarray             # (len(EVENT_NAMES),) bool
    contacts: dict = field(default_factory=dict)


class Crib:
    def __init__(self, cfg: EnvConfig | None = None, seed: int = 0):
        self.cfg = cfg or EnvConfig()
        self.model = mujoco.MjModel.from_xml_string(CRIB_XML)
        self.model.opt.timestep = self.cfg.timestep
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        m = self.model
        self._jnt_lo = m.jnt_range[[m.joint("shoulder").id, m.joint("elbow").id], 0].copy()
        self._jnt_hi = m.jnt_range[[m.joint("shoulder").id, m.joint("elbow").id], 1].copy()
        self._obj_qadr = {o: m.jnt_qposadr[m.joint(o).id] for o in OBJECTS}
        self._lever_qadr = m.jnt_qposadr[m.joint("lever").id]
        self._hand_site = m.site("hand").id
        self._light_geom = m.geom("toy_light").id
        self._light_off = m.geom_rgba[self._light_geom].copy()
        self._light_on = np.array([1.0, 0.95, 0.2, 1.0])
        self._renderer = None
        mujoco.mj_resetData(m, self.data)
        mujoco.mj_forward(m, self.data)
        self._home_xy = {o: self.data.qpos[self._obj_qadr[o]:self._obj_qadr[o] + 2].copy() for o in OBJECTS}

    # ------------------------------------------------------------------ utils
    def theta_to_targets(self, theta: np.ndarray) -> np.ndarray:
        th = np.clip(np.asarray(theta, dtype=np.float64).reshape(N_VIA, 2), -1.0, 1.0)
        return self._jnt_lo + (th + 1.0) * 0.5 * (self._jnt_hi - self._jnt_lo)

    def _obj_xy(self, o):
        a = self._obj_qadr[o]
        return self.data.qpos[a:a + 2].copy()

    def _render(self):
        if self._renderer is None:
            s = self.cfg.image_size
            self._renderer = mujoco.Renderer(self.model, height=s, width=s)
        self._renderer.update_scene(self.data, camera=self.cfg.camera)
        return self._renderer.render().copy()  # (H, W, 3) uint8

    def _proprio_touch(self, touch_max):
        d = self.data
        q = d.qpos[:2]
        return np.concatenate([np.cos(q), np.sin(q), d.qvel[:2] * 0.1,
                               [np.log1p(touch_max["hand"]), np.log1p(touch_max["arm"])]])

    def _sense(self, touch_max, lever_max, light_on):
        body = self._proprio_touch(touch_max)
        if self.cfg.perception == "pixels":
            return {"body": body.astype(np.float32), "image": self._render()}
        objs = np.concatenate([self._obj_xy(o) for o in OBJECTS] + [[lever_max, float(light_on)]])
        return {"body": body.astype(np.float32), "state": objs.astype(np.float32)}

    # ---------------------------------------------------------------- episode
    def reset(self, context: np.ndarray | None = None):
        """Reset the scene; ``context`` (6,) fixes the object start positions instead of sampling them."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        ctx = []
        for i, o in enumerate(OBJECTS):
            a = self._obj_qadr[o]
            if context is not None:
                xy = np.asarray(context[2 * i:2 * i + 2], dtype=np.float64)
            else:
                xy = self._home_xy[o] + self.rng.uniform(-1, 1, 2) * self.cfg.context_noise
            d.qpos[a:a + 2] = xy
            ctx.append(xy)
        d.qpos[:2] = HOME_Q
        d.ctrl[:] = HOME_Q
        m.geom_rgba[self._light_geom] = self._light_off
        mujoco.mj_forward(m, d)
        return np.concatenate(ctx)

    def run(self, theta: np.ndarray, record: bool = False, record_camera: str = "overview",
            record_size=(240, 320), record_every: int = 10, context: np.ndarray | None = None) -> Episode | tuple:
        cfg, m, d = self.cfg, self.model, self.data
        context = self.reset(context)
        touch_max = {"hand": 0.0, "arm": 0.0}
        obs0 = self._sense(touch_max, 0.0, False)
        targets = self.theta_to_targets(theta)
        n_hold = int(round(cfg.hold / (cfg.timestep * cfg.frame_skip)))
        n_settle = int(round(cfg.settle / (cfg.timestep * cfg.frame_skip)))
        plan = [targets[i] for i in range(N_VIA) for _ in range(n_hold)] + [targets[-1]] * n_settle
        start = {o: self._obj_xy(o) for o in OBJECTS}
        acc = np.zeros(m.nsensordata)   # running max of |sensor| over the episode
        light_on = False
        frames = []
        recorder = None
        if record:
            recorder = mujoco.Renderer(m, height=record_size[0], width=record_size[1])
        for k, tgt in enumerate(plan):
            d.ctrl[:] = tgt
            mujoco.mj_step(m, d, nstep=cfg.frame_skip)
            np.maximum(acc, np.abs(d.sensordata), out=acc)
            if not light_on and acc[1] > LIGHT_THRESHOLD:
                light_on = True
                m.geom_rgba[self._light_geom] = self._light_on
            if record and k % record_every == 0:
                recorder.update_scene(d, camera=record_camera)
                frames.append(recorder.render().copy())
        lever_max = float(acc[1])
        touched = [n for n, v in zip(("light_box", "ball", "heavy_box", "toy"), acc[2:6]) if v > 0]
        touch_max = {"hand": float(acc[0]), "arm": float(len(touched) > 0)}
        obs1 = self._sense(touch_max, lever_max, light_on)
        end = {o: self._obj_xy(o) for o in OBJECTS}
        disp = {o: float(np.linalg.norm(end[o] - start[o])) for o in OBJECTS}
        hand = d.site_xpos[self._hand_site][:2].copy()
        outcome = np.concatenate([hand] + [end[o] for o in OBJECTS] + [[lever_max]])
        ev = {
            "touch_light_box": "light_box" in touched,
            "touch_ball": "ball" in touched,
            "touch_heavy_box": "heavy_box" in touched,
            "touch_toy": "toy" in touched,
            "move_light_box": disp["light_box"] > 0.02,
            "move_ball": disp["ball"] > 0.02,
            "move_heavy_box": disp["heavy_box"] > 0.02,
            "push_light_box_far": disp["light_box"] > 0.10,
            "roll_ball_far": disp["ball"] > 0.10,
            "push_heavy_box_far": disp["heavy_box"] > 0.05,
            "toy_light_on": light_on,
            "object_hits_object": bool(acc[6:9].max() > 0),
            "object_hits_wall": bool(acc[9:21].max() > 0),
            "push_heavy_box_15cm": disp["heavy_box"] > 0.15,
            "toy_full_swing": lever_max > 1.0,
            "light_box_to_back": end["light_box"][1] > 0.40,
            "light_box_to_right_half": end["light_box"][0] > 0.10,
        }
        events = np.array([ev[n] for n in EVENT_NAMES], dtype=bool)
        epi = Episode(theta=np.asarray(theta, dtype=np.float32), context=context.astype(np.float32),
                      outcome=outcome.astype(np.float32), obs0=obs0, obs1=obs1, events=events,
                      contacts={"touched": touched})
        if record:
            recorder.close()
            return epi, frames
        return epi

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def set_headless_gl():
    """MuJoCo offscreen rendering backend: CGL on macOS, EGL elsewhere."""
    if "MUJOCO_GL" not in os.environ:
        import sys
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"
