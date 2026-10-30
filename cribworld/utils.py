from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def free_memory_pct() -> float | None:
    """macOS `memory_pressure` free percentage; None elsewhere."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else None


def wait_for_memory(min_pct: float, poll: float = 60.0) -> None:
    """Block until at least ``min_pct`` percent of RAM is free (macOS only)."""
    while True:
        pct = free_memory_pct()
        if pct is None or pct >= min_pct:
            return
        print(f"  waiting: only {pct:.0f}% memory free (< {min_pct}%)", flush=True)
        time.sleep(poll)


def run_path(out_dir: str | Path, perception: str, agent: str, seed: int) -> Path:
    return Path(out_dir) / "freeplay" / f"{perception}_{agent}_s{seed}.npz"
