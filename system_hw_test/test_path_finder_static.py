"""
path_finder + costmap 단독 동작 검증.

KIST L8 lab-like 환경에서 inflate_costmap → astar 순으로 실행하고
결과를 PNG로 저장합니다.

Usage:
    uv run system_hw_test/test_path_finder.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from providers.utils.route_utils import astar, c2m, inflate_costmap, m2c

# ── Load config ───────────────────────────────────────────────────────────────
_cfg_path = _ROOT / "src" / "providers" / "config" / "navigation" / "config.yaml"
_cfg      = yaml.safe_load(_cfg_path.read_text(encoding="utf-8"))

BASE_COST  = float(_cfg["costmap"]["base_cost"])
OBS_COST   = float(_cfg["costmap"]["obs_cost"])
DECAY_RATE = float(_cfg["costmap"]["decay_rate"])
WEIGHT     = float(_cfg["path"]["astar_weight"])

# ── Map parameters ────────────────────────────────────────────────────────────
X_MIN, X_MAX = 0.0, 8.18
Y_MIN, Y_MAX = 0.0, 6.69
RES  = 0.05
NX   = int((X_MAX - X_MIN) / RES)
NY   = int((Y_MAX - Y_MIN) / RES)

# ── Routes ────────────────────────────────────────────────────────────────────
ROUTES = [
    ("robot -> fridge", (0.5, 3.0), (6.3, 1.2)),
    ("robot -> table",  (0.5, 3.0), (4.5, 2.3)),
]

COLORS = ["royalblue", "darkorange"]


# ── Mock OccupancyGrid (nav_msgs/msg/OccupancyGrid 동일 interface) ─────────────

@dataclass
class _Point:
    x: float = 0.0
    y: float = 0.0

@dataclass
class _Pose:
    position: _Point = field(default_factory=_Point)

@dataclass
class _MapMeta:
    resolution: float = RES
    width:      int   = NX
    height:     int   = NY
    origin:     _Pose = field(default_factory=_Pose)

@dataclass
class OccupancyGrid:
    info: _MapMeta = field(default_factory=_MapMeta)
    data: bytes    = b""


def _fill(arr: np.ndarray, x0, y0, x1, y1):
    c0 = int((x0 - X_MIN) / RES)
    c1 = int((x1 - X_MIN) / RES)
    r0 = int((y0 - Y_MIN) / RES)
    r1 = int((y1 - Y_MIN) / RES)
    arr[r0:r1, c0:c1] = 100


def make_grid() -> OccupancyGrid:
    arr = np.zeros((NY, NX), dtype=np.int8)
    arr[0, :] = arr[-1, :] = arr[:, 0] = arr[:, -1] = 100
    _fill(arr, 0.5, 5.5, 6.0, 6.5)   # worktable
    _fill(arr, 3.0, 2.8, 6.5, 4.5)   # center table
    _fill(arr, 1.2, 1.2, 2.0, 2.0)   # pillar
    _fill(arr, 6.8, 0.5, 7.9, 2.0)   # cabinet/fridge
    return OccupancyGrid(
        info=_MapMeta(origin=_Pose(position=_Point(x=X_MIN, y=Y_MIN))),
        data=arr.tobytes(),
    )


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(grid: OccupancyGrid, costmap: np.ndarray, results):
    info = grid.info
    obs  = np.frombuffer(grid.data, dtype=np.int8).reshape(info.height, info.width)
    ext  = [X_MIN, X_MAX, Y_MIN, Y_MAX]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left — obstacle map + paths
    ax1.set_title("Obstacle Map  +  A* Path")
    rgb = np.ones((info.height, info.width, 3), dtype=np.float32)
    rgb[obs == 100] = [0.25, 0.25, 0.25]
    ax1.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")

    for (name, start_m, goal_m, path), col in zip(results, COLORS):
        ax1.plot(*start_m, "o", color=col, markersize=10, zorder=5)
        ax1.plot(*goal_m,  "*", color=col, markersize=14, zorder=5)
        if path:
            xs, ys = zip(*[c2m(grid, r, c) for r, c in path])
            ax1.plot(xs, ys, "-", color=col, lw=2, label=name, zorder=4)
        else:
            ax1.plot([], [], "-", color=col, lw=2, label=f"{name} (no path)")

    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

    # Right — inflated costmap + paths
    ax2.set_title("Inflated Costmap  +  A* Path")
    disp = np.log1p(np.clip(costmap, 0, OBS_COST - 1)).astype(np.float32)
    disp[obs == 100] = np.nan
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("black")
    im = ax2.imshow(disp, origin="lower", extent=ext,
                    cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax2, label="log(cost+1)")

    for (name, start_m, goal_m, path), col in zip(results, COLORS):
        ax2.plot(*start_m, "o", color=col, markersize=10, zorder=5)
        ax2.plot(*goal_m,  "*", color=col, markersize=14, zorder=5)
        if path:
            xs, ys = zip(*[c2m(grid, r, c) for r, c in path])
            ax2.plot(xs, ys, "-", color=col, lw=2, zorder=4)

    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = "path_finder_test.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    grid    = make_grid()
    costmap = inflate_costmap(grid, BASE_COST, OBS_COST, DECAY_RATE)

    print(f"Map: {NX} x {NY}  res={RES}m  decay_rate={DECAY_RATE}  weight={WEIGHT}")

    results = []
    for name, start_m, goal_m in ROUTES:
        sc_ = m2c(grid, *start_m)
        gc_ = m2c(grid, *goal_m)
        path = astar(costmap, sc_, gc_, OBS_COST, WEIGHT)
        if path:
            print(f"  {name}: {len(path)} cells ({len(path) * RES:.2f} m)")
        else:
            print(f"  {name}: no path found")
        results.append((name, start_m, goal_m, path))

    visualize(grid, costmap, results)
