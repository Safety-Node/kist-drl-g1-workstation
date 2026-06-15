"""
Route planner simulation — A* on BFS-inflated costmap.

Standalone (no ROS). KIST L8 lab-like environment (8.18 x 6.69 m, 0.05 m res).

OccupancyGrid 입력은 nav_msgs/msg/OccupancyGrid 와 동일한 구조의 mock을 사용.
NavigationProvider 가 받는 실제 msg 와 attribute 이름이 동일하므로
route planner 로직을 그대로 이식 가능.

Usage:
    uv run system_hw_test/test_route_planner.py
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')   # headless — save PNG without needing a display
import matplotlib.pyplot as plt
import numpy as np


# ── nav_msgs/msg/OccupancyGrid 구조 모사 ─────────────────────────────────────
# (ROS 없이도 동일한 .attribute 접근으로 route planner 로직 검증)

@dataclass
class _Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class _Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

@dataclass
class _Pose:
    position: _Point    = field(default_factory=_Point)
    orientation: _Quaternion = field(default_factory=_Quaternion)

@dataclass
class MapMetaData:
    resolution: float = 0.05
    width:      int   = 0
    height:     int   = 0
    origin:     _Pose = field(default_factory=_Pose)

@dataclass
class OccupancyGrid:
    """nav_msgs/msg/OccupancyGrid 와 동일한 interface."""
    info: MapMetaData = field(default_factory=MapMetaData)
    data: bytes       = b''   # int8 row-major (0=free, 100=obstacle, -1=unknown)


# ── Map parameters ───────────────────────────────────────────────────────────
X_MIN, X_MAX = 0.0, 8.18
Y_MIN, Y_MAX = 0.0, 6.69
RES = 0.05
NX = int((X_MAX - X_MIN) / RES)   # ~163
NY = int((Y_MAX - Y_MIN) / RES)   # ~133

# ── Costmap parameters ───────────────────────────────────────────────────────
BASE_COST = 1
INFLATION = {1: 3, 2: 2, 3: 1}
OBS_COST  = 9999


# ── Simulated OccupancyGrid builder ──────────────────────────────────────────

def _fill(arr: np.ndarray, x0, y0, x1, y1):
    c0, c1 = int((x0 - X_MIN) / RES), int((x1 - X_MIN) / RES)
    r0, r1 = int((y0 - Y_MIN) / RES), int((y1 - Y_MIN) / RES)
    arr[r0:r1, c0:c1] = 100


def make_sim_occupancy_grid() -> OccupancyGrid:
    """
    KIST L8 lab-like layout → nav_msgs/OccupancyGrid 형식으로 반환.

    +-----------------------------------------+ y=6.69
    | [worktable                             ] |
    |                                          |
    |               [center table]             |
    |                                          |
    | [pillar]               [cabinet/fridge]  |
    +-----------------------------------------+ y=0.0
    x=0.0                                  x=8.18
    """
    arr = np.zeros((NY, NX), dtype=np.int8)

    # Boundary walls
    arr[0, :]  = 100
    arr[-1, :] = 100
    arr[:, 0]  = 100
    arr[:, -1] = 100

    # Worktable along top wall
    _fill(arr, 0.5, 5.5, 6.0, 6.5)
    # Center table
    _fill(arr, 3.0, 2.8, 6.5, 4.5)
    # Pillar (lower-left)
    _fill(arr, 1.2, 1.2, 2.0, 2.0)
    # Cabinet / fridge (lower-right)
    _fill(arr, 6.8, 0.5, 7.9, 2.0)

    meta = MapMetaData(
        resolution=RES,
        width=NX,
        height=NY,
        origin=_Pose(position=_Point(x=X_MIN, y=Y_MIN)),
    )
    # data: row-major int8 bytes (same as real OccupancyGrid.data)
    return OccupancyGrid(info=meta, data=arr.tobytes())


# ── BFS costmap inflation ─────────────────────────────────────────────────────
# NavigationProvider._compute_repulsion 과 동일한 방식으로 data 디코딩

def inflate_costmap(grid: OccupancyGrid) -> np.ndarray:
    """
    nav_msgs/OccupancyGrid → BFS 3-layer inflation → traversal cost map.

    grid.data 디코딩 방식은 NavigationProvider._compute_repulsion 과 동일.
    """
    info = grid.info
    h, w = info.height, info.width
    obs = np.frombuffer(grid.data, dtype=np.int8).reshape(h, w)

    cost = np.full((h, w), float(BASE_COST), dtype=np.float32)
    cost[obs == 100] = OBS_COST

    visited = (obs == 100).copy()
    queue   = deque()
    for r, c in zip(*np.where(obs == 100)):
        queue.append((int(r), int(c), 0))

    dirs8   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    max_d   = max(INFLATION.keys())

    while queue:
        r, c, d = queue.popleft()
        if d >= max_d:
            continue
        for dr, dc in dirs8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if visited[nr, nc]:
                continue
            visited[nr, nc] = True
            if obs[nr, nc] != 100:
                cost[nr, nc] += INFLATION[d + 1]
            queue.append((nr, nc, d + 1))

    return cost


# ── A* path planner ───────────────────────────────────────────────────────────

_DIRS8     = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_STEP_MULT = [math.sqrt(2), 1.0, math.sqrt(2), 1.0,
              1.0, math.sqrt(2), 1.0, math.sqrt(2)]


def astar(cost: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) \
        -> Optional[List[Tuple[int,int]]]:
    """
    Numpy-optimized A* with closed set.

    Each cell is expanded at most once (consistent euclidean heuristic),
    keeping heap size O(n) instead of blowing up with stale duplicates.
    """
    H, W = cost.shape
    sr, sc = start
    gr, gc = goal

    if cost[sr, sc] >= OBS_COST:
        print("  [!] Start cell is obstacle"); return None
    if cost[gr, gc] >= OBS_COST:
        print("  [!] Goal cell is obstacle");  return None

    # Precompute entire heuristic in one vectorized call
    rr, cc = np.mgrid[0:H, 0:W]
    h_map = np.sqrt(((rr - gr)**2 + (cc - gc)**2).astype(np.float32)).ravel()

    cost_flat = cost.ravel()
    g_arr  = np.full(H * W, np.inf, dtype=np.float32)
    prev   = np.full(H * W, -1,     dtype=np.int32)
    closed = np.zeros(H * W,        dtype=bool)

    start_idx = sr * W + sc
    goal_idx  = gr * W + gc
    g_arr[start_idx] = 0.0

    heap = [(float(h_map[start_idx]), 0.0, start_idx)]

    while heap:
        _, g_now, idx = heapq.heappop(heap)

        if closed[idx]:
            continue
        closed[idx] = True

        if idx == goal_idx:
            path, cur = [], goal_idx
            while cur != start_idx:
                path.append((cur // W, cur % W))
                p = int(prev[cur])
                if p == -1: return None
                cur = p
            path.append((sr, sc))
            path.reverse()
            return path

        r, c = idx // W, idx % W

        for (dr, dc), mult in zip(_DIRS8, _STEP_MULT):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            nidx = nr * W + nc
            if closed[nidx]:
                continue
            nc_cost = cost_flat[nidx]
            if nc_cost >= OBS_COST:
                continue
            ng = g_now + nc_cost * mult
            if ng < g_arr[nidx]:
                g_arr[nidx] = ng
                prev[nidx]  = idx
                heapq.heappush(heap, (ng + float(h_map[nidx]), ng, nidx))

    return None


# ── Coordinate helpers ────────────────────────────────────────────────────────

def m2c(grid: OccupancyGrid, x: float, y: float) -> Tuple[int, int]:
    """Meter (x,y) -> (row, col)."""
    info = grid.info
    c = max(0, min(info.width  - 1, int((x - info.origin.position.x) / info.resolution)))
    r = max(0, min(info.height - 1, int((y - info.origin.position.y) / info.resolution)))
    return r, c


def c2m(grid: OccupancyGrid, r: int, c: int) -> Tuple[float, float]:
    """(row, col) -> meter (x,y) cell center."""
    info = grid.info
    x = info.origin.position.x + (c + 0.5) * info.resolution
    y = info.origin.position.y + (r + 0.5) * info.resolution
    return x, y


# ── Visualization ─────────────────────────────────────────────────────────────

COLORS = ['royalblue', 'darkorange', 'limegreen', 'purple']


def visualize(grid: OccupancyGrid, cost: np.ndarray, results):
    info = grid.info
    obs  = np.frombuffer(grid.data, dtype=np.int8).reshape(info.height, info.width)
    ext  = [info.origin.position.x,
            info.origin.position.x + info.width  * info.resolution,
            info.origin.position.y,
            info.origin.position.y + info.height * info.resolution]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: obstacle map + paths
    ax1.set_title("Obstacle Map  +  A* Path")
    rgb = np.ones((info.height, info.width, 3), dtype=np.float32)
    rgb[obs == 100] = [0.25, 0.25, 0.25]
    ax1.imshow(rgb, origin='lower', extent=ext, interpolation='nearest')

    for i, (name, start_m, goal_m, path) in enumerate(results):
        col = COLORS[i % len(COLORS)]
        ax1.plot(*start_m, 'o', color=col, markersize=10, zorder=5)
        ax1.plot(*goal_m,  '*', color=col, markersize=14, zorder=5)
        if path:
            xs, ys = zip(*[c2m(grid, r, c) for r, c in path])
            ax1.plot(xs, ys, '-', color=col, lw=2, label=name, zorder=4)
        else:
            ax1.plot([], [], '-', color=col, lw=2, label=f"{name} (no path)")

    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)

    # Right: costmap + paths
    ax2.set_title("Inflated Costmap  +  A* Path")
    disp = np.clip(cost, 0, 10).astype(np.float32)
    disp[obs == 100] = np.nan
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('black')
    im = ax2.imshow(disp, origin='lower', extent=ext,
                    cmap=cmap, vmin=0, vmax=10, interpolation='nearest')
    plt.colorbar(im, ax=ax2, label='traversal cost')

    for i, (name, start_m, goal_m, path) in enumerate(results):
        col = COLORS[i % len(COLORS)]
        ax2.plot(*start_m, 'o', color=col, markersize=10, zorder=5)
        ax2.plot(*goal_m,  '*', color=col, markersize=14, zorder=5)
        if path:
            xs, ys = zip(*[c2m(grid, r, c) for r, c in path])
            ax2.plot(xs, ys, '-', color=col, lw=2, zorder=4)

    ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = 'route_planner_test.png'
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

ROUTES = [
    ("robot -> fridge", (0.5, 3.0), (6.3, 1.2)),   # cabinet 왼쪽 앞
    ("robot -> table",  (0.5, 3.0), (4.5, 2.3)),   # center table 아래 앞
]

if __name__ == '__main__':
    grid = make_sim_occupancy_grid()
    info = grid.info
    print(f"OccupancyGrid: {info.width} x {info.height} = {info.width*info.height:,} cells"
          f"  res={info.resolution}m"
          f"  origin=({info.origin.position.x}, {info.origin.position.y})")

    cost = inflate_costmap(grid)

    results = []
    for name, start_m, goal_m in ROUTES:
        sc = m2c(grid, *start_m)
        gc = m2c(grid, *goal_m)
        print(f"\n{name}")
        print(f"  start {start_m} -> cell {sc}")
        print(f"  goal  {goal_m} -> cell {gc}")
        path = astar(cost, sc, gc)
        if path:
            print(f"  path: {len(path)} cells  ({len(path) * info.resolution:.2f} m)")
        else:
            print("  no path found")
        results.append((name, start_m, goal_m, path))

    visualize(grid, cost, results)
