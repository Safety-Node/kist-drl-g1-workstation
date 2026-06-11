from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np

_DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def inflate_costmap(grid,
                    base_cost:  float,
                    obs_cost:   float,
                    decay_rate: float) -> np.ndarray:
    """
    OccupancyGrid → BFS inflation → float32 traversal cost array.

    A free cell at BFS distance d (1-indexed) from the nearest obstacle
    is assigned total cost:
        cell_cost = obs_cost × (1 - decay_rate)^d
    Propagation stops when cell_cost falls to base_cost or below.
    """
    info = grid.info
    h, w = int(info.height), int(info.width)
    obs  = np.frombuffer(bytes(grid.data), dtype=np.int8).reshape(h, w)

    cost = np.full((h, w), float(base_cost), dtype=np.float32)
    cost[obs == 100] = obs_cost

    if decay_rate <= 0 or decay_rate >= 1:
        return cost

    factor  = 1.0 - decay_rate
    visited = (obs == 100).copy()
    queue   = deque()
    for r, c in zip(*np.where(obs == 100)):
        queue.append((int(r), int(c), obs_cost * factor))

    while queue:
        r, c, cell_cost = queue.popleft()
        if cell_cost <= base_cost:
            continue
        for dr, dc in _DIRS8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if visited[nr, nc]:
                continue
            visited[nr, nc] = True
            if obs[nr, nc] != 100:
                cost[nr, nc] = cell_cost
            queue.append((nr, nc, cell_cost * factor))

    return cost


def m2c(grid, x: float, y: float) -> Tuple[int, int]:
    """Meter (x,y) → (row, col) clipped to grid bounds."""
    info = grid.info
    c = max(0, min(int(info.width)  - 1, int((x - info.origin.position.x) / info.resolution)))
    r = max(0, min(int(info.height) - 1, int((y - info.origin.position.y) / info.resolution)))
    return r, c


def c2m(grid, r: int, c: int) -> Tuple[float, float]:
    """(row, col) → meter (x,y) cell centre."""
    info = grid.info
    x = info.origin.position.x + (c + 0.5) * info.resolution
    y = info.origin.position.y + (r + 0.5) * info.resolution
    return x, y
