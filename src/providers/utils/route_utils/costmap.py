from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_cdt


def inflate_costmap(grid,
                    base_cost:  float,
                    obs_cost:   float,
                    decay_rate: float) -> np.ndarray:
    """
    OccupancyGrid → scipy distance transform → float32 traversal cost array.

    Chessboard (8-connected) distance d from nearest obstacle:
        cell_cost = max(base_cost, obs_cost × (1 - decay_rate)^d)
    Obstacle cells keep obs_cost (d=0 → factor^0 = 1).
    """
    info = grid.info
    h, w = int(info.height), int(info.width)
    obs  = np.frombuffer(bytes(grid.data), dtype=np.int8).reshape(h, w)

    obs_mask = obs == 100

    if decay_rate <= 0.0 or decay_rate >= 1.0:
        cost = np.full((h, w), float(base_cost), dtype=np.float32)
        cost[obs_mask] = float(obs_cost)
        return cost

    dist = distance_transform_cdt(~obs_mask, metric='chessboard').astype(np.float32)
    cost = np.maximum(
        float(base_cost),
        float(obs_cost) * ((1.0 - float(decay_rate)) ** dist),
    ).astype(np.float32)
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
