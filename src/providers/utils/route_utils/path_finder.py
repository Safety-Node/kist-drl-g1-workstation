from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from skimage.graph import route_through_array


def astar(cost:     np.ndarray,
          start:    Tuple[int, int],
          goal:     Tuple[int, int],
          obs_cost: float,
          weight:   float = 1.0) -> Optional[List[Tuple[int, int]]]:
    """
    Optimal path on a float32 costmap via skimage MCP (C extension).

    Uses 8-connected movement with geometric (diagonal = sqrt(2)) step cost.
    Obstacle cells (>= obs_cost) are set to inf before search.
    `weight` is kept for API compatibility but ignored (MCP is always optimal).
    Returns list of (row, col) from start to goal inclusive, or None.
    """
    sr, sc = start
    gr, gc = goal

    if cost[sr, sc] >= obs_cost or cost[gr, gc] >= obs_cost:
        return None

    traversal = cost.astype(np.float64)
    traversal[cost >= obs_cost] = np.inf

    try:
        path, _ = route_through_array(
            traversal, start, goal,
            fully_connected=True,
            geometric=True,
        )
        return [tuple(p) for p in path]
    except Exception:
        return None
