from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

_DIRS8     = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_STEP_MULT = [math.sqrt(2), 1.0, math.sqrt(2), 1.0,
              1.0, math.sqrt(2), 1.0, math.sqrt(2)]


def astar(cost:     np.ndarray,
          start:    Tuple[int, int],
          goal:     Tuple[int, int],
          obs_cost: float,
          weight:   float = 1.0) -> Optional[List[Tuple[int, int]]]:
    """
    Weighted A* on a float32 costmap with closed set.

    Returns list of (row, col) from start to goal inclusive, or None.
    weight > 1.0 makes the search goal-directed, suppressing detours
    at the cost of path optimality.
    """
    H, W = cost.shape
    sr, sc = start
    gr, gc = goal

    if cost[sr, sc] >= obs_cost or cost[gr, gc] >= obs_cost:
        return None

    rr, cc    = np.mgrid[0:H, 0:W]
    h_map     = np.sqrt(((rr - gr)**2 + (cc - gc)**2).astype(np.float32)).ravel()
    cost_flat = cost.ravel()

    g_arr  = np.full(H * W, np.inf, dtype=np.float32)
    prev   = np.full(H * W, -1,     dtype=np.int32)
    closed = np.zeros(H * W,        dtype=bool)

    start_idx = sr * W + sc
    goal_idx  = gr * W + gc
    g_arr[start_idx] = 0.0
    heap = [(weight * float(h_map[start_idx]), 0.0, start_idx)]

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
                if p == -1:
                    return None
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
            if nc_cost >= obs_cost:
                continue
            ng = g_now + nc_cost * mult
            if ng < g_arr[nidx]:
                g_arr[nidx] = ng
                prev[nidx]  = idx
                heapq.heappush(heap, (ng + weight * float(h_map[nidx]), ng, nidx))

    return None
