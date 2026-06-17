"""
route plan 단독 벤치마크 — 더미 맵으로 inflate_costmap + astar 시간 측정.

Usage:
    uv run system_hw_test/bench_route_plan.py [--iters N] [--size WxH]

    --iters : 반복 횟수 (기본 200)
    --size  : 맵 크기 (기본 164x134 ≈ KIST L8)  예) --size 300x250
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from providers.utils.route_utils import astar, inflate_costmap

# ── Config ────────────────────────────────────────────────────────────────────
_cfg_path = _ROOT / "src" / "providers" / "config" / "navigation" / "config.yaml"
_cfg      = yaml.safe_load(_cfg_path.read_text(encoding="utf-8"))

BASE_COST  = float(_cfg["costmap"]["base_cost"])
OBS_COST   = float(_cfg["costmap"]["obs_cost"])
DECAY_RATE = float(_cfg["costmap"]["decay_rate"])
WEIGHT     = float(_cfg["path"]["astar_weight"])
RES        = 0.05


# ── 더미 OccupancyGrid 생성 ───────────────────────────────────────────────────
def _make_mock_grid(w: int, h: int, obs_ratio: float = 0.08):
    """랜덤 장애물이 있는 더미 OccupancyGrid-like 객체."""
    rng  = np.random.default_rng(42)
    data = np.zeros(h * w, dtype=np.int8)
    obs_idx = rng.choice(h * w, size=int(h * w * obs_ratio), replace=False)
    data[obs_idx] = 100

    class _Info:
        pass

    info = _Info()
    info.height     = h
    info.width      = w
    info.resolution = RES
    info.origin     = _Info()
    info.origin.position = _Info()
    info.origin.position.x = 0.0
    info.origin.position.y = 0.0

    grid = _Info()
    grid.info = info
    grid.data = data.tolist()
    return grid, w, h


# ── 벤치마크 ──────────────────────────────────────────────────────────────────
def bench(iters: int, w: int, h: int) -> None:
    grid, w, h = _make_mock_grid(w, h)

    # 시작: 좌하단 근처  /  목표: 우상단 근처
    start = (int(h * 0.1), int(w * 0.1))
    goal  = (int(h * 0.9), int(w * 0.9))

    inflate_times: list[float] = []
    astar_times:   list[float] = []
    total_times:   list[float] = []

    print(f"맵 크기 : {w}×{h} = {w*h:,} cells")
    print(f"반복 수 : {iters}")
    print(f"start   : {start}  goal : {goal}")
    print("-" * 60)
    print(f"{'iter':>6}  {'inflate_ms':>10}  {'astar_ms':>9}  {'total_ms':>9}")
    print("-" * 60)

    for i in range(iters):
        t0      = time.perf_counter()
        costmap = inflate_costmap(grid, BASE_COST, OBS_COST, DECAY_RATE)
        t1      = time.perf_counter()
        path    = astar(costmap, start, goal, OBS_COST, WEIGHT)
        t2      = time.perf_counter()

        inf_ms   = (t1 - t0) * 1e3
        ast_ms   = (t2 - t1) * 1e3
        tot_ms   = (t2 - t0) * 1e3

        inflate_times.append(inf_ms)
        astar_times.append(ast_ms)
        total_times.append(tot_ms)

        if i < 5 or i % 50 == 49:
            cells = len(path) if path else 0
            print(f"{i:>6}  {inf_ms:>10.2f}  {ast_ms:>9.2f}  {tot_ms:>9.2f}"
                  f"  path={cells} cells")

    # ── 통계 출력 ─────────────────────────────────────────────────────────────
    def _stats(name, arr):
        a = np.array(arr)
        print(f"  {name:<12}: avg={a.mean():.2f}  "
              f"min={a.min():.2f}  max={a.max():.2f}  "
              f"p50={np.percentile(a,50):.2f}  "
              f"p95={np.percentile(a,95):.2f}  ms")

    print("=" * 60)
    print("결과 (ms)")
    _stats("inflate", inflate_times)
    _stats("astar",   astar_times)
    _stats("total",   total_times)

    avg_total = np.mean(total_times)
    achievable_hz = 1000.0 / avg_total
    print(f"\n  달성 가능 Hz (total avg 기준): {achievable_hz:.1f} Hz")
    if achievable_hz >= 20:
        print("  → 20 Hz 달성 가능")
    elif achievable_hz >= 10:
        print("  → 10 Hz 달성 가능 (20 Hz 불가)")
    else:
        print("  → 10 Hz 미만 — 최적화 필요")


# ── 진입점 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--size",  type=str, default="164x134",
                        help="WxH (기본: 164x134 ≈ KIST L8)")
    args = parser.parse_args()

    w, h = (int(x) for x in args.size.split("x"))
    bench(args.iters, w, h)
