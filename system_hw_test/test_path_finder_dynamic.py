"""
Path finder 동적 검증 — UnitreeG1Provider 실시간 데이터 사용, 20Hz 연속 루프.

실제 로봇(또는 브리지)으로부터:
  - 장애물 맵  : /bridge/sensors/lidar/occupancy  (OccupancyGrid)
  - 현재 위치  : /bridge/sensors/location          (PoseStamped)
  - 목적지     : providers/config/navigation/locations.json5

위 데이터로 inflate_costmap → astar 를 20Hz 로 반복 실행하며 타이밍을 출력합니다.
PNG 는 --save-interval 초마다 저장됩니다 (기본 2.0s).

Usage:
    uv run system_hw_test/test_path_finder_dynamic.py <destination> [--save-interval SECS]

    destination  : locations.json5 에 등록된 키 (기본값: fridge)
    --save-interval : PNG 저장 주기 (초, 기본 2.0)

    예) uv run system_hw_test/test_path_finder_dynamic.py table --save-interval 3
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json5
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from providers.unitree_g1_provider import UnitreeG1Provider
from providers.utils.route_utils import astar, c2m, inflate_costmap, m2c

# ── Config 로드 ───────────────────────────────────────────────────────────────
_cfg_path = _ROOT / "src" / "providers" / "config" / "navigation" / "config.yaml"
_cfg      = yaml.safe_load(_cfg_path.read_text(encoding="utf-8"))

BASE_COST       = float(_cfg["costmap"]["base_cost"])
OBS_COST        = float(_cfg["costmap"]["obs_cost"])
DECAY_RATE      = float(_cfg["costmap"]["decay_rate"])
WEIGHT          = float(_cfg["path"]["astar_weight"])
PLANNER_RATE_HZ = float(_cfg["path"]["planner_rate_hz"])

_locs_path = _ROOT / _cfg.get("locations_file",
                               "src/providers/config/navigation/locations.json5")
LOCATIONS  = {k: tuple(v) for k, v in
              json5.loads(_locs_path.read_text(encoding="utf-8")).items()}

COLORS = ["royalblue", "darkorange", "limegreen", "purple"]

# ── 데이터 대기 ───────────────────────────────────────────────────────────────
WAIT_TIMEOUT_S = 10.0
STALE_TTL_S    = 1.0


def wait_for_data(g1: UnitreeG1Provider, timeout: float = WAIT_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        now = time.monotonic()
        occ_ok = not g1.occupancy.stale(now, STALE_TTL_S)
        loc_ok = not g1.location.stale(now, STALE_TTL_S)
        if occ_ok and loc_ok:
            return True
        missing = []
        if not occ_ok: missing.append("occupancy")
        if not loc_ok: missing.append("location")
        print(f"  대기 중 ... ({', '.join(missing)} 미수신)")
        time.sleep(0.5)
    return False


# ── 시각화 ────────────────────────────────────────────────────────────────────

def save_png(grid, costmap: np.ndarray, robot_pos, results, out: str) -> None:
    info = grid.info
    obs  = np.frombuffer(bytes(grid.data), dtype=np.int8).reshape(
        int(info.height), int(info.width))
    ext  = [
        info.origin.position.x,
        info.origin.position.x + int(info.width)  * info.resolution,
        info.origin.position.y,
        info.origin.position.y + int(info.height) * info.resolution,
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.set_title("Obstacle Map  +  A* Path  [dynamic]")
    rgb = np.ones((int(info.height), int(info.width), 3), dtype=np.float32)
    rgb[obs == 100] = [0.25, 0.25, 0.25]
    ax1.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
    ax1.plot(*robot_pos, "g^", markersize=12, zorder=6, label="robot")

    for (name, goal_m, path), col in zip(results, COLORS):
        ax1.plot(*goal_m, "*", color=col, markersize=14, zorder=5)
        if path:
            xs, ys = zip(*[c2m(grid, r, c) for r, c in path])
            ax1.plot(xs, ys, "-", color=col, lw=2, label=name, zorder=4)
        else:
            ax1.plot([], [], "-", color=col, lw=2, label=f"{name} (no path)")

    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

    ax2.set_title("Inflated Costmap  +  A* Path  [dynamic]")
    disp = np.log1p(np.clip(costmap, 0, OBS_COST - 1)).astype(np.float32)
    disp[obs == 100] = np.nan
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("black")
    im = ax2.imshow(disp, origin="lower", extent=ext,
                    cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax2, label="log(cost+1)")
    ax2.plot(*robot_pos, "g^", markersize=12, zorder=6)

    for (name, goal_m, path), col in zip(results, COLORS):
        ax2.plot(*goal_m, "*", color=col, markersize=14, zorder=5)
        if path:
            xs, ys = zip(*[c2m(grid, r, c) for r, c in path])
            ax2.plot(xs, ys, "-", color=col, lw=2, zorder=4)

    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [png] Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="20Hz 연속 경로 탐색 검증 (Ctrl+C 로 종료)")
    parser.add_argument("destination", nargs="?", default="fridge",
                        help="목적지 키 (locations.json5 등록 키, 기본: fridge)")
    parser.add_argument("--save-interval", type=float, default=2.0,
                        metavar="SECS",
                        help="PNG 저장 주기 (초, 기본: 2.0)")
    args = parser.parse_args()

    dest_key = args.destination
    if dest_key not in LOCATIONS:
        print(f"[!] '{dest_key}' not in locations.json5.")
        print(f"    등록된 목적지: {list(LOCATIONS)}")
        return 1

    gx, gy      = LOCATIONS[dest_key]
    period_s    = 1.0 / PLANNER_RATE_HZ
    save_ivl    = args.save_interval

    print("UnitreeG1Provider 시작 ...")
    g1 = UnitreeG1Provider()
    g1.start()

    # Ctrl+C 를 깨끗하게 처리
    stop = False
    def _on_sigint(sig, _):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        print(f"센서 데이터 대기 (최대 {WAIT_TIMEOUT_S}s) ...")
        if not wait_for_data(g1):
            print("[!] 타임아웃: occupancy 또는 location 데이터를 수신하지 못했습니다.")
            return 1

        print(f"\n목적지: {dest_key}  ({gx:.3f}, {gy:.3f})")
        print(f"루프 주파수 설정: {PLANNER_RATE_HZ:.0f} Hz  "
              f"(PNG 저장 주기: {save_ivl:.1f}s)\n")
        print(f"{'iter':>6}  {'plan_ms':>8}  {'inflate_ms':>10}  "
              f"{'astar_ms':>8}  {'hz_actual':>10}  path_cells")
        print("-" * 66)

        itr           = 0
        last_png_t    = time.monotonic()
        last_loop_t   = time.monotonic()

        # 통계 집계
        plan_times_ms: list[float] = []

        while not stop:
            loop_start = time.monotonic()

            # ── 신선도 확인 ──────────────────────────────────────────────────
            now = time.monotonic()
            if g1.occupancy.stale(now, STALE_TTL_S) or g1.location.stale(now, STALE_TTL_S):
                print(f"{'':>6}  [stale data — waiting]")
                time.sleep(period_s)
                last_loop_t = time.monotonic()
                continue

            # ── costmap inflate ──────────────────────────────────────────────
            grid  = g1.occupancy.value
            pose  = g1.location.value
            rx    = pose.pose.position.x
            ry    = pose.pose.position.y

            t0 = time.perf_counter()
            costmap = inflate_costmap(grid, BASE_COST, OBS_COST, DECAY_RATE)
            inflate_ms = (time.perf_counter() - t0) * 1e3

            # ── A* ──────────────────────────────────────────────────────────
            start_cell = m2c(grid, rx, ry)
            goal_cell  = m2c(grid, gx, gy)

            t1 = time.perf_counter()
            path = astar(costmap, start_cell, goal_cell, OBS_COST, WEIGHT)
            astar_ms = (time.perf_counter() - t1) * 1e3

            plan_ms      = inflate_ms + astar_ms
            plan_times_ms.append(plan_ms)

            # ── 실제 Hz 계산 ─────────────────────────────────────────────────
            elapsed_loop = loop_start - last_loop_t
            hz_actual    = 1.0 / elapsed_loop if elapsed_loop > 0 else float("inf")
            last_loop_t  = loop_start

            cells_str = str(len(path)) if path else "no path"
            print(f"{itr:>6}  {inflate_ms:>8.2f}  {inflate_ms:>10.2f}  "
                  f"{astar_ms:>8.2f}  {hz_actual:>10.1f}  {cells_str}")

            # ── 주기적 PNG 저장 ───────────────────────────────────────────────
            if loop_start - last_png_t >= save_ivl:
                save_png(grid, costmap, (rx, ry),
                         [(dest_key, (gx, gy), path)],
                         "path_finder_dynamic_test.png")
                last_png_t = loop_start

            itr += 1

            # ── 루프 주기 맞추기 ─────────────────────────────────────────────
            elapsed = time.monotonic() - loop_start
            sleep_t = period_s - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        g1.stop()

    # ── 통계 출력 ─────────────────────────────────────────────────────────────
    if plan_times_ms:
        arr = np.array(plan_times_ms)
        print(f"\n총 {len(arr)} 회  |  plan 시간: "
              f"avg={arr.mean():.2f}ms  "
              f"max={arr.max():.2f}ms  "
              f"p95={np.percentile(arr, 95):.2f}ms")
    print("종료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
