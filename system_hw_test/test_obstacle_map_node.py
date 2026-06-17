# test_obstacle_map_node.py
"""
obstacle_map_node — real-time occupancy grid + A* path visualization (TASK-53).

/bridge/sensors/lidar/occupancy  (nav_msgs/OccupancyGrid) : 2D obstacle map
/bridge/sensors/location         (geometry_msgs/PoseStamped) : robot pose

OccupancyGrid를 격자로 렌더링하고 로봇 현재 위치 + A* 경로를 오버레이한다.
FREE(0) → 흰색, OBSTACLE(100) → 빨간색, 로봇 → 파란 화살표.
경로 → 초록 선, 목적지 → 노란 별.

플래너는 별도 프로세스로 분리해 GIL 간섭 없이 동작.
시각화는 ~20Hz (50ms) 로 draw_idle/flush_events 루프.

Usage:
    python system_hw_test/test_obstacle_map_node.py [destination]

    destination : locations.json5 에 등록된 키 (기본값: fridge)
"""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import json5
import yaml

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
LOCATIONS = {k: tuple(v) for k, v in
             json5.loads(_locs_path.read_text(encoding="utf-8")).items()}

# ── 상수 ─────────────────────────────────────────────────────────────────────
_VIZ_HZ      = 20
_VIZ_PERIOD  = 1.0 / _VIZ_HZ
_ARROW_LEN_M = 0.3

_QOS_BE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


# ── shared state (메인 프로세스, ROS spin 스레드 ↔ viz 루프) ─────────────────
class _State:
    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self.grid: np.ndarray | None = None
        self.info      = None
        self.grid_arr: np.ndarray | None = None  # float32 for planner
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.resolution: float = 0.05
        self.width: int  = 0
        self.height: int = 0
        self.robot_x:   float | None = None
        self.robot_y:   float | None = None
        self.robot_yaw: float | None = None
        # planner 결과 (플래너 프로세스 → 메인)
        self.path:    list | None = None
        self.plan_ms: float = 0.0
        self.plan_hz: float = 0.0

    def push_grid(self, msg: OccupancyGrid) -> None:
        arr = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        with self._lock:
            self.grid       = arr
            self.info       = msg.info
            self.grid_arr   = arr
            self.origin_x   = msg.info.origin.position.x
            self.origin_y   = msg.info.origin.position.y
            self.resolution = msg.info.resolution
            self.width      = msg.info.width
            self.height     = msg.info.height

    def push_pose(self, x: float, y: float, yaw: float) -> None:
        with self._lock:
            self.robot_x   = x
            self.robot_y   = y
            self.robot_yaw = yaw

    def push_path(self, path, plan_ms: float, plan_hz: float) -> None:
        with self._lock:
            self.path     = path
            self.plan_ms  = plan_ms
            self.plan_hz  = plan_hz

    def planner_input(self):
        """플래너 프로세스로 보낼 직렬화 가능한 snapshot."""
        with self._lock:
            if self.grid_arr is None or self.robot_x is None:
                return None
            return (
                self.grid_arr.copy(),
                self.origin_x, self.origin_y,
                self.resolution, self.width, self.height,
                self.robot_x, self.robot_y,
            )

    def snapshot(self):
        with self._lock:
            return (
                self.grid, self.info,
                self.origin_x, self.origin_y, self.resolution,
                self.width, self.height,
                self.robot_x, self.robot_y, self.robot_yaw,
                self.path, self.plan_ms, self.plan_hz,
            )


# ── ROS2 구독 노드 ────────────────────────────────────────────────────────────
class _Subscriber(Node):
    def __init__(self, state: _State) -> None:
        super().__init__('test_obstacle_map_viz')
        self._state = state
        self.create_subscription(
            OccupancyGrid, '/bridge/sensors/lidar/occupancy',
            self._on_grid, _QOS_BE,
        )
        self.create_subscription(
            PoseStamped, '/bridge/sensors/location',
            self._on_location, _QOS_BE,
        )

    def _on_grid(self, msg: OccupancyGrid) -> None:
        self._state.push_grid(msg)

    def _on_location(self, msg: PoseStamped) -> None:
        self._state.push_pose(
            msg.pose.position.x,
            msg.pose.position.y,
            _quat_to_yaw(msg.pose.orientation),
        )


# ── 플래너 프로세스 (GIL 없음) ────────────────────────────────────────────────
def _planner_worker(in_q: mp.Queue, out_q: mp.Queue,
                    goal: tuple, cfg: dict) -> None:
    """별도 프로세스: inflate_costmap + astar 반복."""
    # 프로세스 내부에서 임포트
    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root / "src"))
    from providers.utils.route_utils import astar, inflate_costmap

    base_cost = cfg["base_cost"]
    obs_cost  = cfg["obs_cost"]
    decay     = cfg["decay_rate"]
    weight    = cfg["weight"]
    period    = 1.0 / cfg["rate_hz"]
    gx, gy    = goal

    last_t = time.monotonic()

    while True:
        item = in_q.get()
        if item is None:
            break

        (grid_arr, ox, oy, res, w, h, rx, ry) = item

        # m2c 인라인 (OccupancyGrid 없이)
        def _m2c(mx, my):
            c = int((mx - ox) / res)
            r = int((my - oy) / res)
            c = max(0, min(w - 1, c))
            r = max(0, min(h - 1, r))
            return (r, c)

        # inflate_costmap 용 mock grid
        class _G:
            pass
        g = _G()
        g.data  = grid_arr.flatten().tolist()
        g.info  = _G()
        g.info.height     = h
        g.info.width      = w
        g.info.resolution = res
        g.info.origin     = _G()
        g.info.origin.position = _G()
        g.info.origin.position.x = ox
        g.info.origin.position.y = oy

        t0      = time.perf_counter()
        costmap = inflate_costmap(g, base_cost, obs_cost, decay)
        path    = astar(costmap, _m2c(rx, ry), _m2c(gx, gy), obs_cost, weight)
        plan_ms = (time.perf_counter() - t0) * 1e3

        now     = time.monotonic()
        elapsed = now - last_t
        plan_hz = 1.0 / elapsed if elapsed > 0 else 0.0
        last_t  = now

        # path: list of (row, col) → convert to (x, y) here
        if path:
            xs = [ox + (c + 0.5) * res for r, c in path]
            ys = [oy + (r + 0.5) * res for r, c in path]
            out_q.put((xs, ys, plan_ms, plan_hz))
        else:
            out_q.put((None, None, plan_ms, plan_hz))


class _PlannerBridge(threading.Thread):
    """메인 프로세스 측: 주기적으로 입력 큐에 데이터 push, 출력 큐를 state 에 반영."""

    def __init__(self, state: _State, in_q: mp.Queue, out_q: mp.Queue,
                 stop_evt: threading.Event) -> None:
        super().__init__(daemon=True, name='planner_bridge')
        self._state   = state
        self._in_q    = in_q
        self._out_q   = out_q
        self._stop    = stop_evt
        self._period  = 1.0 / PLANNER_RATE_HZ

    def run(self) -> None:
        while not self._stop.is_set():
            loop_start = time.monotonic()

            inp = self._state.planner_input()
            if inp is not None:
                # 큐가 이미 차있으면 버림 (뒤처지지 않게)
                if self._in_q.empty():
                    self._in_q.put(inp)

            # 출력 큐 드레인
            while not self._out_q.empty():
                item = self._out_q.get_nowait()
                xs, ys, plan_ms, plan_hz = item
                path = list(zip(xs, ys)) if xs else None
                self._state.push_path(path, plan_ms, plan_hz)

            sleep_t = self._period - (time.monotonic() - loop_start)
            if sleep_t > 0:
                time.sleep(sleep_t)


# ── matplotlib 시각화 (draw_idle 루프, ~20Hz) ─────────────────────────────────
def _run_viz(state: _State, dest_key: str, goal: tuple,
             stop_evt: threading.Event) -> None:
    cmap = ListedColormap(['white', 'tomato'])

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#333333')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    fig.suptitle(
        f'obstacle_map_node — OccupancyGrid + A* → {dest_key}',
        fontsize=13, color='white',
    )
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    img = ax.imshow(
        np.zeros((1, 1)), cmap=cmap, vmin=0, vmax=100,
        origin='lower', extent=[0, 1, 0, 1],
    )
    path_line, = ax.plot([], [], '-', color='limegreen', lw=2,
                         zorder=4, label='A* path')
    ax.plot(*goal, '*', color='gold', ms=14, zorder=5, label=dest_key)
    arrow = ax.quiver(
        [0], [0], [0], [0],
        angles='xy', scale_units='xy', scale=1,
        color='deepskyblue', width=0.008,
        headwidth=4, headlength=5, zorder=6, visible=False,
    )
    robot_dot, = ax.plot([], [], 'o', color='deepskyblue', ms=8, zorder=7)
    ax.legend(loc='upper right', facecolor='#333333',
              labelcolor='white', fontsize=9)

    info_txt = ax.text(
        0.02, 0.98, 'Waiting...',
        transform=ax.transAxes,
        va='top', ha='left', fontsize=9, family='monospace', color='white',
        bbox=dict(boxstyle='round,pad=0.3', fc='#333333', alpha=0.85),
    )

    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    prev_extent  = None
    last_loop_t  = time.monotonic()
    viz_hz       = 0.0
    draw_ms      = 0.0

    while not stop_evt.is_set():
        loop_s = time.monotonic()

        (grid, info, ox, oy, res, w, h,
         rx, ry, ryaw,
         path, plan_ms, plan_hz) = state.snapshot()

        if grid is not None and info is not None:
            ext = [ox, ox + w * res, oy, oy + h * res]
            img.set_data(grid)
            if ext != prev_extent:
                img.set_extent(ext)
                ax.set_xlim(ox - 0.5, ox + w * res + 0.5)
                ax.set_ylim(oy - 0.5, oy + h * res + 0.5)
                prev_extent = ext

            obstacle_count = int((grid == 100).sum())
            free_count     = int((grid == 0).sum())

            # 경로 오버레이 (플래너가 이미 xy 로 변환)
            if path:
                xs, ys = zip(*path)
                path_line.set_data(xs, ys)
                path_str = f'{len(path)} cells  ({len(path)*res:.2f} m)'
            else:
                path_line.set_data([], [])
                path_str = 'no path'

            if rx is not None:
                robot_dot.set_data([rx], [ry])
                dx = _ARROW_LEN_M * math.cos(ryaw)
                dy = _ARROW_LEN_M * math.sin(ryaw)
                arrow.set_offsets([[rx, ry]])
                arrow.set_UVC([dx], [dy])
                arrow.set_visible(True)
                pose_str = (f'robot  : ({rx:.2f}, {ry:.2f}) m  '
                            f'yaw={math.degrees(ryaw):+.1f}°')
            else:
                robot_dot.set_data([], [])
                arrow.set_visible(False)
                pose_str = 'robot  : waiting...'

            info_txt.set_text(
                f'grid   : {w}×{h}  res={res}m\n'
                f'obstacle: {obstacle_count}  free: {free_count}\n'
                f'{pose_str}\n'
                f'path   : {path_str}\n'
                f'plan   : {plan_ms:.1f} ms  {plan_hz:.1f} Hz\n'
                f'viz    : {draw_ms:.1f} ms  {viz_hz:.1f} Hz'
            )
        else:
            info_txt.set_text('Waiting for OccupancyGrid...')

        t_draw = time.perf_counter()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        draw_ms = (time.perf_counter() - t_draw) * 1e3

        now         = time.monotonic()
        viz_hz      = 1.0 / (now - last_loop_t) if (now - last_loop_t) > 0 else 0.0
        last_loop_t = now

        elapsed = now - loop_s
        sleep_t = _VIZ_PERIOD - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# ── 진입점 ───────────────────────────────────────────────────────────────────
def main() -> int:
    dest_key = sys.argv[1] if len(sys.argv) > 1 else "fridge"
    if dest_key not in LOCATIONS:
        print(f"[!] '{dest_key}' not in locations.json5.")
        print(f"    등록된 목적지: {list(LOCATIONS)}")
        return 1

    goal = LOCATIONS[dest_key]

    # 플래너 프로세스 시작
    in_q  = mp.Queue(maxsize=1)
    out_q = mp.Queue(maxsize=4)
    planner_cfg = dict(
        base_cost=BASE_COST, obs_cost=OBS_COST, decay_rate=DECAY_RATE,
        weight=WEIGHT, rate_hz=PLANNER_RATE_HZ,
    )
    planner_proc = mp.Process(
        target=_planner_worker,
        args=(in_q, out_q, goal, planner_cfg),
        daemon=True,
    )
    planner_proc.start()

    rclpy.init()
    state    = _State()
    stop_evt = threading.Event()

    node = _Subscriber(state)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True, name='ros_spin',
    )
    spin_thread.start()

    bridge = _PlannerBridge(state, in_q, out_q, stop_evt)
    bridge.start()

    print(f'[test_obstacle_map_node] dest={dest_key} {goal}  '
          f'planner={PLANNER_RATE_HZ:.0f}Hz  viz={_VIZ_HZ}Hz')
    try:
        _run_viz(state, dest_key, goal, stop_evt)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        in_q.put(None)          # 플래너 프로세스 종료 신호
        planner_proc.join(timeout=3.0)
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
