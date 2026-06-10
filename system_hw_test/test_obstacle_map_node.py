# test_obstacle_map_node.py
"""
obstacle_map_node — real-time occupancy grid visualization test (TASK-53).

/bridge/sensors/lidar/occupancy  (nav_msgs/OccupancyGrid) : 2D obstacle map
/bridge/sensors/location         (geometry_msgs/PoseStamped) : robot pose

OccupancyGrid를 격자로 렌더링하고 로봇 현재 위치를 오버레이한다.
FREE(0) → 흰색, OBSTACLE(100) → 빨간색, 로봇 → 파란 화살표.

Usage:
    python system_hw_test/test_obstacle_map_node.py
"""

from __future__ import annotations

import math
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

_REFRESH_MS = 200

_QOS_BE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_ARROW_LEN_M = 0.3


def _quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


# ── shared state ──────────────────────────────────────────────
class _State:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.grid: np.ndarray | None = None   # (height, width) int8
        self.info = None                       # nav_msgs/MapMetaData
        self.robot_x: float | None = None
        self.robot_y: float | None = None
        self.robot_yaw: float | None = None

    def push_grid(self, grid: np.ndarray, info) -> None:
        with self._lock:
            self.grid = grid
            self.info = info

    def push_pose(self, x: float, y: float, yaw: float) -> None:
        with self._lock:
            self.robot_x   = x
            self.robot_y   = y
            self.robot_yaw = yaw

    def snapshot(self):
        with self._lock:
            return self.grid, self.info, self.robot_x, self.robot_y, self.robot_yaw


# ── ROS2 구독 노드 ────────────────────────────────────────────
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
        self.get_logger().info(
            'test_obstacle_map_node: waiting for /bridge/sensors/lidar/occupancy ...'
        )

    def _on_grid(self, msg: OccupancyGrid) -> None:
        grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self._state.push_grid(grid, msg.info)

    def _on_location(self, msg: PoseStamped) -> None:
        self._state.push_pose(
            msg.pose.position.x,
            msg.pose.position.y,
            _quat_to_yaw(msg.pose.orientation),
        )


# ── matplotlib 시각화 ─────────────────────────────────────────
def _run_viz(state: _State, stop_evt: threading.Event) -> None:
    # FREE=0 → white, OBSTACLE=100 → red
    cmap = ListedColormap(['white', 'tomato'])

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#333333')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    fig.suptitle(
        'obstacle_map_node — OccupancyGrid  (TASK-53)',
        fontsize=13, color='white',
    )
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

    img = ax.imshow(
        np.zeros((1, 1)), cmap=cmap, vmin=0, vmax=100,
        origin='lower', extent=[0, 1, 0, 1],
    )

    arrow = ax.quiver(
        [0], [0], [0], [0],
        angles='xy', scale_units='xy', scale=1,
        color='deepskyblue', width=0.008,
        headwidth=4, headlength=5, zorder=5, visible=False,
    )
    robot_dot, = ax.plot([], [], 'o', color='deepskyblue', ms=8, zorder=6)

    info_txt = ax.text(
        0.02, 0.98, 'Waiting...',
        transform=ax.transAxes,
        va='top', ha='left', fontsize=9, family='monospace', color='white',
        bbox=dict(boxstyle='round,pad=0.3', fc='#333333', alpha=0.85),
    )

    def _update(_frame):
        if stop_evt.is_set():
            return

        grid, info, rx, ry, ryaw = state.snapshot()

        if grid is None or info is None:
            info_txt.set_text('Waiting for OccupancyGrid...')
            return img, arrow, robot_dot, info_txt

        ox = info.origin.position.x
        oy = info.origin.position.y
        res = info.resolution
        w, h = info.width, info.height

        # 격자 이미지 업데이트
        img.set_data(grid)
        img.set_extent([ox, ox + w * res, oy, oy + h * res])
        img.set_clim(0, 100)
        ax.set_xlim(ox - 0.5, ox + w * res + 0.5)
        ax.set_ylim(oy - 0.5, oy + h * res + 0.5)

        obstacle_count = int((grid == 100).sum())
        free_count     = int((grid == 0).sum())

        # 로봇 위치/방향 오버레이
        if rx is not None:
            robot_dot.set_data([rx], [ry])
            dx = _ARROW_LEN_M * math.cos(ryaw)
            dy = _ARROW_LEN_M * math.sin(ryaw)
            arrow.set_offsets([[rx, ry]])
            arrow.set_UVC([dx], [dy])
            arrow.set_visible(True)
            pose_str = f'robot  : ({rx:.2f}, {ry:.2f}) m  yaw={math.degrees(ryaw):+.1f}°'
        else:
            robot_dot.set_data([], [])
            arrow.set_visible(False)
            pose_str = 'robot  : waiting...'

        info_txt.set_text(
            f'grid   : {w}×{h}  res={res}m\n'
            f'obstacle: {obstacle_count} cells\n'
            f'free   : {free_count} cells\n'
            f'{pose_str}'
        )

        return img, arrow, robot_dot, info_txt

    ani = animation.FuncAnimation(
        fig, _update, interval=_REFRESH_MS, blit=False, cache_frame_data=False,
    )

    try:
        plt.tight_layout()
        plt.show()
    finally:
        stop_evt.set()
        ani.event_source.stop()


# ── 진입점 ───────────────────────────────────────────────────
def main() -> int:
    rclpy.init()
    state  = _State()
    stop_evt = threading.Event()

    node = _Subscriber(state)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True, name='ros_spin',
    )
    spin_thread.start()

    print('[test_obstacle_map_node] viz started — run run_onboard.sh on NX')
    try:
        _run_viz(state, stop_evt)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
