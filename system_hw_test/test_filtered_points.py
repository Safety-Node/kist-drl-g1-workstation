# test_filtered_points.py
"""
obstacle_map_node — filtered point cloud visualization test (TASK-53).

/bridge/sensors/lidar/filtered_points  (sensor_msgs/PointCloud2) : 필터링된 포인트
/bridge/sensors/location               (geometry_msgs/PoseStamped) : robot pose

Z·robot_radius 필터를 통과한 포인트를 map 프레임에서 산점도로 렌더링한다.
test_obstacle_map_node.py 와 병렬 실행 가능.

Usage:
    python system_hw_test/test_filtered_points.py
"""

from __future__ import annotations

import math
import struct
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_REFRESH_MS = 200

_QOS_BE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_ARROW_LEN_M = 0.3

_FIELD_DTYPE = {
    PointField.INT8:    np.int8,
    PointField.UINT8:   np.uint8,
    PointField.INT16:   np.int16,
    PointField.UINT16:  np.uint16,
    PointField.INT32:   np.int32,
    PointField.UINT32:  np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def _quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _parse_xy(msg: PointCloud2) -> np.ndarray | None:
    if msg.width == 0 or not msg.data:
        return None
    fields = {f.name: f for f in msg.fields}
    if 'x' not in fields or 'y' not in fields:
        return None
    dtype = np.dtype([
        (f.name, _FIELD_DTYPE.get(f.datatype, np.float32))
        for f in sorted(msg.fields, key=lambda f: f.offset)
    ])
    try:
        arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    except Exception:
        return None
    xy = np.stack([arr['x'].astype(np.float32), arr['y'].astype(np.float32)], axis=1)
    return xy[np.isfinite(xy).all(axis=1)]


# ── shared state ──────────────────────────────────────────────
class _State:
    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self.pts_xy: np.ndarray | None = None
        self.robot_x: float | None = None
        self.robot_y: float | None = None
        self.robot_yaw: float | None = None

    def push_cloud(self, pts_xy: np.ndarray) -> None:
        with self._lock:
            self.pts_xy = pts_xy

    def push_pose(self, x: float, y: float, yaw: float) -> None:
        with self._lock:
            self.robot_x   = x
            self.robot_y   = y
            self.robot_yaw = yaw

    def snapshot(self):
        with self._lock:
            return self.pts_xy, self.robot_x, self.robot_y, self.robot_yaw


# ── ROS2 구독 노드 ────────────────────────────────────────────
class _Subscriber(Node):
    def __init__(self, state: _State) -> None:
        super().__init__('test_filtered_points_viz')
        self._state = state

        self.create_subscription(
            PointCloud2, '/bridge/sensors/lidar/filtered_points',
            self._on_cloud, _QOS_BE,
        )
        self.create_subscription(
            PoseStamped, '/bridge/sensors/location',
            self._on_location, _QOS_BE,
        )
        self.get_logger().info(
            'test_filtered_points: waiting for /bridge/sensors/lidar/filtered_points ...'
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        xy = _parse_xy(msg)
        if xy is not None:
            self._state.push_cloud(xy)

    def _on_location(self, msg: PoseStamped) -> None:
        self._state.push_pose(
            msg.pose.position.x,
            msg.pose.position.y,
            _quat_to_yaw(msg.pose.orientation),
        )


# ── matplotlib 시각화 ─────────────────────────────────────────
def _run_viz(state: _State, stop_evt: threading.Event) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#222222')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    fig.suptitle(
        'obstacle_map_node — filtered points  (TASK-53)',
        fontsize=13, color='white',
    )
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 8)

    scatter = ax.scatter([], [], s=4, c='lime', alpha=0.6, linewidths=0)

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

        pts_xy, rx, ry, ryaw = state.snapshot()

        if pts_xy is None:
            info_txt.set_text('Waiting for filtered_points...')
            return scatter, arrow, robot_dot, info_txt

        if len(pts_xy) > 0:
            scatter.set_offsets(pts_xy)
        else:
            scatter.set_offsets(np.empty((0, 2)))

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
            f'points : {len(pts_xy)}\n'
            f'{pose_str}'
        )

        return scatter, arrow, robot_dot, info_txt

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
    state    = _State()
    stop_evt = threading.Event()

    node = _Subscriber(state)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True, name='ros_spin',
    )
    spin_thread.start()

    print('[test_filtered_points] viz started — run run_onboard.sh on NX')
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
