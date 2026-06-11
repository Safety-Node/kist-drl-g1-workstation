# test_lidar_node.py
"""
lidar_node — real-time point cloud visualization test (TASK-52).

/bridge/sensors/lidar/points  (sensor_msgs/PointCloud2) : G1 utlidar point cloud

Subscribes and renders a live top-down (X-Y) scatter plot coloured by Z height.

Usage:
    python system_hw_test/test_lidar_node.py
"""

from __future__ import annotations

import struct
import sys
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── visualization parameters ──────────────────────────────────
_REFRESH_MS: int = 200
_MAX_POINTS: int = 20_000   # subsample for render performance
_Z_MIN_M: float = -1.0
_Z_MAX_M: float = 2.0

_QOS_BE = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ── PointCloud2 binary parser ─────────────────────────────────

_DTYPE_MAP = {
    PointField.INT8:    np.int8,
    PointField.UINT8:   np.uint8,
    PointField.INT16:   np.int16,
    PointField.UINT16:  np.uint16,
    PointField.INT32:   np.int32,
    PointField.UINT32:  np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def _parse_xyz(msg: PointCloud2) -> Optional[np.ndarray]:
    """Return (N, 3) float32 array of [x, y, z] from a PointCloud2 message."""
    if msg.width == 0 or not msg.data:
        return None

    # Build numpy dtype from PointCloud2 fields
    fields = {f.name: f for f in msg.fields}
    for axis in ('x', 'y', 'z'):
        if axis not in fields:
            return None

    dtype = np.dtype([
        (f.name, _DTYPE_MAP.get(f.datatype, np.float32))
        for f in sorted(msg.fields, key=lambda f: f.offset)
    ])

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n_points = msg.width * msg.height
    point_step = msg.point_step

    # Reconstruct as structured array via strides
    try:
        arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    except Exception:
        return None

    x = arr['x'].astype(np.float32)
    y = arr['y'].astype(np.float32)
    z = arr['z'].astype(np.float32)

    xyz = np.stack([x, y, z], axis=1)
    # Drop NaN / Inf points
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid]


# ── shared state ──────────────────────────────────────────────
class _State:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._xyz: Optional[np.ndarray] = None
        self._stamp: float = 0.0
        self._count: int = 0
        self._hz_buf: list[float] = []

    def push(self, xyz: np.ndarray, stamp: float) -> None:
        with self._lock:
            self._xyz = xyz
            now = time.monotonic()
            self._hz_buf.append(now)
            self._hz_buf = [t for t in self._hz_buf if now - t < 2.0]
            self._stamp = stamp
            self._count += 1

    def snapshot(self):
        with self._lock:
            xyz = self._xyz
            hz = len(self._hz_buf) / 2.0 if len(self._hz_buf) >= 2 else 0.0
            return xyz, self._stamp, hz


# ── ROS2 구독 노드 ────────────────────────────────────────────
class _Subscriber(Node):
    def __init__(self, state: _State) -> None:
        super().__init__('test_lidar_viz')
        self._state = state
        self.create_subscription(
            PointCloud2, '/bridge/sensors/lidar/points',
            self._on_cloud, _QOS_BE,
        )
        self.get_logger().info('test_lidar_node: waiting for /bridge/sensors/lidar/points ...')

    def _on_cloud(self, msg: PointCloud2) -> None:
        xyz = _parse_xyz(msg)
        if xyz is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._state.push(xyz, stamp)


# ── matplotlib 시각화 ─────────────────────────────────────────
def _run_viz(state: _State, stop_evt: threading.Event) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("lidar_node — utlidar point cloud top-down view  (TASK-52)", fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#1e1e1e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    fig.suptitle(
        "lidar_node — utlidar point cloud top-down view  (TASK-52)",
        fontsize=13, color="white",
    )

    sc = ax.scatter([], [], s=0.5, c=[], cmap="plasma",
                    vmin=_Z_MIN_M, vmax=_Z_MAX_M, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("z (m)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    info_txt = ax.text(
        0.02, 0.98, "Waiting...",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9, family="monospace", color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#333333", alpha=0.85),
    )

    def _update(_frame):
        if stop_evt.is_set():
            return

        xyz, stamp, hz = state.snapshot()
        if xyz is None or len(xyz) == 0:
            info_txt.set_text("Waiting for point cloud...")
            return sc, info_txt

        # Subsample for render performance
        if len(xyz) > _MAX_POINTS:
            idx = np.random.choice(len(xyz), _MAX_POINTS, replace=False)
            xyz = xyz[idx]

        sc.set_offsets(xyz[:, :2])
        sc.set_array(xyz[:, 2])

        xlim = (xyz[:, 0].min() - 0.5, xyz[:, 0].max() + 0.5)
        ylim = (xyz[:, 1].min() - 0.5, xyz[:, 1].max() + 0.5)
        half = max((xlim[1] - xlim[0]) / 2, (ylim[1] - ylim[0]) / 2, 1.0)
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

        info_txt.set_text(
            f"points : {len(xyz):,}\n"
            f"hz     : {hz:.1f}\n"
            f"z range: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}] m\n"
            f"stamp  : {stamp:.3f}"
        )

        return sc, info_txt

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
    state = _State()
    stop_evt = threading.Event()

    node = _Subscriber(state)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True, name="ros_spin",
    )
    spin_thread.start()

    print("[test_lidar_node] viz started — run run_onboard.sh on NX to stream point cloud")

    try:
        _run_viz(state, stop_evt)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
