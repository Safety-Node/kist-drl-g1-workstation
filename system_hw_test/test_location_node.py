# test_location_node.py
"""
location_node — real-time visualization test (TASK-51).

/bridge/sensors/location  (geometry_msgs/PoseStamped) : EKF fused position
/bridge/sensors/uwb_pose  (geometry_msgs/PoseStamped) : raw UWB measurement

Subscribes to both topics and plots real-time trajectory + heading arrow
using matplotlib.

Usage:
    python system_hw_test/test_location_node.py
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── visualization parameters ──────────────────────────────────
_TRAIL_LEN: int = 500
_MARGIN_M: float = 2.0
_ARROW_LEN_M: float = 0.4
_REFRESH_MS: int = 200

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


# ── shared data buffer ────────────────────────────────────────
class _State:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ekf_x: deque[float] = deque(maxlen=_TRAIL_LEN)
        self.ekf_y: deque[float] = deque(maxlen=_TRAIL_LEN)
        self.ekf_yaw: float | None = None
        self.raw_x: deque[float] = deque(maxlen=_TRAIL_LEN)
        self.raw_y: deque[float] = deque(maxlen=_TRAIL_LEN)
        self.odom_x: float | None = None
        self.odom_y: float | None = None
        self.odom_yaw: float | None = None
        self.std_bias_deg: float | None = None
        self.diag_calibrated: bool | None = None

    def push_ekf(self, x: float, y: float, yaw: float) -> None:
        with self._lock:
            self.ekf_x.append(x)
            self.ekf_y.append(y)
            self.ekf_yaw = yaw

    def push_raw(self, x: float, y: float) -> None:
        with self._lock:
            self.raw_x.append(x)
            self.raw_y.append(y)

    def push_odom(self, x: float, y: float, yaw: float) -> None:
        with self._lock:
            self.odom_x = x
            self.odom_y = y
            self.odom_yaw = yaw

    def push_diag(self, std_bias_deg: float, calibrated: bool) -> None:
        with self._lock:
            self.std_bias_deg = std_bias_deg
            self.diag_calibrated = calibrated

    def snapshot(self):
        with self._lock:
            return (
                list(self.ekf_x), list(self.ekf_y), self.ekf_yaw,
                list(self.raw_x), list(self.raw_y),
                self.odom_x, self.odom_y, self.odom_yaw,
                self.std_bias_deg, self.diag_calibrated,
            )


# ── ROS2 구독 노드 ────────────────────────────────────────────
class _Subscriber(Node):
    def __init__(self, state: _State) -> None:
        super().__init__('test_location_viz')
        self._state = state

        self.create_subscription(
            PoseStamped, '/bridge/sensors/location',
            self._on_location, _QOS_BE,
        )
        self.create_subscription(
            PoseStamped, '/bridge/sensors/uwb_pose',
            self._on_uwb, _QOS_BE,
        )
        self.create_subscription(
            Odometry, '/bridge/sensors/odom',
            self._on_odom, _QOS_BE,
        )
        self.create_subscription(
            DiagnosticStatus, '/bridge/sensors/location/diagnostics',
            self._on_diag, _QOS_BE,
        )
        self.get_logger().info('test_location_node: subscriptions started')

    def _on_location(self, msg: PoseStamped) -> None:
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = _quat_to_yaw(msg.pose.orientation)
        self._state.push_ekf(x, y, yaw)

    def _on_uwb(self, msg: PoseStamped) -> None:
        self._state.push_raw(msg.pose.position.x, msg.pose.position.y)

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = _quat_to_yaw(msg.pose.pose.orientation)
        self._state.push_odom(x, y, yaw)

    def _on_diag(self, msg: DiagnosticStatus) -> None:
        std_bias = None
        calibrated = msg.level == DiagnosticStatus.OK
        for kv in msg.values:
            if kv.key == 'std_bias_deg':
                try:
                    std_bias = float(kv.value)
                except ValueError:
                    pass
        if std_bias is not None:
            self._state.push_diag(std_bias, calibrated)


# ── matplotlib 시각화 ─────────────────────────────────────────
def _auto_lim(ax, xs: list[float], ys: list[float]) -> None:
    if not xs:
        return
    xmin, xmax = min(xs) - _MARGIN_M, max(xs) + _MARGIN_M
    ymin, ymax = min(ys) - _MARGIN_M, max(ys) + _MARGIN_M
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max((xmax - xmin) / 2, (ymax - ymin) / 2, 1.0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def _run_viz(state: _State, stop_evt: threading.Event) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("location_node — UWB + Odom EKF  (TASK-51)", fontsize=13)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    raw_sc,  = ax.plot([], [], "rx", ms=5, alpha=0.5, label="raw UWB")
    ekf_ln,  = ax.plot([], [], "b-", lw=1.2, label="EKF path")
    ekf_cur, = ax.plot([], [], "bo", ms=9, zorder=6, label="EKF pos")
    arrow = ax.quiver(
        [0], [0], [0], [0],
        angles="xy", scale_units="xy", scale=1,
        color="royalblue", width=0.006, headwidth=4, headlength=5,
        zorder=6, visible=False,
    )
    ax.legend(loc="upper left", fontsize=9)
    info_txt = ax.text(
        0.98, 0.98, "Waiting...",
        transform=ax.transAxes,
        va="top", ha="right", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
    )

    status_txt = ax.text(
        0.98, 0.02, "odom: no data",
        transform=ax.transAxes,
        va="bottom", ha="right", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.75),
    )

    def _update(_frame):
        if stop_evt.is_set():
            return

        ex, ey, eyaw, rx, ry, ox, oy, oyaw, std_bias, diag_cal = state.snapshot()

        raw_sc.set_data(rx, ry)
        ekf_ln.set_data(ex, ey)
        ekf_cur.set_data([ex[-1]] if ex else [], [ey[-1]] if ey else [])

        if ex and eyaw is not None:
            dx = _ARROW_LEN_M * math.cos(eyaw)
            dy = _ARROW_LEN_M * math.sin(eyaw)
            arrow.set_offsets([[ex[-1], ey[-1]]])
            arrow.set_UVC([dx], [dy])
            arrow.set_visible(True)
        else:
            arrow.set_visible(False)

        _auto_lim(ax, ex + rx, ey + ry)

        # top-right: EKF status + calibration progress
        cal_str = f"{'YES' if diag_cal else 'NO'}" if diag_cal is not None else "---"
        bias_str = f"{std_bias:.2f} deg" if std_bias is not None else "---"
        if ex:
            yaw_deg = math.degrees(eyaw) if eyaw is not None else float("nan")
            info_txt.set_text(
                f"EKF pos : ({ex[-1]:.3f}, {ey[-1]:.3f}) m\n"
                f"yaw     : {yaw_deg:+.1f} deg\n"
                f"samples : ekf={len(ex)}  uwb={len(rx)}\n"
                f"std_bias: {bias_str}\n"
                f"calib   : {cal_str}"
            )
        else:
            info_txt.set_text(
                f"Waiting for EKF...\n"
                f"std_bias: {bias_str}\n"
                f"calib   : {cal_str}\n"
                f"(move robot to calibrate)"
            )

        # bottom-right: odom
        if ox is not None:
            oyaw_deg = math.degrees(oyaw) if oyaw is not None else float("nan")
            status_txt.set_text(
                f"odom x   : {ox:.3f} m\n"
                f"odom y   : {oy:.3f} m\n"
                f"odom yaw : {oyaw_deg:+.1f} deg"
            )
        else:
            status_txt.set_text("odom: no data")

        return raw_sc, ekf_ln, ekf_cur, arrow, info_txt, status_txt

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

    print("[test_location_node] viz started — run run_onboard.sh on NX then move the robot")
    print("  EKF trajectory will not appear until yaw_calibrated is achieved.")

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
