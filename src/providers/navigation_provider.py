"""
NavigationProvider [TASK-48, REQ "PC NavigationProvider"]

PC-side navigation: translates nav sub-task prompts → continuous walking
velocity (vx/vy/vyaw) via UnitreeG1Provider.publish_twist.

Position source : /bridge/sensors/location  (EKF-fused PoseStamped, map frame)
Obstacle source : /bridge/sensors/lidar/occupancy  (OccupancyGrid, map frame)
Location table  : config/locations.json5  (place_id → [x, y])

Algorithm: holonomic potential-field.
  - Attractive velocity toward goal (P-controller, capped to max_speed).
  - Repulsive velocity away from OccupancyGrid obstacles within
    repulsion_radius.
  - Net velocity rotated from map → body frame using robot yaw.
  - Yaw rate = P-controller toward the net velocity heading.

Lifecycle:
  start()  → spin control thread + register estop callback
  stop()   → stop thread + publish zero Twist + unregister callback
  submit_nav_subtask(prompt)  → parse prompt → enqueue goal (non-blocking)

Threading: submit_nav_subtask() is called from the asyncio loop thread;
it only writes _goal under _goal_lock (non-blocking).  The control thread
reads _goal under the same lock.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import json5
import numpy as np

from .singleton import singleton
from .unitree_g1_provider import UnitreeG1Provider

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Config + State
# ---------------------------------------------------------------------------

@dataclass
class NavigationProviderConfig:
    """NavigationProvider runtime configuration."""

    control_rate_hz: float = 10.0
    locations_file: str = "config/locations.json5"

    # Speed limits
    max_speed: float = 0.5      # m/s — translational cap (body frame)
    max_vyaw: float = 0.8       # rad/s — yaw rate cap

    # Attraction gains
    kp_xy: float = 0.8          # P-gain for goal attraction (1/s)
    kp_yaw: float = 1.5         # P-gain for heading alignment (1/s)

    # Arrival
    arrival_tol: float = 0.25   # m — goal reached when dist < this

    # Obstacle repulsion
    repulsion_gain: float = 0.2   # m²/s
    repulsion_radius: float = 0.8 # m — obstacle influence distance

    # Stale-data handling
    pose_timeout_s: float = 0.5   # s — wait for pose if stale


@dataclass(frozen=True)
class NavigationState:
    t_monotonic: float
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    mode: str = "IDLE"    # IDLE / NAVIGATING / ARRIVED / ESTOP / WAITING_POSE
    goal_id: Optional[str] = None
    dist_to_goal: float = 0.0


# ---------------------------------------------------------------------------
# NavigationProvider
# ---------------------------------------------------------------------------

@singleton
class NavigationProvider:
    """
    PC-side holonomic navigation (REQ "PC NavigationProvider").

    Consumes nav sub-task prompts from MoveConnector, drives the robot to
    named locations defined in config/locations.json5.
    """

    def __init__(self, config: Optional[NavigationProviderConfig] = None):
        self._config = config or NavigationProviderConfig()
        self._unitree_g1 = UnitreeG1Provider()

        # Location table
        self._locations: Dict[str, Tuple[float, float]] = {}
        self._load_locations()

        # Goal (written by submit_nav_subtask, read by worker)
        self._goal_lock = threading.Lock()
        self._goal: Optional[Tuple[float, float]] = None
        self._goal_id: Optional[str] = None

        # E-STOP flag (written by estop callback, read by worker)
        self._estop_active: bool = False

        # State (written by worker, read by external callers)
        self._state_lock = threading.Lock()
        self._latest_state = NavigationState(t_monotonic=time.monotonic())

        # Worker thread
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        _log.info(
            "NavigationProvider: initialized (rate=%.1fHz, %d locations)",
            self._config.control_rate_hz, len(self._locations),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._unitree_g1.register_estop_callback(self._on_estop)

        self._stop_evt.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="NavigationProviderWorker",
            daemon=True,
        )
        self._thread.start()
        _log.info("NavigationProvider: started")

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                _log.warning("NavigationProvider: worker thread did not stop in 2s")
            self._thread = None

        self._unitree_g1.unregister_estop_callback(self._on_estop)
        self._unitree_g1.publish_twist(0.0, 0.0, 0.0)

        with self._state_lock:
            self._latest_state = NavigationState(
                t_monotonic=time.monotonic(), mode="IDLE",
            )
        _log.info("NavigationProvider: stopped")

    # ------------------------------------------------------------------
    # Sub-task dispatch (called from MoveConnector → asyncio loop thread)
    # ------------------------------------------------------------------

    def submit_nav_subtask(self, prompt: str) -> None:
        """Parse prompt → location → arm control loop. Non-blocking."""
        location_id = self._parse_location(prompt)
        if location_id is None:
            _log.warning("NavigationProvider: no known location in prompt %r", prompt)
            return
        gx, gy = self._locations[location_id]
        with self._goal_lock:
            self._goal = (gx, gy)
            self._goal_id = location_id
        _log.info("NavigationProvider: goal set → %s (%.2f, %.2f)", location_id, gx, gy)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> NavigationState:
        with self._state_lock:
            return self._latest_state

    # ------------------------------------------------------------------
    # Internal: worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        dt = 1.0 / self._config.control_rate_hz
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:
                _log.exception("NavigationProvider: tick error")
            elapsed = time.monotonic() - t0
            self._stop_evt.wait(max(0.0, dt - elapsed))

    def _tick(self) -> None:
        # E-STOP wins unconditionally
        if self._estop_active:
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(
                t_monotonic=time.monotonic(), mode="ESTOP",
            ))
            return

        with self._goal_lock:
            goal = self._goal
            goal_id = self._goal_id

        if goal is None:
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(t_monotonic=time.monotonic()))
            return

        # Get robot pose
        pose_cache = self._unitree_g1.location
        now = time.monotonic()
        if pose_cache.stale(now, self._config.pose_timeout_s):
            _log.warning("NavigationProvider: waiting for robot pose...", stacklevel=2)
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(
                t_monotonic=now, mode="WAITING_POSE", goal_id=goal_id,
            ))
            return

        pose_msg = pose_cache.value
        rx = pose_msg.pose.position.x
        ry = pose_msg.pose.position.y
        ryaw = _quat_to_yaw(pose_msg.pose.orientation)

        gx, gy = goal
        dx_g = gx - rx
        dy_g = gy - ry
        dist = math.sqrt(dx_g * dx_g + dy_g * dy_g)

        if dist < self._config.arrival_tol:
            with self._goal_lock:
                self._goal = None
                self._goal_id = None
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(
                t_monotonic=time.monotonic(), mode="ARRIVED", goal_id=goal_id,
            ))
            _log.info("NavigationProvider: arrived at %s", goal_id)
            return

        # Attractive velocity (map frame, capped)
        att_speed = min(self._config.kp_xy * dist, self._config.max_speed)
        vx_map = att_speed * dx_g / dist
        vy_map = att_speed * dy_g / dist

        # Repulsive velocity from occupancy grid
        occ_cache = self._unitree_g1.occupancy
        if not occ_cache.stale(now, 0.5) and occ_cache.value is not None:
            vx_rep, vy_rep = self._compute_repulsion(occ_cache.value, rx, ry)
            vx_map += vx_rep
            vy_map += vy_rep

        # Clamp total translational speed
        speed = math.sqrt(vx_map * vx_map + vy_map * vy_map)
        if speed > self._config.max_speed:
            s = self._config.max_speed / speed
            vx_map *= s
            vy_map *= s

        # Map frame → body frame
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)
        vx_body =  cos_y * vx_map + sin_y * vy_map
        vy_body = -sin_y * vx_map + cos_y * vy_map

        # Yaw: turn toward net velocity heading
        desired_hdg = math.atan2(vy_map, vx_map)
        hdg_err = _wrap_pi(desired_hdg - ryaw)
        vyaw = max(-self._config.max_vyaw,
                   min(self._config.max_vyaw, self._config.kp_yaw * hdg_err))

        self._unitree_g1.publish_twist(vx_body, vy_body, vyaw)
        self._set_state(NavigationState(
            t_monotonic=time.monotonic(),
            vx=vx_body, vy=vy_body, vyaw=vyaw,
            mode="NAVIGATING",
            goal_id=goal_id,
            dist_to_goal=dist,
        ))

    def _compute_repulsion(self, grid, rx: float, ry: float) -> Tuple[float, float]:
        info = grid.info
        res = float(info.resolution)
        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)
        w = int(info.width)
        h = int(info.height)
        r_r = self._config.repulsion_radius
        gain = self._config.repulsion_gain

        # Decode obstacle cells
        data = np.frombuffer(bytes(grid.data), dtype=np.int8).reshape(h, w)
        rows, cols = np.where(data == 100)
        if len(rows) == 0:
            return 0.0, 0.0

        # Obstacle cell centers in map frame
        cx = ox + (cols.astype(np.float32) + 0.5) * res
        cy = oy + (rows.astype(np.float32) + 0.5) * res

        # Vectors from obstacles toward robot (repulsion direction)
        ddx = rx - cx
        ddy = ry - cy
        dist = np.sqrt(ddx * ddx + ddy * ddy)

        # Only obstacles within repulsion radius; exclude zero-distance singularity
        mask = (dist < r_r) & (dist > 0.05)
        ddx = ddx[mask]
        ddy = ddy[mask]
        dist = dist[mask]
        if len(dist) == 0:
            return 0.0, 0.0

        # Gradient of 0.5 * gain * (1/d - 1/r)^2
        mag = gain * (1.0 / dist - 1.0 / r_r) / (dist * dist)
        vx_rep = float(np.sum(mag * ddx / dist))
        vy_rep = float(np.sum(mag * ddy / dist))
        return vx_rep, vy_rep

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _load_locations(self) -> None:
        locs_path = _REPO_ROOT / self._config.locations_file
        try:
            raw = json5.loads(locs_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _log.warning("NavigationProvider: locations file not found: %s", locs_path)
            return
        except Exception:
            _log.exception("NavigationProvider: failed to load %s", locs_path)
            return
        for name, coords in raw.items():
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                self._locations[name] = (float(coords[0]), float(coords[1]))
            else:
                _log.warning("NavigationProvider: invalid coords for %r: %r", name, coords)
        _log.info("NavigationProvider: loaded %d locations from %s", len(self._locations), locs_path.name)

    def _parse_location(self, prompt: str) -> Optional[str]:
        lower = prompt.lower()
        # Try longer keys first to prefer more specific matches
        for name in sorted(self._locations.keys(), key=len, reverse=True):
            if name.lower() in lower:
                return name
        return None

    def _on_estop(self, active: bool, ts: float) -> None:
        self._estop_active = active
        if active:
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            _log.warning("NavigationProvider: E-STOP active — halted")
        else:
            _log.info("NavigationProvider: E-STOP cleared")

    def _set_state(self, state: NavigationState) -> None:
        with self._state_lock:
            self._latest_state = state
