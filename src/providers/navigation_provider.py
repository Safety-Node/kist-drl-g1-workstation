"""
NavigationProvider [TASK-48, REQ "PC NavigationProvider"]

PC-side navigation: translates nav sub-task prompts → continuous walking
velocity (vx/vy/vyaw) via UnitreeG1Provider.publish_twist.

Position source : /bridge/sensors/location  (EKF-fused PoseStamped, map frame)
Obstacle source : /bridge/sensors/lidar/points  (PointCloud2 — consumed by route_planner_py C++ pipeline)
Location table  : config/locations.json5  (place_id → [x, y])

Algorithm: route_planner_py (C++) pipeline + lookahead path follower.
  1. Route planning  — RoutePlannerPipeline (T1-T6 threads) continuously
     processes PointCloud2 → EDT Costmap → A* path in the background.
  2. Path following  — Each tick the robot projects itself onto the path,
     picks a point `lookahead_cells` ahead, and drives toward it with a
     P-controller (map-frame attraction, then rotated to body frame).
  3. Yaw control    — P-controller toward the net velocity heading.

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
from dataclasses import dataclass, field

from .nav_types import NavVelCmd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json5
import numpy as np
import yaml

import route_planner_py as rp

from .singleton import singleton
from .unitree_g1_provider import UnitreeG1Provider

_log = logging.getLogger(__name__)

_REPO_ROOT        = Path(__file__).resolve().parent.parent.parent
_NAV_CONFIG_PATH  = Path(__file__).resolve().parent / "config" / "navigation" / "config.yaml"


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

    control_rate_hz: float = 20.0
    locations_file:  str   = "src/providers/config/navigation/locations.json5"

    # Speed limits
    max_speed: float = 0.5
    max_vyaw:  float = 0.8

    # Attraction gains
    kp_xy:  float = 0.8
    kp_yaw: float = 1.5

    # Arrival
    arrival_tol: float = 0.25
    yaw_tol:     float = 0.10   # rad — final heading tolerance (~6°)

    # Path following
    lookahead_cells:          int   = 8
    target_ema_alpha:         float = 0.35   # 0=frozen, 1=no smoothing
    path_update_every_n_ticks: int  = 5      # refresh path from C++ every N ticks

    # Stale-data handling
    pose_timeout_s: float = 0.5


def _load_nav_config() -> NavigationProviderConfig:
    """Load NavigationProviderConfig from config/navigation/config.yaml."""
    try:
        raw = yaml.safe_load(_NAV_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        _log.warning("NavigationProvider: config not found at %s, using defaults", _NAV_CONFIG_PATH)
        return NavigationProviderConfig()
    except Exception:
        _log.exception("NavigationProvider: failed to load config, using defaults")
        return NavigationProviderConfig()

    def _g(section: str, key: str, default):
        return raw.get(section, {}).get(key, default)

    return NavigationProviderConfig(
        control_rate_hz    = _g("control",  "rate_hz",           20.0),
        pose_timeout_s     = _g("control",  "pose_timeout_s",     0.5),
        max_speed          = _g("speed",    "max_speed",           0.5),
        max_vyaw           = _g("speed",    "max_vyaw",            0.8),
        kp_xy              = _g("gains",    "kp_xy",               0.8),
        kp_yaw             = _g("gains",    "kp_yaw",              1.5),
        arrival_tol        = _g("arrival",  "tol",                 0.25),
        yaw_tol            = _g("arrival",  "yaw_tol",             0.10),
        lookahead_cells           = _g("path", "lookahead_cells",           8),
        target_ema_alpha          = _g("path", "target_ema_alpha",          0.35),
        path_update_every_n_ticks = _g("path", "path_update_every_n_ticks", 5),
        locations_file     = raw.get("locations_file", "src/providers/config/navigation/locations.json5"),
    )



@dataclass(frozen=True)
class NavState:
    t_monotonic: float
    vel: NavVelCmd = NavVelCmd()
    mode: str = "IDLE"    # IDLE / NAVIGATING / ALIGNING / ARRIVED / ESTOP / WAITING_POSE / PLANNING
    goal_id: Optional[str] = None
    dist_to_goal: float = 0.0


NavigationState = NavState  # backwards compatibility alias


# ---------------------------------------------------------------------------
# NavigationProvider
# ---------------------------------------------------------------------------

@singleton
class NavigationProvider:
    """
    PC-side A*+lookahead navigation (REQ "PC NavigationProvider").

    Consumes nav sub-task prompts from MoveConnector, drives the robot to
    named locations defined in config/locations.json5.
    """

    def __init__(self, config: Optional[NavigationProviderConfig] = None):
        self._config = config if config is not None else _load_nav_config()
        self._unitree_g1 = UnitreeG1Provider()

        # Location table: name → (x, y, Optional[yaw_rad])
        self._locations: Dict[str, Tuple[float, float, Optional[float]]] = {}
        self._load_locations()

        # Route-planner C++ pipeline (PointCloud2 → EDT Costmap → A*)
        if not rp.ok():
            rp.init([])
        self._pipeline = rp.RoutePlannerPipeline()

        # Goal (written by submit_nav_subtask, read by worker)
        self._goal_lock = threading.Lock()
        self._goal:     Optional[Tuple[float, float]] = None
        self._goal_yaw: Optional[float] = None
        self._goal_id:  Optional[str] = None

        # Path-following smoothing state (written/read only by worker thread)
        self._smooth_tx:    Optional[float] = None
        self._smooth_ty:    Optional[float] = None
        self._cached_path:  Optional[List]  = None
        self._path_tick_count: int          = 0
        self._last_set_goal: Optional[Tuple[float, float]] = None

        # E-STOP flag (written by estop callback, read by worker)
        self._estop_active: bool = False

        # State (written by worker, read by external callers)
        self._state_lock = threading.Lock()
        self._latest_state = NavigationState(t_monotonic=time.monotonic())

        # Worker thread
        self._stop_evt = threading.Event()
        self._thread:   Optional[threading.Thread] = None
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

        self._pipeline.start()
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

        self._pipeline.stop()
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
        """Parse prompt → location → queue for control loop. Non-blocking."""
        location_id = self._parse_location(prompt)
        if location_id is None:
            _log.warning("NavigationProvider: no known location in prompt %r", prompt)
            return
        gx, gy, gyaw = self._locations[location_id]
        with self._goal_lock:
            self._goal     = (gx, gy)
            self._goal_yaw = gyaw
            self._goal_id  = location_id
        _log.info("NavigationProvider: goal set → %s (%.2f, %.2f) yaw=%s",
                  location_id, gx, gy, f"{math.degrees(gyaw):.1f}°" if gyaw is not None else "none")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> NavigationState:
        with self._state_lock:
            return self._latest_state

    @property
    def vel_cmd(self) -> Optional[NavVelCmd]:
        state = self.get_state()
        return state.vel if state.mode == "NAVIGATING" else None

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
            self._set_state(NavigationState(t_monotonic=time.monotonic(), mode="ESTOP"))
            return

        with self._goal_lock:
            goal     = self._goal
            goal_yaw = self._goal_yaw
            goal_id  = self._goal_id

        if goal is None:
            self._path = None
            self._smooth_tx = self._smooth_ty = None
            self._cached_path = None
            self._path_tick_count = 0
            self._last_set_goal = None
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(t_monotonic=time.monotonic()))
            return

        # Get robot pose
        pose_cache = self._unitree_g1.location
        now = time.monotonic()
        if pose_cache.stale(now, self._config.pose_timeout_s):
            _log.warning("NavigationProvider: waiting for robot pose...")
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(
                t_monotonic=now, mode="WAITING_POSE", goal_id=goal_id,
            ))
            return

        pose_msg = pose_cache.value
        rx    = pose_msg.pose.position.x
        ry    = pose_msg.pose.position.y
        ryaw  = _quat_to_yaw(pose_msg.pose.orientation)

        gx, gy = goal
        dx_g = gx - rx
        dy_g = gy - ry
        dist = math.sqrt(dx_g * dx_g + dy_g * dy_g)

        if dist < self._config.arrival_tol:
            # 목표 yaw 정렬이 필요한 경우 ALIGNING 단계
            if goal_yaw is not None:
                yaw_err = _wrap_pi(goal_yaw - ryaw)
                if abs(yaw_err) > self._config.yaw_tol:
                    vyaw = max(-self._config.max_vyaw,
                               min(self._config.max_vyaw, self._config.kp_yaw * yaw_err))
                    self._unitree_g1.publish_twist(0.0, 0.0, vyaw)
                    self._set_state(NavigationState(
                        t_monotonic=now, mode="ALIGNING", goal_id=goal_id,
                        dist_to_goal=dist,
                    ))
                    return

            with self._goal_lock:
                self._goal     = None
                self._goal_yaw = None
                self._goal_id  = None
            self._path = None
            self._smooth_tx = self._smooth_ty = None
            self._cached_path = None
            self._path_tick_count = 0
            self._last_set_goal = None
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(
                t_monotonic=now, mode="ARRIVED", goal_id=goal_id,
            ))
            _log.info("NavigationProvider: arrived at %s", goal_id)
            return

        # Only call set_goal when goal actually changes to avoid unnecessary replanning
        if (gx, gy) != self._last_set_goal:
            self._pipeline.set_goal(gx, gy)
            self._last_set_goal = (gx, gy)
            self._cached_path = None
            self._path_tick_count = 0
            self._smooth_tx = self._smooth_ty = None

        # Refresh path from C++ pipeline at reduced rate to suppress oscillation
        self._path_tick_count += 1
        if self._path_tick_count >= self._config.path_update_every_n_ticks or self._cached_path is None:
            fresh = self._pipeline.get_path()
            if fresh:
                self._cached_path = fresh
            self._path_tick_count = 0

        path = self._cached_path

        # Drive toward lookahead target — stop if no path available yet
        if not path:
            self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
            self._set_state(NavigationState(
                t_monotonic=now, mode="PLANNING",
                goal_id=goal_id, dist_to_goal=dist,
            ))
            return

        tx, ty = self._get_lookahead_target(rx, ry, path)

        # EMA smoothing on lookahead target to dampen high-frequency path jitter
        alpha = self._config.target_ema_alpha
        if self._smooth_tx is None:
            self._smooth_tx, self._smooth_ty = tx, ty
        else:
            self._smooth_tx = alpha * tx + (1.0 - alpha) * self._smooth_tx
            self._smooth_ty = alpha * ty + (1.0 - alpha) * self._smooth_ty
        tx, ty = self._smooth_tx, self._smooth_ty
        dx_t = tx - rx
        dy_t = ty - ry
        td   = math.sqrt(dx_t * dx_t + dy_t * dy_t)
        if td < 0.01:
            vx_map = vy_map = 0.0
        else:
            att_speed = min(self._config.kp_xy * td, self._config.max_speed)
            vx_map    = att_speed * dx_t / td
            vy_map    = att_speed * dy_t / td

        # Map frame → body frame
        cos_y  =  math.cos(ryaw)
        sin_y  =  math.sin(ryaw)
        vx_body =  cos_y * vx_map + sin_y * vy_map
        vy_body = -sin_y * vx_map + cos_y * vy_map

        # Yaw: align with net velocity heading
        desired_hdg = math.atan2(vy_map, vx_map)
        hdg_err     = _wrap_pi(desired_hdg - ryaw)
        vyaw        = max(-self._config.max_vyaw,
                         min(self._config.max_vyaw, self._config.kp_yaw * hdg_err))

        self._unitree_g1.publish_twist(vx_body, vy_body, vyaw)
        self._set_state(NavState(
            t_monotonic=now,
            vel=NavVelCmd(vx=vx_body, vy=vy_body, vyaw=vyaw),
            mode="NAVIGATING",
            goal_id=goal_id,
            dist_to_goal=dist,
        ))

    # ------------------------------------------------------------------
    # Internal: path following
    # ------------------------------------------------------------------

    def _get_lookahead_target(self, rx: float, ry: float,
                              path: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Project robot onto path (map-frame waypoints), return the point
        `lookahead_cells` ahead. Collapses to the last waypoint near the end.
        """
        n    = len(path)
        look = self._config.lookahead_cells

        best_idx  = 0
        best_dist = float('inf')
        for i, (px, py) in enumerate(path):
            d = (px - rx)**2 + (py - ry)**2
            if d < best_dist:
                best_dist = d
                best_idx  = i

        target_idx = min(best_idx + look, n - 1)
        return path[target_idx]

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
                yaw = float(coords[2]) if len(coords) >= 3 and coords[2] is not None else None
                self._locations[name] = (float(coords[0]), float(coords[1]), yaw)
            else:
                _log.warning("NavigationProvider: invalid coords for %r: %r", name, coords)
        _log.info("NavigationProvider: loaded %d locations from %s",
                  len(self._locations), locs_path.name)

    def _parse_location(self, prompt: str) -> Optional[str]:
        lower = prompt.lower()
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
