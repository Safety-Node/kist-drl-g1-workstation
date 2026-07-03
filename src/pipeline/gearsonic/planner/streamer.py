import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.providers.nav_types import NavVelCmd
from src.pipeline.pico_vr.reader import PicoVRController, PicoVRReader
from src.providers.singleton import singleton

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "streamer_config.yaml"


def _load_config() -> dict:
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "joystick_deadzone": raw["streamer"]["joystick_deadzone"],
        "yaw_speed":         raw["streamer"]["yaw_speed"],
        "dt":                raw["streamer"]["dt"],
    }


@dataclass
class PlannerCommand:
    mode: int
    target_vel: float
    movement_direction: np.ndarray  # (3,) float32 unit vector
    facing_direction: np.ndarray    # (3,) float32 unit vector
    random_seed: int


class LocomotionMode(IntEnum):
    IDLE                = 0
    SLOW_WALK           = 1
    WALK                = 2
    RUN                 = 3
    IDLE_SQUAT          = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL          = 6
    IDLE_LYING          = 7
    IDLE_CRAWLING       = 8
    IDLE_BOXING         = 9
    WALK_BOXING         = 10
    LEFT_PUNCH          = 11
    RIGHT_PUNCH         = 12
    RANDOM_PUNCH        = 13
    ELBOW_CRAWLING      = 14
    LEFT_HOOK           = 15
    RIGHT_HOOK          = 16
    FORWARD_JUMP        = 17
    STEALTH_WALK        = 18
    INJURED_WALK        = 19


@singleton
class PlannerStreamer:

    depends_on = [PicoVRReader]

    def __init__(self):
        self._config = _load_config()

        self._lock = threading.Lock()
        self._latest_command: Optional[PlannerCommand] = None
        self._latest_input_mode: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._mode = LocomotionMode.IDLE
        self._yaw: float = 0.0
        self._prev_ab = False
        self._prev_xy = False
        self._nav_source = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="planner_streamer",
            daemon=True,
        )
        self._thread.start()
        logger.info("PlannerStreamer started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("PlannerStreamer: thread did not stop within 2s")
        self._thread = None
        logger.info("PlannerStreamer stopped")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def command(self) -> Optional[PlannerCommand]:
        with self._lock:
            return self._latest_command

    @property
    def input_mode(self) -> Optional[int]:
        with self._lock:
            return self._latest_input_mode

    def set_nav_source(self, source) -> None:
        self._nav_source = source

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _run(self) -> None:
        pico_vr_reader = PicoVRReader()
        if self._nav_source is not None:
            nav_provider = self._nav_source
        else:
            from src.providers.navigation_provider import NavigationProvider
            nav_provider = NavigationProvider()
        
        while not self._stop_event.is_set():
            pico_vr_controller = pico_vr_reader.controller
            nav_vel_cmd        = nav_provider.vel_cmd
            if pico_vr_controller is not None:
                self._update_mode(pico_vr_controller)
            mode = self._input_mode(pico_vr_controller, nav_vel_cmd)
            if mode == -1:
                cmd = self._compute_default()
                with self._lock:
                    self._latest_command = cmd
                    self._latest_input_mode = mode
                time.sleep(0.002)
                continue
            elif mode == 1:
                cmd = self._compute_from_controller(pico_vr_controller)
            else:
                cmd = self._compute_from_nav(nav_vel_cmd)
            with self._lock:
                self._latest_command = cmd
                self._latest_input_mode = mode
            time.sleep(self._config["dt"])

    def _input_mode(self, pico_vr_controller: Optional[PicoVRController], nav_vel_cmd: Optional[NavVelCmd]) -> int:
        if pico_vr_controller is not None:
            lx, ly = pico_vr_controller.left_joystick
            if math.hypot(lx, ly) > self._config["joystick_deadzone"]:
                return 1
        if nav_vel_cmd is not None:
            return 0
        return -1

    def _update_mode(self, controller: PicoVRController) -> None:
        ab_now = controller.btn_a and controller.btn_b
        xy_now = controller.btn_x and controller.btn_y
        if ab_now and not self._prev_ab:
            self._mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self._mode + 1))
        if xy_now and not self._prev_xy:
            self._mode = LocomotionMode(max(LocomotionMode.IDLE, self._mode - 1))
        self._prev_ab = ab_now
        self._prev_xy = xy_now

    def _compute_from_controller(self, pico_vr_controller: PicoVRController) -> PlannerCommand:
        lx, ly = pico_vr_controller.left_joystick
        rx, _  = pico_vr_controller.right_joystick
        dt = self._config["dt"]

        self._yaw += rx * self._config["yaw_speed"] * dt
        facing = np.array([math.cos(self._yaw), math.sin(self._yaw)], dtype=np.float32)

        raw_mag = float(np.clip(math.hypot(lx, ly), 0.0, 1.0))
        mag = (raw_mag - self._config["joystick_deadzone"]) / (1.0 - self._config["joystick_deadzone"])
        mode_to_send = self._mode
        if self._mode == LocomotionMode.SLOW_WALK:
            target_vel = 0.1 + 0.5 * mag
        elif self._mode == LocomotionMode.WALK:
            target_vel = -1.0
        elif self._mode == LocomotionMode.RUN:
            target_vel = 1.5 + 3.0 * mag
        else:
            target_vel = mag
        scale = mag / raw_mag
        movement_local = np.array([-lx, ly], dtype=np.float32) * scale
        perp = np.array([-facing[1], facing[0]], dtype=np.float32)
        movement_xy = np.stack([perp, facing], axis=1) @ movement_local
        movement_direction = np.array([movement_xy[0], movement_xy[1], 0.0], dtype=np.float32)
        facing_direction = np.array([facing[0], facing[1], 0.0], dtype=np.float32)

        return PlannerCommand(
            mode=mode_to_send,
            target_vel=float(target_vel),
            movement_direction=movement_direction,
            facing_direction=facing_direction,
            random_seed=0,
        )

    def _compute_from_nav(self, nav_vel_cmd: NavVelCmd) -> PlannerCommand:
        dt = self._config["dt"]

        self._yaw += nav_vel_cmd.vyaw * dt
        facing = np.array([math.cos(self._yaw), math.sin(self._yaw)], dtype=np.float32)

        vx, vy = nav_vel_cmd.vx, nav_vel_cmd.vy
        speed = float(math.hypot(vx, vy))
        if speed < 1e-3:
            movement_direction = np.array([facing[0], facing[1], 0.0], dtype=np.float32)
            target_vel = -1.0
            mode_to_send = LocomotionMode.IDLE
        else:
            movement_direction = np.array([vx / speed, vy / speed, 0.0], dtype=np.float32)
            target_vel = speed
            mode_to_send = self._mode
        facing_direction = np.array([facing[0], facing[1], 0.0], dtype=np.float32)

        return PlannerCommand(
            mode=mode_to_send,
            target_vel=float(target_vel),
            movement_direction=movement_direction,
            facing_direction=facing_direction,
            random_seed=0,
        )

    def _compute_default(self) -> PlannerCommand:
        facing = np.array([math.cos(self._yaw), math.sin(self._yaw)], dtype=np.float32)
        direction = np.array([facing[0], facing[1], 0.0], dtype=np.float32)
        return PlannerCommand(
            mode=self._mode,
            target_vel=-1.0,
            movement_direction=direction,
            facing_direction=direction,
            random_seed=0,
        )
