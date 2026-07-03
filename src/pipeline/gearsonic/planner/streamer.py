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
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

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

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _run(self) -> None:
        pico_vr_reader = PicoVRReader()
        ctrl_builder   = ControllerCommandBuilder(self._config)

        while not self._stop_event.is_set():
            ctrl = pico_vr_reader.controller
            if ctrl is not None:
                ctrl_builder.update_mode(ctrl)
                cmd = ctrl_builder.compute(ctrl) if ctrl_builder.is_active(ctrl) else ctrl_builder.default()
            else:
                cmd = ctrl_builder.default()
            with self._lock:
                self._latest_command = cmd
            time.sleep(self._config["dt"])


class ControllerCommandBuilder:
    """Converts PicoVR controller input → PlannerCommand."""

    def __init__(self, config: dict):
        self._deadzone  = config["joystick_deadzone"]
        self._yaw_speed = config["yaw_speed"]
        self._dt        = config["dt"]

        self._mode    = LocomotionMode.IDLE
        self._yaw     = 0.0
        self._prev_ab = False
        self._prev_xy = False

    def is_active(self, ctrl: PicoVRController) -> bool:
        lx, ly = ctrl.left_joystick
        rx, _  = ctrl.right_joystick
        return math.hypot(lx, ly) > self._deadzone or abs(rx) > self._deadzone

    def update_mode(self, ctrl: PicoVRController) -> None:
        ab_now = ctrl.btn_a and ctrl.btn_b
        xy_now = ctrl.btn_x and ctrl.btn_y
        if ab_now and not self._prev_ab:
            self._mode = LocomotionMode(min(LocomotionMode.RUN, self._mode + 1))
        if xy_now and not self._prev_xy:
            self._mode = LocomotionMode(max(LocomotionMode.IDLE, self._mode - 1))
        self._prev_ab = ab_now
        self._prev_xy = xy_now

    def compute(self, ctrl: PicoVRController) -> PlannerCommand:
        if self._mode == LocomotionMode.IDLE:
            return self.default()

        lx, ly = ctrl.left_joystick
        rx, _  = ctrl.right_joystick

        self._yaw += rx * self._yaw_speed * self._dt
        facing = np.array([math.cos(self._yaw), math.sin(self._yaw)], dtype=np.float32)

        raw_mag = float(np.clip(math.hypot(lx, ly), 0.0, 1.0))
        if raw_mag < 1e-6:
            movement_direction = np.array([facing[0], facing[1], 0.0], dtype=np.float32)
            mag = 0.0
        else:
            mag   = (raw_mag - self._deadzone) / (1.0 - self._deadzone)
            scale = mag / raw_mag
            movement_local = np.array([-lx, ly], dtype=np.float32) * scale
            perp = np.array([-facing[1], facing[0]], dtype=np.float32)
            movement_xy = np.stack([perp, facing], axis=1) @ movement_local
            movement_direction = np.array([movement_xy[0], movement_xy[1], 0.0], dtype=np.float32)

        return PlannerCommand(
            mode=self._mode,
            target_vel=-1.0,
            movement_direction=movement_direction,
            facing_direction=np.array([facing[0], facing[1], 0.0], dtype=np.float32),
            random_seed=0,
        )

    def default(self) -> PlannerCommand:
        facing = np.array([math.cos(self._yaw), math.sin(self._yaw)], dtype=np.float32)
        direction = np.array([facing[0], facing[1], 0.0], dtype=np.float32)
        return PlannerCommand(
            mode=self._mode,
            target_vel=-1.0,
            movement_direction=direction,
            facing_direction=direction,
            random_seed=0,
        )
