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

from src.providers.pico_vr.reader import PicoVRController, PicoVRReader
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


class _YawAccumulator:

    def __init__(self, yaw_speed: float):
        self._yaw = 0.0
        self._yaw_speed = yaw_speed

    def reset(self) -> None:
        self._yaw = 0.0

    def update(self, rx: float, dt: float) -> np.ndarray:
        self._yaw += rx * self._yaw_speed * dt
        return np.array([math.cos(self._yaw), math.sin(self._yaw)], dtype=np.float32)


@singleton
class PlannerStreamer:

    depends_on = [PicoVRReader]

    def __init__(self):
        self._config = _load_config()
        self._mode = LocomotionMode.IDLE
        self._yaw = _YawAccumulator(self._config["yaw_speed"])
        self._prev_ab = False
        self._prev_xy = False

        self._lock = threading.Lock()
        self._latest_command: Optional[PlannerCommand] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ready_event = threading.Event()

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
        reader = PicoVRReader()
        while not self._stop_event.is_set():
            controller = reader.controller
            if controller is None:
                time.sleep(self._config["dt"])
                continue

            cmd = self._compute(controller)
            with self._lock:
                self._latest_command = cmd
                if not self.ready_event.is_set():
                    self.ready_event.set()

    def _compute(self, controller: PicoVRController) -> PlannerCommand:
        # --- mode switching (rising edge) ---
        ab_now = controller.btn_a and controller.btn_b
        xy_now = controller.btn_x and controller.btn_y
        if ab_now and not self._prev_ab:
            self._mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self._mode + 1))
        if xy_now and not self._prev_xy:
            self._mode = LocomotionMode(max(LocomotionMode.IDLE, self._mode - 1))
        self._prev_ab = ab_now
        self._prev_xy = xy_now

        lx, ly = controller.left_joystick
        rx, _  = controller.right_joystick

        # --- facing direction ---
        facing = self._yaw.update(rx, self._config["dt"])  # (2,) unit vector

        # --- movement direction + speed ---
        raw_mag = float(np.clip(math.hypot(lx, ly), 0.0, 1.0))
        if raw_mag < self._config["joystick_deadzone"]:
            mag = 0.0
            target_vel = -1.0
            mode_to_send = LocomotionMode.IDLE
        else:
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

        scale = mag / raw_mag if raw_mag > 0.0 else 0.0
        movement_local = np.array([-lx, ly], dtype=np.float32) * scale

        # rotate movement from local (facing) frame to global frame
        perp = np.array([-facing[1], facing[0]], dtype=np.float32)
        rotation = np.stack([perp, facing], axis=1)   # (2, 2)
        movement_xy = rotation @ movement_local        # (2,)

        movement_direction = np.array([movement_xy[0], movement_xy[1], 0.0], dtype=np.float32)
        facing_direction   = np.array([facing[0], facing[1], 0.0], dtype=np.float32)

        return PlannerCommand(
            mode=mode_to_send,
            target_vel=float(target_vel),
            movement_direction=movement_direction,
            facing_direction=facing_direction,
            random_seed=0,
        )
