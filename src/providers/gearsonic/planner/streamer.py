import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from src.providers.pico_vr.reader import PicoVRController


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


@dataclass
class PlannerCommand:
    mode: int
    target_vel: float
    movement_direction: np.ndarray  # (3,) float32 unit vector
    facing_direction: np.ndarray    # (3,) float32 unit vector
    random_seed: int


_JOYSTICK_DEADZONE = 0.15
_YAW_SPEED = 1.5  # rad/s at rx=1.0


class _YawAccumulator:

    def __init__(self):
        self._yaw = 0.0

    def reset(self) -> None:
        self._yaw = 0.0

    def update(self, rx: float, dt: float) -> np.ndarray:
        self._yaw += rx * _YAW_SPEED * dt
        return np.array([math.cos(self._yaw), math.sin(self._yaw)], dtype=np.float32)


class PlannerStreamer:

    depends_on = [PicoVRController]

    def __init__(self, dt: float = 0.05):
        self._dt = dt
        self._mode = LocomotionMode.IDLE
        self._yaw = _YawAccumulator()
        self._prev_ab = False
        self._prev_xy = False

    def reset(self) -> None:
        self._yaw.reset()
        self._mode = LocomotionMode.IDLE
        self._prev_ab = False
        self._prev_xy = False

    def update(self, controller: PicoVRController) -> PlannerCommand:
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
        facing = self._yaw.update(rx, self._dt)  # (2,) unit vector

        # --- movement direction + speed ---
        raw_mag = float(np.clip(math.hypot(lx, ly), 0.0, 1.0))
        if raw_mag < _JOYSTICK_DEADZONE:
            mag = 0.0
            target_vel = -1.0
            mode_to_send = LocomotionMode.IDLE
        else:
            mag = (raw_mag - _JOYSTICK_DEADZONE) / (1.0 - _JOYSTICK_DEADZONE)
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
