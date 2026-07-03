import math

import numpy as np

from src.providers.gearsonic.planner.types import PlannerCommand
from src.providers.pico_vr.reader import PicoVRController

_JOYSTICK_DEADZONE = 0.1
_YAW_SPEED = 1.5  # rad/s at rx=1.0

_MODE_IDLE        = 0
_MODE_SLOW_WALK   = 1
_MODE_WALK        = 2
_MODE_RUN         = 3
_MODE_MAX         = 19


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
        self._mode = _MODE_IDLE
        self._yaw = _YawAccumulator()
        self._prev_ab = False
        self._prev_xy = False

    def reset(self) -> None:
        self._yaw.reset()
        self._mode = _MODE_IDLE
        self._prev_ab = False
        self._prev_xy = False

    def update(self, controller: PicoVRController) -> PlannerCommand:
        # --- mode switching (rising edge) ---
        ab_now = controller.btn_a and controller.btn_b
        xy_now = controller.btn_x and controller.btn_y
        if ab_now and not self._prev_ab:
            self._mode = min(_MODE_MAX, self._mode + 1)
        if xy_now and not self._prev_xy:
            self._mode = max(_MODE_IDLE, self._mode - 1)
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
            mode_to_send = _MODE_IDLE
        else:
            mag = (raw_mag - _JOYSTICK_DEADZONE) / (1.0 - _JOYSTICK_DEADZONE)
            mode_to_send = self._mode
            if self._mode == _MODE_SLOW_WALK:
                target_vel = 0.1 + 0.5 * mag
            elif self._mode == _MODE_WALK:
                target_vel = -1.0
            elif self._mode == _MODE_RUN:
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
