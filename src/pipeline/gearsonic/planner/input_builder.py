"""
PlannerInputBuilder — assembles ONNX input tensors for the GearSonic locomotion planner.

context_mujoco_qpos layout (per frame, 36 values):
    [0:3]   x, y, z          — root position (m), normalized to [0, 0, height] at init
    [3:7]   qw, qx, qy, qz   — root quaternion (world frame)
    [7:36]  joint positions   — 29 joints in isaaclab order

Joints in context are stored in isaaclab order (matches C++ UpdateContextFromMotion).
Planner ONNX output mujoco_qpos has joints in isaaclab order at [7:36].
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as sRot, Slerp

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "input_builder_config.yaml"
_ROBOT_HZ    = 50  # robot playback rate for gen_frame


def _load_config() -> dict:
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "default_height":          raw["planner"]["default_height"],
        "allowed_pred_num_tokens": raw["planner"]["allowed_pred_num_tokens"],
        "mujoco_to_isaaclab":      raw["planner"]["mujoco_to_isaaclab"],
    }


class PlannerInputBuilder:

    def __init__(self):
        self._config = _load_config()
        self._mujoco_to_isaaclab = self._config["mujoco_to_isaaclab"]
        self._context = np.zeros((4, 36), dtype=np.float32)
        self._target_vel: float = 0.0
        self._mode: int = 0
        self._movement_direction = np.zeros(3, dtype=np.float32)
        self._facing_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._random_seed: int = 0

        self._trajectory: Optional[np.ndarray] = None  # (N, 36) float32
        self._traj_time:  Optional[float] = None        # monotonic timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_command(
        self,
        mode: int,
        target_vel: float,
        movement_direction: np.ndarray,
        facing_direction: np.ndarray,
        random_seed: int,
    ) -> None:
        self._mode = mode
        self._target_vel = target_vel
        self._movement_direction = np.asarray(movement_direction, dtype=np.float32)
        self._facing_direction = np.asarray(facing_direction, dtype=np.float32)
        self._random_seed = random_seed

    def initialize(self, joint_pos_mujoco: np.ndarray) -> None:
        """
        Fill context_mujoco_qpos with current robot state repeated 4 times.
        Matches C++ InitializeContext.

        Input:
            joint_pos_mujoco  (29,) float32 — joint positions in mujoco order

        Process:
            pos  = [0, 0, default_height]   — normalized origin
            quat = [1, 0, 0, 0]             — identity (zero yaw)
            joints[isaac_i] = joint_pos[mujoco_i]   — mujoco→isaaclab remap
            context[0..3] = frame           — 4 identical frames

        Output:
            self._context  (4, 36) float32 updated in-place
        """
        frame = np.zeros(36, dtype=np.float32)
        frame[2] = self._config["default_height"]
        frame[3] = 1.0  # qw
        for mujoco_i, isaac_i in enumerate(self._mujoco_to_isaaclab):
            frame[7 + isaac_i] = joint_pos_mujoco[mujoco_i]

        for n in range(4):
            self._context[n] = frame

        self._trajectory = None
        self._traj_time  = None
        logger.info("PlannerInputBuilder: context initialized")

    def set_trajectory(self, trajectory: np.ndarray) -> None:
        """Store ONNX output trajectory so build() can update context on the next call."""
        self._trajectory = trajectory
        self._traj_time  = time.monotonic()

    def build(self) -> dict:
        """
        Update context from stored trajectory (if any), then return all planner
        ONNX inputs as a dict ready for session.run().

        Output keys and shapes:
            context_mujoco_qpos      [1, 4, 36] float32
            target_vel               [1]        float32
            mode                     [1]        int64
            movement_direction       [1, 3]     float32
            facing_direction         [1, 3]     float32
            random_seed              [1]        int64
            has_specific_target      [1, 1]     int64
            specific_target_positions[1, 4, 3]  float32
            specific_target_headings [1, 4]     float32
            allowed_pred_num_tokens  [1, 11]    int64
            height                   [1]        float32
        """
        if self._trajectory is not None:
            gen_frame = int((time.monotonic() - self._traj_time) * _ROBOT_HZ)
            self._update_context(self._trajectory, gen_frame)

        return {
            "context_mujoco_qpos":       self._context[np.newaxis].copy(),
            "target_vel":                np.array([self._target_vel], dtype=np.float32),
            "mode":                      np.array([self._mode], dtype=np.int64),
            "movement_direction":        self._movement_direction[np.newaxis].copy(),
            "facing_direction":          self._facing_direction[np.newaxis].copy(),
            "random_seed":               np.array([self._random_seed], dtype=np.int64),
            "has_specific_target":       np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings":  np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens":   np.array([self._config["allowed_pred_num_tokens"]], dtype=np.int64),
            "height":                    np.array([-1.0], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_context(self, trajectory: np.ndarray, gen_frame: int) -> None:
        """
        Resample previous planner trajectory into context_mujoco_qpos.
        Matches C++ UpdateContextFromMotion.

        Input:
            trajectory  (N, 36) float32 — planner mujoco_qpos output (N ≤ 64, 30Hz)
                        [0:3] pos, [3:7] quat wxyz, [7:36] joints in isaaclab order
            gen_frame   int — current playback frame index (50Hz counter)

        Process:
            For each of 4 context frames n=0..3:
                t      = gen_frame/50 + n/30     — sample time in seconds
                f_30hz = t * 30                  — fractional 30Hz frame index
                f0, f1 = floor/ceil of f_30hz    — clamped to [0, N-1]
                w0     = 1 - (f_30hz - f0)

                pos    = lerp(traj[f0, 0:3], traj[f1, 0:3], w1)
                quat   = slerp(traj[f0, 3:7], traj[f1, 3:7], f_30hz - f0)
                joints = lerp(traj[f0, 7:], traj[f1, 7:], w1)
        """
        T        = len(trajectory)
        gen_time = gen_frame / 50.0

        for n in range(4):
            t      = gen_time + n / 30.0
            f_30hz = t * 30.0
            f0     = min(int(np.floor(f_30hz)), T - 1)
            f1     = min(f0 + 1, T - 1)
            w0     = float(1.0 - (f_30hz - f0))
            w1     = float(1.0 - w0)

            self._context[n, 0:3] = w0 * trajectory[f0, 0:3] + w1 * trajectory[f1, 0:3]
            self._context[n, 3:7] = _quat_slerp(
                trajectory[f0, 3:7], trajectory[f1, 3:7], f_30hz - f0
            )
            self._context[n, 7:] = w0 * trajectory[f0, 7:] + w1 * trajectory[f1, 7:]


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between scalar-first quats [qw,qx,qy,qz]. Returns scalar-first float32."""
    r0    = sRot.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r1    = sRot.from_quat([q1[1], q1[2], q1[3], q1[0]])
    slerp = Slerp([0.0, 1.0], sRot.concatenate([r0, r1]))
    q     = slerp(float(np.clip(t, 0.0, 1.0))).as_quat()  # scalar-last
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
