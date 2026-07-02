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
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as sRot, Slerp

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "planner_input_config.yaml"


def _load_config() -> dict:
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "default_height":          raw["planner"]["default_height"],
        "allowed_pred_num_tokens": raw["planner"]["allowed_pred_num_tokens"],
    }


class PlannerInputBuilder:

    # policy_parameters.hpp: mujoco_to_isaaclab[mujoco_i] = isaaclab_i
    _MUJOCO_TO_ISAACLAB = [
        0, 6, 12, 1,  7, 13, 2,  8, 14,
        3, 9, 15, 22, 4, 10, 16, 23, 5,
        11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ]

    def __init__(self):
        self._config = _load_config()
        self._context = np.zeros((4, 36), dtype=np.float32)

        # operator-controlled inputs — set externally before build()
        self.target_vel: float = 0.0
        self.mode: int = 0
        self.movement_direction = np.zeros(3, dtype=np.float32)
        self.facing_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.random_seed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        frame[0] = 0.0
        frame[1] = 0.0
        frame[2] = self._config["default_height"]
        frame[3] = 1.0   # qw
        frame[4] = 0.0   # qx
        frame[5] = 0.0   # qy
        frame[6] = 0.0   # qz
        for mujoco_i, isaac_i in enumerate(self._MUJOCO_TO_ISAACLAB):
            frame[7 + isaac_i] = joint_pos_mujoco[mujoco_i]

        for n in range(4):
            self._context[n] = frame

        logger.info("PlannerInputBuilder: context initialized")

    def update_from_trajectory(self, trajectory: np.ndarray, gen_frame: int) -> None:
        """
        Update context_mujoco_qpos by resampling previous planner trajectory.
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
                joints = lerp(traj[f0, 7:], traj[f1, 7:], w1)   — isaaclab order, direct copy

        Output:
            self._context  (4, 36) float32 updated in-place
        """
        T = len(trajectory)
        gen_time = gen_frame / 50.0

        for n in range(4):
            t = gen_time + n / 30.0
            f_30hz = t * 30.0
            f0 = min(int(np.floor(f_30hz)), T - 1)
            f1 = min(f0 + 1, T - 1)
            w0 = float(1.0 - (f_30hz - f0))
            w1 = float(1.0 - w0)

            # position
            self._context[n, 0:3] = w0 * trajectory[f0, 0:3] + w1 * trajectory[f1, 0:3]

            # quaternion slerp [qw, qx, qy, qz]
            self._context[n, 3:7] = _quat_slerp(
                trajectory[f0, 3:7], trajectory[f1, 3:7], f_30hz - f0
            )

            # joints: planner output is isaaclab order → direct copy
            self._context[n, 7:] = w0 * trajectory[f0, 7:] + w1 * trajectory[f1, 7:]

    def build(self) -> dict:
        """
        Return all planner ONNX inputs as a dict ready for session.run().

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
        return {
            "context_mujoco_qpos":       self._context[np.newaxis].copy(),
            "target_vel":                np.array([self.target_vel], dtype=np.float32),
            "mode":                      np.array([self.mode], dtype=np.int64),
            "movement_direction":        self.movement_direction[np.newaxis].copy(),
            "facing_direction":          self.facing_direction[np.newaxis].copy(),
            "random_seed":               np.array([self.random_seed], dtype=np.int64),
            "has_specific_target":       np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings":  np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens":   np.array([self._config["allowed_pred_num_tokens"]], dtype=np.int64),
            "height":                    np.array([-1.0], dtype=np.float32),
        }

# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Slerp between two scalar-first quaternions [qw, qx, qy, qz].
    Returns scalar-first [qw, qx, qy, qz] float32.
    """
    # scipy uses scalar-last [qx, qy, qz, qw]
    r0 = sRot.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r1 = sRot.from_quat([q1[1], q1[2], q1[3], q1[0]])
    slerp = Slerp([0.0, 1.0], sRot.concatenate([r0, r1]))
    q = slerp(float(np.clip(t, 0.0, 1.0))).as_quat()  # scalar-last
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
