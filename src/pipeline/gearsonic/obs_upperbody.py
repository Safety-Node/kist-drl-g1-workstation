"""
UpperbodyObsBuilder — 1762-dim encoder obs vector for upperbody teleop (mode 1).

Reads robot state directly from UnitreeG1Provider and SMPL data from PicoVRBodyPose.
Returns np.ndarray (1762,) float32, ready to pass to the encoder ONNX session.

Encoder input layout (mode 1 fields only, rest are zero):
    [0:4]     encoder_mode_4              = [0,1,0,0]
    [595:601] motion_anchor_orientation   = 6D(inv(robot_base) * apply_delta_heading * smpl_root)
    [661:781] lowerbody joint pos history = 10x12, step-5
    [781:901] lowerbody joint vel history = 10x12, step-5
    [901:910] vr_3point_local_target      = [lw, rw, neck] positions (3x3)
    [910:922] vr_3point_local_orn_target  = [lw, rw, neck] quats scalar-first (3x4)
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from src.pipeline.pico_vr.reader import PicoVRBodyPose
from .obs_base import ObsBuilderBase

logger = logging.getLogger(__name__)


@dataclass
class _CalibState:
    neck_inv_rot: sRot
    lwrist_pos_offset: np.ndarray   # (3,)
    rwrist_pos_offset: np.ndarray   # (3,)
    lwrist_rot_offset: sRot
    rwrist_rot_offset: sRot
    apply_delta_heading: sRot       # yaw(init_base) * inv(yaw(init_smpl_root))


class UpperbodyObsBuilder(ObsBuilderBase):
    """
    Builds the 1762-dim encoder input for upperbody teleop (mode 1).

    Reads robot state directly from UnitreeG1Provider and SMPL from PicoVRBodyPose.
    Maintains its own lower body history buffer (step-5, 50 frames).

    Usage:
        builder = UpperbodyObsBuilder(
            g1_provider,
            g1_lwrist_ref=np.array([...]),   # G1 FK at zero pose (optional)
            g1_rwrist_ref=np.array([...]),
            g1_lwrist_rot=sRot(...),
            g1_rwrist_rot=sRot(...),
        )
        builder.calibrate(body_pose)          # operator in zero-reference pose
        obs = builder.build(body_pose)        # np.ndarray (1762,) or None
    """

    _MODE_ONEHOT = np.array([0, 1, 0, 0], dtype=np.float32)

    _TORSO_Z_OFFSET = 0.05   # m
    _NECK_LENGTH    = 0.35   # m

    _LOWER_JOINT_NAMES = [
        "left_hip_pitch",  "left_hip_roll",  "left_hip_yaw",
        "left_knee",       "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee",      "right_ankle_pitch", "right_ankle_roll",
    ]
    _HIS_LEN  = 50
    _N_FRAMES = 10
    _STEP     = 5

    def __init__(
        self,
        g1_provider,
        g1_lwrist_ref: Optional[np.ndarray] = None,
        g1_rwrist_ref: Optional[np.ndarray] = None,
        g1_lwrist_rot: Optional[sRot] = None,
        g1_rwrist_rot: Optional[sRot] = None,
    ):
        """
        Args:
            g1_provider: UnitreeG1Provider instance (must be started).
            g1_lwrist_ref: G1 left wrist position (3,) in robot frame at zero pose.
                           If None, wrist position calibration is skipped.
            g1_rwrist_ref: G1 right wrist position (3,) in robot frame at zero pose.
            g1_lwrist_rot: G1 left wrist orientation at zero pose.
            g1_rwrist_rot: G1 right wrist orientation at zero pose.
        """
        self._g1 = g1_provider
        self._g1_lwrist_ref = g1_lwrist_ref
        self._g1_rwrist_ref = g1_rwrist_ref
        self._g1_lwrist_rot = g1_lwrist_rot
        self._g1_rwrist_rot = g1_rwrist_rot

        self._pos_lower_history: deque = deque(maxlen=self._HIS_LEN)
        self._vel_lower_history: deque = deque(maxlen=self._HIS_LEN)
        self._joint_name_to_idx: Optional[dict] = None

        self._calib: Optional[_CalibState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calib is not None

    def calibrate(self, body_pose: PicoVRBodyPose) -> bool:
        """
        Capture calibration from current SMPL frame.
        Operator must be in zero-reference pose (차렷) when called.

        Returns True on success.
        """
        try:
            robot_state = self._read_robot_state()
            base_quat_wxyz = robot_state[2] if robot_state is not None \
                else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            raw = self._extract_3pt_pose(body_pose.body_poses_np)
            self._calib = self._compute_calib(raw, body_pose.body_poses_np, base_quat_wxyz)
            logger.info("UpperbodyObsBuilder: calibration captured")
            return True
        except Exception:
            logger.exception("UpperbodyObsBuilder: calibration failed")
            return False

    def build(self, body_pose: PicoVRBodyPose) -> Optional[np.ndarray]:
        """
        Build the 1762-dim encoder input.

        Reads robot state from UnitreeG1Provider internally and updates
        lower body history buffers. Returns None if not calibrated or
        robot state is unavailable.
        """
        if self._calib is None:
            return None

        robot_state = self._read_robot_state()
        if robot_state is None:
            return None

        pos_lower, vel_lower, base_quat_wxyz = robot_state

        self._pos_lower_history.appendleft(pos_lower)
        self._vel_lower_history.appendleft(vel_lower)

        smpl = body_pose.body_poses_np

        raw = self._extract_3pt_pose(smpl)
        vr_3pt = self._apply_calib(raw)  # (3, 7)

        anchor_6d = self._anchor_6d(smpl, base_quat_wxyz, self._calib.apply_delta_heading)

        obs = np.zeros(1762, dtype=np.float32)
        obs[0:4]     = self._MODE_ONEHOT
        obs[595:601] = anchor_6d
        obs[661:781] = self._sample_history(self._pos_lower_history, self._N_FRAMES, self._STEP)
        obs[781:901] = self._sample_history(self._vel_lower_history, self._N_FRAMES, self._STEP)
        obs[901:910] = vr_3pt[:, :3].flatten()
        obs[910:922] = vr_3pt[:, 3:].flatten()
        return obs

    # ------------------------------------------------------------------
    # Robot state reading
    # ------------------------------------------------------------------

    def _read_robot_state(self) -> Optional[tuple]:
        """
        Read current lower body joint positions/velocities and base IMU quat.

        Returns (pos_lower (12,), vel_lower (12,), base_quat_wxyz (4,)) or None.
        """
        js_cache = self._g1.joint_state
        imu_cache = self._g1.imu_base

        if js_cache.last_seen_ts == 0.0 or js_cache.value is None:
            return None

        js = js_cache.value

        if self._joint_name_to_idx is None:
            self._joint_name_to_idx = {n: i for i, n in enumerate(js.name)}
            missing = [n for n in self._LOWER_JOINT_NAMES if n not in self._joint_name_to_idx]
            if missing:
                logger.warning("UpperbodyObsBuilder: missing joints: %s", missing)

        try:
            pos_lower = np.array([
                js.position[self._joint_name_to_idx[n]] for n in self._LOWER_JOINT_NAMES
            ], dtype=np.float32)
            vel_lower = np.array([
                js.velocity[self._joint_name_to_idx[n]] for n in self._LOWER_JOINT_NAMES
            ], dtype=np.float32)
        except (KeyError, IndexError):
            self._joint_name_to_idx = None
            return None

        base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity fallback
        if imu_cache.last_seen_ts != 0.0 and imu_cache.value is not None:
            q = imu_cache.value.orientation
            base_quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)

        return pos_lower, vel_lower, base_quat_wxyz

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _compute_calib(
        self, raw: np.ndarray, smpl: np.ndarray, base_quat_wxyz: np.ndarray
    ) -> _CalibState:
        neck_rot = sRot.from_quat(raw[2, 3:], scalar_first=True)
        neck_inv = neck_rot.inv()

        lw_pos = neck_inv.apply(raw[0, :3])
        rw_pos = neck_inv.apply(raw[1, :3])
        lw_rot = neck_inv * sRot.from_quat(raw[0, 3:], scalar_first=True)
        rw_rot = neck_inv * sRot.from_quat(raw[1, 3:], scalar_first=True)

        lw_offset = lw_pos - self._g1_lwrist_ref if self._g1_lwrist_ref is not None \
            else np.zeros(3, dtype=np.float32)
        rw_offset = rw_pos - self._g1_rwrist_ref if self._g1_rwrist_ref is not None \
            else np.zeros(3, dtype=np.float32)

        lw_rot_offset = self._g1_lwrist_rot * lw_rot.inv() if self._g1_lwrist_rot is not None \
            else sRot.identity()
        rw_rot_offset = self._g1_rwrist_rot * rw_rot.inv() if self._g1_rwrist_rot is not None \
            else sRot.identity()

        return _CalibState(
            neck_inv_rot=neck_inv,
            lwrist_pos_offset=lw_offset,
            rwrist_pos_offset=rw_offset,
            lwrist_rot_offset=lw_rot_offset,
            rwrist_rot_offset=rw_rot_offset,
            apply_delta_heading=self._compute_delta_heading(smpl, base_quat_wxyz),
        )

    def _apply_calib(self, raw: np.ndarray) -> np.ndarray:
        c = self._calib
        out = raw.copy()

        neck_rot = sRot.from_quat(raw[2, 3:], scalar_first=True)
        out[2, 3:] = (c.neck_inv_rot * neck_rot).as_quat(scalar_first=True)

        out[0, :3] = c.neck_inv_rot.apply(raw[0, :3]) - c.lwrist_pos_offset
        out[1, :3] = c.neck_inv_rot.apply(raw[1, :3]) - c.rwrist_pos_offset

        lw = c.neck_inv_rot * sRot.from_quat(raw[0, 3:], scalar_first=True)
        out[0, 3:] = (c.lwrist_rot_offset * lw).as_quat(scalar_first=True)
        rw = c.neck_inv_rot * sRot.from_quat(raw[1, 3:], scalar_first=True)
        out[1, 3:] = (c.rwrist_rot_offset * rw).as_quat(scalar_first=True)

        neck_z = sRot.from_quat(out[2, 3:], scalar_first=True).apply([0, 0, 1])
        out[2, :3] = (
            np.array([0, 0, self._TORSO_Z_OFFSET]) + self._NECK_LENGTH * neck_z
        ).astype(np.float32)

        return out
