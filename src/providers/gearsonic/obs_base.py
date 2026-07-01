"""
ObsBuilderBase — shared math and utilities for GearSonic obs builders.

Subclasses:
    UpperbodyObsBuilder  (obs_upperbody.py) — mode 1, VR 3-point teleop
    G1ObsBuilder         (obs_g1.py)        — mode 0, motion imitation (TODO)
"""

import numpy as np
from scipy.spatial.transform import Rotation as sRot


class ObsBuilderBase:

    # Unity (Y-up, left-handed) → Robot (Z-up, right-handed)
    # Unity [x, y, z] → Robot [-x, z, y]
    _Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)

    # SMPL joint indices
    _SMPL_ROOT   = 0
    _SMPL_LWRIST = 22
    _SMPL_RWRIST = 23
    _SMPL_NECK   = 12

    # Rotation offsets per keypoint (Root, L-Wrist, R-Wrist, Neck)
    _OFFSETS = [
        sRot.from_euler("xyz", [0,   0,   -90], degrees=True),
        sRot.from_euler("xyz", [90,  0,     0], degrees=True),
        sRot.from_euler("xyz", [-90, 0,   180], degrees=True),
        sRot.from_euler("xyz", [0,   0,   -90], degrees=True),
    ]

    # ------------------------------------------------------------------
    # Frame transform
    # ------------------------------------------------------------------

    @staticmethod
    def _unity_to_robot(quat_unity_scalar_last: np.ndarray) -> sRot:
        """Unity scalar-last quat → robot-frame Rotation."""
        rot = sRot.from_quat(quat_unity_scalar_last, scalar_first=False)
        Q = ObsBuilderBase._Q
        return sRot.from_matrix(Q @ rot.as_matrix() @ Q.T)

    @staticmethod
    def _smpl_root_to_robot(smpl: np.ndarray) -> sRot:
        """SMPL joint 0 orientation in robot world frame (position ignored)."""
        return ObsBuilderBase._unity_to_robot(smpl[ObsBuilderBase._SMPL_ROOT, 3:])

    # ------------------------------------------------------------------
    # 3-point pose extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_3pt_pose(smpl: np.ndarray) -> np.ndarray:
        """
        Convert SMPL (24, 7) Unity frame → (3, 7) robot frame, root-relative.

        Returns [L-Wrist, R-Wrist, Neck] as (3, 7) float32, scalar-first quat.
        """
        smpl = smpl.copy()
        cls = ObsBuilderBase
        kp_indices = [cls._SMPL_ROOT, cls._SMPL_LWRIST, cls._SMPL_RWRIST, cls._SMPL_NECK]
        kp = np.zeros((4, 7), dtype=np.float64)

        for kp_i, smpl_j in enumerate(kp_indices):
            pos = cls._Q @ smpl[smpl_j, :3]
            rot = cls._unity_to_robot(smpl[smpl_j, 3:]) * cls._OFFSETS[kp_i]
            kp[kp_i, :3] = pos
            kp[kp_i, 3:] = rot.as_quat(scalar_first=False)

        root_pos = kp[0, :3].copy()
        root_rot = sRot.from_quat(kp[0, 3:], scalar_first=False)

        for i in range(1, 4):
            kp[i, :3] = root_rot.inv().apply(kp[i, :3] - root_pos)
            kp[i, 3:] = (
                root_rot.inv() * sRot.from_quat(kp[i, 3:], scalar_first=False)
            ).as_quat(scalar_first=True)

        return kp[1:].astype(np.float32)  # (3, 7) scalar-first

    # ------------------------------------------------------------------
    # Anchor orientation
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_heading_quat(rot: sRot) -> sRot:
        """Yaw-only rotation (matches C++ calc_heading_quat_d)."""
        fwd = rot.apply([1.0, 0.0, 0.0])
        return sRot.from_euler("z", np.arctan2(fwd[1], fwd[0]))

    @staticmethod
    def _compute_delta_heading(smpl: np.ndarray, base_quat_wxyz: np.ndarray) -> sRot:
        """
        apply_delta_heading = yaw(init_base) * inv(yaw(init_smpl_root)).
        Matches C++ ComputeApplyDeltaHeading — call once at calibration time.
        """
        robot_base = sRot.from_quat(base_quat_wxyz, scalar_first=True)
        init_heading = ObsBuilderBase._calc_heading_quat(robot_base)
        smpl_root = ObsBuilderBase._smpl_root_to_robot(smpl)
        data_heading_inv = ObsBuilderBase._calc_heading_quat(smpl_root).inv()
        return init_heading * data_heading_inv

    @staticmethod
    def _anchor_6d(
        smpl: np.ndarray,
        base_quat_wxyz: np.ndarray,
        apply_delta_heading: sRot,
    ) -> np.ndarray:
        """
        Single-frame anchor orientation: 6D(inv(base) * apply_delta_heading * smpl_root).
        Matches C++ GatherMotionAnchorOrientationMutiFrame (num_frames=1).
        """
        smpl_root = ObsBuilderBase._smpl_root_to_robot(smpl)
        aligned = apply_delta_heading * smpl_root
        base = sRot.from_quat(base_quat_wxyz, scalar_first=True)
        return ObsBuilderBase._quat_to_6d(base.inv() * aligned)

    # ------------------------------------------------------------------
    # Rotation → 6D
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_6d(rot: sRot) -> np.ndarray:
        """First 2 columns of rotation matrix, row-wise → (6,) float32."""
        m = rot.as_matrix()
        return np.array([m[0,0], m[0,1], m[1,0], m[1,1], m[2,0], m[2,1]], dtype=np.float32)

    # ------------------------------------------------------------------
    # History sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_history(buf, n_frames: int, step: int) -> np.ndarray:
        """
        Sample n_frames at given step from newest-first deque.
        Output order: [oldest, ..., newest] (newest_first=false).
        Pads with oldest available frame when buffer is short.
        """
        buf_len = len(buf)
        frames = []
        for i in range(n_frames - 1, -1, -1):
            idx = i * step
            frames.append(buf[min(idx, buf_len - 1)])
        return np.concatenate(frames).astype(np.float32)
