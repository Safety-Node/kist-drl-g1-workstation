"""
VRCoordProvider

PicoReaderProvider 의 헤드셋/컨트롤러 포즈를 Unity frame → Robot frame 으로 변환하고,
teleop encoder 에 넣을 vr_3point_local_target (9,) + vr_3point_local_orn_target (12,) 를
생성한다.

좌표 변환 (gear_sonic/scripts/pico_manager_thread_server.py 동일):
    Q = [[-1,0,0],[0,0,1],[0,1,0]]  (Unity → Robot frame)
    Unity X-right  → Robot -X
    Unity Y-up     → Robot  Z
    Unity Z-forward → Robot  Y

3-point 정의 (L-Wrist, R-Wrist, Neck 순서):
    L-Wrist = left_controller_pose
    R-Wrist = right_controller_pose
    Neck    = headset_pose

기준점(anchor) = 인체 골반(pelvis):
    pelvis_unity = headset_pos_unity - [0, HEADSET_TO_PELVIS_Y, 0]
    * HEADSET_TO_PELVIS_Y 는 Unity Y (up) 방향 오프셋 (기본 0.70 m)
    * XRT SDK 원점이 헤드셋 초기 위치이므로, 초기화 시 캘리브레이션 권장

회전 오프셋 (원본 GearSonic SMPL joint frame 정렬):
    Anchor/Root : yaw -90°  (SMPL 골반 frame → robot anchor frame)
    L-Wrist     : roll +90°
    R-Wrist     : roll -90°, yaw +180°
    Neck        : yaw -90°

출력:
    vr_3point_local_target    ndarray (9,)   = [lw_pos, rw_pos, neck_pos] in anchor local
    vr_3point_local_orn_target ndarray (12,)  = [lw_q, rw_q, neck_q] scalar-first, anchor local
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from .pico_reader_provider import PicoReaderProvider, VRPoseData
from .singleton import singleton

logger = logging.getLogger(__name__)

# Unity → Robot 좌표 변환 행렬
_Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)

# Unity Y축 방향 헤드셋→골반 거리 (미터)
# Unity Y-up 기준이면 양수 (머리가 골반보다 Y+ 에 있음)
# XRT SDK Y-down 기준이면 음수로 변경 필요
_HEADSET_TO_PELVIS_Y_DEFAULT = 0.70

# 원본 GearSonic SMPL joint frame 정렬용 회전 오프셋 (post-multiply = 내재 회전)
# gear_sonic/scripts/pico_manager_thread_server.py 의 OFFSETS 와 동일
_OFFSET_ANCHOR = sRot.from_euler("xyz", [0,   0, -90], degrees=True)   # Root/Pelvis
_OFFSET_LW     = sRot.from_euler("xyz", [90,  0,   0], degrees=True)   # L-Wrist
_OFFSET_RW     = sRot.from_euler("xyz", [-90, 0, 180], degrees=True)   # R-Wrist
_OFFSET_NECK   = sRot.from_euler("xyz", [0,   0, -90], degrees=True)   # Neck


@dataclass
class VRCoordData:
    """좌표 변환 완료된 VR 3-point 스냅샷."""

    # Robot frame 절대 위치/방향 (골반 기준화 전)
    headset_pos_robot: np.ndarray       # (3,) Robot frame
    headset_quat_robot: np.ndarray      # (4,) scalar-first [w,x,y,z]
    lw_pos_robot: np.ndarray            # (3,)
    lw_quat_robot: np.ndarray           # (4,)
    rw_pos_robot: np.ndarray            # (3,)
    rw_quat_robot: np.ndarray           # (4,)
    pelvis_pos_robot: np.ndarray        # (3,) 추정 골반 위치

    # encoder 입력 (골반 local frame 기준화 완료)
    vr_3point_local_target: np.ndarray       # (9,)  [lw,rw,neck] positions
    vr_3point_local_orn_target: np.ndarray   # (12,) [lw,rw,neck] quats scalar-first

    timestamp_ns: int


def _yaw_only(q_wxyz: np.ndarray) -> np.ndarray:
    """쿼터니언에서 yaw(Z축 회전)만 추출. 헤드셋 pitch/roll 무시."""
    yaw = sRot.from_quat(q_wxyz, scalar_first=True).as_euler('xyz')[2]
    return sRot.from_euler('z', yaw).as_quat(scalar_first=True)


def _smpl_to_3pt_local(body_poses_unity: np.ndarray) -> tuple:
    """
    SMPL 24관절(Unity frame, scalar-last) → encoder 입력 (target 9, orn 12).

    GearSonic pico_manager_thread_server._process_3pt_pose() 와 동일한 파이프라인.

    1. 전체 관절을 Robot frame 으로 변환
    2. joint 0(Root), 22(L-Wrist), 23(R-Wrist), 12(Neck) 추출 + OFFSET 적용
    3. Root 기준으로 상대 위치/방향 계산
    4. [L-Wrist, R-Wrist, Neck] 순서로 flatten

    Args:
        body_poses_unity: (24, 7) float, [x,y,z,qx,qy,qz,qw] scalar-last, Unity frame
    Returns:
        (target (9,), orn_target (12,)) float32, scalar-first, Root local frame
    """
    body_poses_unity = body_poses_unity.copy()

    # 1. Unity → Robot frame
    robot_poses = np.zeros((body_poses_unity.shape[0], 7), dtype=np.float64)
    for i in range(body_poses_unity.shape[0]):
        p = body_poses_unity[i]
        pos_r = _Q @ p[:3]
        rot_r = sRot.from_matrix(_Q @ sRot.from_quat(p[3:], scalar_first=False).as_matrix() @ _Q.T)
        robot_poses[i, :3] = pos_r
        robot_poses[i, 3:] = rot_r.as_quat(scalar_first=True)  # scalar-first

    # 2. 4 keypoints 추출 + OFFSET 적용 (scalar-last 로 임시 저장)
    JOINT_MAP = {0: 0, 22: 1, 23: 2, 12: 3}
    kp = np.zeros((4, 7), dtype=np.float64)
    for smpl_idx, kp_idx in JOINT_MAP.items():
        q_sf = robot_poses[smpl_idx, 3:]                             # scalar-first
        q_sl = (sRot.from_quat(q_sf, scalar_first=True) * _OFFSETS[kp_idx]).as_quat(scalar_first=False)
        kp[kp_idx, :3] = robot_poses[smpl_idx, :3]
        kp[kp_idx, 3:] = q_sl                                        # scalar-last

    # 3. Root-local 기준화 (scalar-last 기준 inv)
    root_pos  = kp[0, :3].copy()
    root_rot_inv = sRot.from_quat(kp[0, 3:]).inv()                  # scalar-last default
    for i in range(1, 4):
        kp[i, :3] = root_rot_inv.apply(kp[i, :3] - root_pos)
        kp[i, 3:] = (root_rot_inv * sRot.from_quat(kp[i, 3:])).as_quat(scalar_first=True)
        # kp[i, 3:] is now scalar-first

    # 4. [L-Wrist(1), R-Wrist(2), Neck(3)] flatten
    target = np.concatenate([kp[1, :3], kp[2, :3], kp[3, :3]]).astype(np.float32)
    orn    = np.concatenate([kp[1, 3:], kp[2, 3:], kp[3, 3:]]).astype(np.float32)
    return target, orn


# SMPL keypoint 별 rotation offset (GearSonic OFFSETS 와 동일)
_OFFSETS = [
    sRot.from_euler("xyz", [0,   0, -90], degrees=True),   # 0: Root
    sRot.from_euler("xyz", [90,  0,   0], degrees=True),   # 1: L-Wrist
    sRot.from_euler("xyz", [-90, 0, 180], degrees=True),   # 2: R-Wrist
    sRot.from_euler("xyz", [0,   0, -90], degrees=True),   # 3: Neck
]


def _unity_to_robot(pose_unity: np.ndarray) -> tuple:
    """
    Unity frame 7-벡터 → Robot frame (pos, quat scalar-first).

    pose_unity: (7,) [x,y,z, qx,qy,qz,qw] scalar-last
    returns: (pos (3,), quat_scalar_first (4,))
    """
    pos_r = _Q @ pose_unity[:3]
    rot_u = sRot.from_quat(pose_unity[3:], scalar_first=False)
    rot_r = sRot.from_matrix(_Q @ rot_u.as_matrix() @ _Q.T)
    return pos_r, rot_r.as_quat(scalar_first=True)


def _make_3point_local(
    neck_pos_r: np.ndarray, neck_q_r: np.ndarray,
    lw_pos_r: np.ndarray, lw_q_r: np.ndarray,
    rw_pos_r: np.ndarray, rw_q_r: np.ndarray,
    anchor_pos_r: np.ndarray, anchor_q_r: np.ndarray,
) -> tuple:
    """
    Robot frame 절대 포즈 → anchor local frame 기준화.

    anchor = 골반 위치·방향 (robot frame).
    반환: (target (9,), orn_target (12,)) — 모두 scalar-first 쿼터니언.
    """
    anchor_rot_inv = sRot.from_quat(anchor_q_r, scalar_first=True).inv()

    def to_local_pos(pos):
        return anchor_rot_inv.apply(pos - anchor_pos_r)

    def to_local_quat(q):
        return (anchor_rot_inv * sRot.from_quat(q, scalar_first=True)).as_quat(scalar_first=True)

    # 순서: L-Wrist, R-Wrist, Neck
    target = np.concatenate([
        to_local_pos(lw_pos_r),
        to_local_pos(rw_pos_r),
        to_local_pos(neck_pos_r),
    ]).astype(np.float32)

    orn = np.concatenate([
        to_local_quat(lw_q_r),
        to_local_quat(rw_q_r),
        to_local_quat(neck_q_r),
    ]).astype(np.float32)

    return target, orn


@singleton
class VRCoordProvider:
    """
    PicoReaderProvider 위에서 동작하는 좌표 변환 레이어.

    PicoReaderProvider 가 start() 된 뒤 이 Provider 를 사용한다.

    사용법:
        pico = PicoReaderProvider(); pico.start()
        vr = VRCoordProvider()

        data: VRCoordData | None = vr.get_latest()

    headset_to_pelvis_y: Unity Y 방향 헤드셋→골반 오프셋 (m).
        양수 = Unity Y-up 기준.
        값이 맞지 않으면 vr_3point_local_target 이 이상해짐 →
        캘리브레이션 후 set_headset_to_pelvis_y() 로 수정.
    """

    def __init__(self, headset_to_pelvis_y: float = _HEADSET_TO_PELVIS_Y_DEFAULT):
        self._headset_to_pelvis_y = headset_to_pelvis_y
        self._pico = PicoReaderProvider()

    def set_headset_to_pelvis_y(self, value: float) -> None:
        """캘리브레이션 후 헤드셋→골반 Unity-Y 오프셋을 업데이트한다."""
        self._headset_to_pelvis_y = value
        logger.info("VRCoordProvider: headset_to_pelvis_y = %.3f m", value)

    def get_latest(self) -> Optional["VRCoordData"]:
        """최신 VRCoordData 를 반환한다. PicoReaderProvider 미연결 시 None."""
        raw: Optional[VRPoseData] = self._pico.get_latest()
        if raw is None:
            return None

        try:
            # body_poses 가 있으면 GearSonic 원본 파이프라인 사용
            # (SMPL root 를 anchor 로 사용 — 컨트롤러 기반 pelvis 추정보다 정확)
            if raw.body_poses is not None:
                return self._from_body_poses(raw)
            else:
                return self._from_controllers(raw)
        except Exception:
            logger.exception("VRCoordProvider: 변환 오류")
            return None

    def _from_body_poses(self, raw: VRPoseData) -> Optional["VRCoordData"]:
        """SMPL body tracking (24관절) 기반 — GearSonic _process_3pt_pose 동일."""
        target, orn = _smpl_to_3pt_local(raw.body_poses)

        # 디버깅/로깅용 필드는 controller 로 채움
        h_pos_r, h_q_r = _unity_to_robot(raw.headset_pose)
        lw_pos_r, lw_q_r = _unity_to_robot(raw.left_controller_pose)
        rw_pos_r, rw_q_r = _unity_to_robot(raw.right_controller_pose)

        # SMPL root → robot frame 위치를 pelvis 로 사용
        pelvis_pos_r = _Q @ raw.body_poses[0, :3]

        return VRCoordData(
            headset_pos_robot=h_pos_r,
            headset_quat_robot=h_q_r,
            lw_pos_robot=lw_pos_r,
            lw_quat_robot=lw_q_r,
            rw_pos_robot=rw_pos_r,
            rw_quat_robot=rw_q_r,
            pelvis_pos_robot=pelvis_pos_r,
            vr_3point_local_target=target,
            vr_3point_local_orn_target=orn,
            timestamp_ns=raw.timestamp_ns,
        )

    def _from_controllers(self, raw: VRPoseData) -> Optional["VRCoordData"]:
        """컨트롤러 포즈 기반 fallback (body tracking 없을 때)."""
        # 1. Unity → Robot frame
        h_pos_r, h_q_r = _unity_to_robot(raw.headset_pose)
        lw_pos_r, lw_q_r = _unity_to_robot(raw.left_controller_pose)
        rw_pos_r, rw_q_r = _unity_to_robot(raw.right_controller_pose)

        # 2. 골반 위치 추정 (Unity Y 방향으로 offset 적용 후 변환)
        pelvis_unity = raw.headset_pose[:3].copy()
        pelvis_unity[1] -= self._headset_to_pelvis_y
        pelvis_pos_r = _Q @ pelvis_unity

        # 3. SMPL joint frame 정렬 오프셋 적용 (post-multiply = 내재 회전)
        def _apply(q_wxyz, offset_rot):
            return (sRot.from_quat(q_wxyz, scalar_first=True) * offset_rot).as_quat(scalar_first=True)

        # anchor: yaw만 사용 — 헤드셋 pitch/roll이 섞이면 목/손목 local 위치가 틀어짐
        pelvis_q_r  = _apply(_yaw_only(h_q_r), _OFFSET_ANCHOR)
        lw_q_r_c    = _apply(lw_q_r, _OFFSET_LW)
        rw_q_r_c    = _apply(rw_q_r, _OFFSET_RW)
        neck_q_r_c  = _apply(h_q_r,  _OFFSET_NECK)

        # 4. anchor-local 기준화
        target, orn = _make_3point_local(
            h_pos_r, neck_q_r_c,
            lw_pos_r, lw_q_r_c,
            rw_pos_r, rw_q_r_c,
            pelvis_pos_r, pelvis_q_r,
        )

        return VRCoordData(
            headset_pos_robot=h_pos_r,
            headset_quat_robot=h_q_r,
            lw_pos_robot=lw_pos_r,
            lw_quat_robot=lw_q_r_c,
            rw_pos_robot=rw_pos_r,
            rw_quat_robot=rw_q_r_c,
            pelvis_pos_robot=pelvis_pos_r,
            vr_3point_local_target=target,
            vr_3point_local_orn_target=orn,
            timestamp_ns=raw.timestamp_ns,
        )

    @property
    def connected(self) -> bool:
        return self._pico.connected
