"""
TeleopEncoderInputProvider

VRCoordProvider + G1ObsProvider 의 출력을 조합해
teleop encoder (ONNX) 에 넣을 입력 벡터(1762 dims)를 조립한다.

encoder input 전체 레이아웃 (총 1762 dims):
    [  0]  encoder_mode_4                               [4]  ← ACTIVE: [1,0,0,0] (mode_id=1 at pos 0)
    [  4]  motion_joint_positions_10frame_step5         [290] zeros
    [294]  motion_joint_velocities_10frame_step5        [290] zeros
    [584]  motion_root_z_position_10frame_step5         [10]  zeros
    [594]  motion_root_z_position                       [1]   zeros
    [595]  motion_anchor_orientation                    [6]   ← ACTIVE
    [601]  motion_anchor_orientation_10frame_step5      [60]  zeros
    [661]  motion_joint_positions_lowerbody_10frame_step5 [120] ← ACTIVE
    [781]  motion_joint_velocities_lowerbody_10frame_step5 [120] ← ACTIVE
    [901]  vr_3point_local_target                       [9]   ← ACTIVE
    [910]  vr_3point_local_orn_target                   [12]  ← ACTIVE
    [922]  smpl_joints_10frame_step1                    [720] zeros
    [1642] smpl_anchor_orientation_10frame_step1        [60]  zeros
    [1702] motion_joint_positions_wrists_10frame_step1  [60]  zeros

obs_dict 키는 활성 필드만 저장 (디버깅용). encoder_input 은 항상 (1762,).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .vr_coord_provider import VRCoordProvider, VRCoordData
from .g1_obs_provider import G1ObsProvider, G1ObsData

logger = logging.getLogger(__name__)

_ENCODER_MODE_TELEOP = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

ENCODER_INPUT_DIM = 1762

# 각 필드의 (offset, dim) — teleop 비활성 필드는 zeros
_LAYOUT = [
    ("encoder_mode_4",                                  0,    4),
    ("motion_joint_positions_10frame_step5",            4,    290),
    ("motion_joint_velocities_10frame_step5",           294,  290),
    ("motion_root_z_position_10frame_step5",            584,  10),
    ("motion_root_z_position",                         594,  1),
    ("motion_anchor_orientation",                       595,  6),
    ("motion_anchor_orientation_10frame_step5",        601,  60),
    ("motion_joint_positions_lowerbody_10frame_step5",  661,  120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 781,  120),
    ("vr_3point_local_target",                         901,  9),
    ("vr_3point_local_orn_target",                     910,  12),
    ("smpl_joints_10frame_step1",                      922,  720),
    ("smpl_anchor_orientation_10frame_step1",          1642, 60),
    ("motion_joint_positions_wrists_10frame_step1",    1702, 60),
]
assert sum(d for _, _, d in _LAYOUT) == ENCODER_INPUT_DIM

# teleop mode 에서 실제로 채워지는 필드 이름
_ACTIVE_FIELDS = {
    "encoder_mode_4",
    "motion_anchor_orientation",
    "motion_joint_positions_lowerbody_10frame_step5",
    "motion_joint_velocities_lowerbody_10frame_step5",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
}


@dataclass
class TeleopEncoderInput:
    """조립된 encoder 입력 스냅샷."""

    encoder_input: np.ndarray        # (1762,) float32 — ONNX 에 바로 넣을 수 있는 벡터
    obs_dict: Dict[str, np.ndarray]  # 활성 필드만 (디버깅/검증용)
    vr_ok: bool
    g1_ok: bool


class TeleopEncoderInputProvider:
    """
    VRCoordProvider + G1ObsProvider 를 합쳐 encoder input (1762,) 을 생성한다.

    사용법:
        pico = PicoReaderProvider(); pico.start()
        g1   = UnitreeG1Provider();  g1.start()

        vr_coord = VRCoordProvider()
        g1_obs   = G1ObsProvider(g1)

        enc_prov = TeleopEncoderInputProvider(vr_coord, g1_obs)

        # 제어 루프 매 스텝
        g1_obs.update()
        enc_input: TeleopEncoderInput | None = enc_prov.build()
    """

    def __init__(self, vr_coord: VRCoordProvider, g1_obs: G1ObsProvider):
        self._vr = vr_coord
        self._g1 = g1_obs

    def build(self) -> Optional[TeleopEncoderInput]:
        """
        최신 VR + G1 관측값으로 encoder input 벡터(1762,)를 조립한다.

        반환: TeleopEncoderInput, 또는 VR+G1 모두 없을 때 None.
        """
        vr_data: Optional[VRCoordData] = self._vr.get_latest()
        g1_data: Optional[G1ObsData] = self._g1.get_latest()

        vr_ok = vr_data is not None
        g1_ok = g1_data is not None

        if not vr_ok and not g1_ok:
            return None

        # VR fallback
        if vr_ok:
            vr_3pt_pos = vr_data.vr_3point_local_target       # (9,)
            vr_3pt_orn = vr_data.vr_3point_local_orn_target   # (12,)
        else:
            logger.debug("TeleopEncoderInputProvider: VR 데이터 없음, zeros 사용")
            vr_3pt_pos = np.zeros(9, dtype=np.float32)
            vr_3pt_orn = np.zeros(12, dtype=np.float32)
            vr_3pt_orn[0] = 1.0   # L-Wrist identity
            vr_3pt_orn[4] = 1.0   # R-Wrist identity
            vr_3pt_orn[8] = 1.0   # Neck identity

        # G1 fallback
        if g1_ok:
            joint_pos_10f = g1_data.joint_pos_lowerbody_10f_s5   # (120,)
            joint_vel_10f = g1_data.joint_vel_lowerbody_10f_s5   # (120,)
            anchor_6d     = g1_data.anchor_orientation_6d         # (6,)
        else:
            logger.debug("TeleopEncoderInputProvider: G1 데이터 없음, zeros 사용")
            joint_pos_10f = np.zeros(120, dtype=np.float32)
            joint_vel_10f = np.zeros(120, dtype=np.float32)
            anchor_6d     = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)

        # 1762-dim 벡터 (기본값 zero)
        encoder_input = np.zeros(ENCODER_INPUT_DIM, dtype=np.float32)

        # 활성 필드 채우기
        encoder_input[0:4]       = _ENCODER_MODE_TELEOP
        encoder_input[595:601]   = anchor_6d
        encoder_input[661:781]   = joint_pos_10f
        encoder_input[781:901]   = joint_vel_10f
        encoder_input[901:910]   = vr_3pt_pos
        encoder_input[910:922]   = vr_3pt_orn

        obs_dict = {
            "encoder_mode_4":                               encoder_input[0:4].copy(),
            "motion_anchor_orientation":                    encoder_input[595:601].copy(),
            "motion_joint_positions_lowerbody_10frame_step5":  encoder_input[661:781].copy(),
            "motion_joint_velocities_lowerbody_10frame_step5": encoder_input[781:901].copy(),
            "vr_3point_local_target":                       encoder_input[901:910].copy(),
            "vr_3point_local_orn_target":                   encoder_input[910:922].copy(),
        }

        return TeleopEncoderInput(
            encoder_input=encoder_input,
            obs_dict=obs_dict,
            vr_ok=vr_ok,
            g1_ok=g1_ok,
        )
