"""
G1ObsProvider

UnitreeG1Provider 에서 관절 상태와 IMU 를 읽어
teleop encoder/decoder 입력에 필요한 관측값을 생성한다.

출력:
    [encoder 입력용]
    joint_pos_lowerbody (12,)              현재 하체 관절 위치 (MuJoCo 순서 0-11)
    joint_vel_lowerbody (12,)
    joint_pos_lowerbody_10f_s5 (120,)     10×12 히스토리 (step=5)
    joint_vel_lowerbody_10f_s5 (120,)
    anchor_orientation_6d (6,)            robot base 방향 6D rotation

    [decoder 입력용]
    joint_pos_all (29,)                   전체 관절 위치 (MuJoCo 순서 0-28)
    joint_vel_all (29,)
    joint_pos_all_10f_s1 (290,)           10×29 히스토리 (step=1)
    joint_vel_all_10f_s1 (290,)
    base_ang_vel (3,)                     현재 기저 각속도 (robot frame)
    base_ang_vel_10f_s1 (30,)            10×3 히스토리 (step=1)
    gravity_dir (3,)                      현재 중력 방향 (robot frame)
    gravity_dir_10f_s1 (30,)             10×3 히스토리 (step=1)

하체 관절 순서 (MuJoCo 0-11):
    0  left_hip_pitch_joint  ...  5  left_ankle_roll_joint
    6  right_hip_pitch_joint ... 11  right_ankle_roll_joint

전체 29 관절 순서 (MuJoCo 0-28):
    0-11  하체 (위와 동일)
    12  waist_yaw_joint
    13  waist_roll_joint
    14  waist_pitch_joint
    15-21 left arm (shoulder_pitch, roll, yaw, elbow, wrist_roll, pitch, yaw)
    22-28 right arm (shoulder_pitch, roll, yaw, elbow, wrist_roll, pitch, yaw)

히스토리 버퍼:
    LOWER BODY: deque(maxlen=50), step5 → indices [0,5,...,45]
    ALL JOINTS / ANG_VEL / GRAVITY: deque(maxlen=10), step1 → indices [0,1,...,9]
    50프레임 미만 시 가장 오래된 프레임으로 padding
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as sRot

logger = logging.getLogger(__name__)

# 하체 관절 이름 (MuJoCo 순서 0-11)
LOWER_BODY_JOINT_NAMES = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
]

# 전체 29 관절 이름 (MuJoCo 순서 0-28)
ALL_JOINT_NAMES = [
    # 하체 (0-11)
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    # 허리 (12-14)
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    # 왼팔 (15-21)
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    # 오른팔 (22-28)
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

N_LOWER = len(LOWER_BODY_JOINT_NAMES)   # 12
N_ALL = len(ALL_JOINT_NAMES)            # 29
N_HISTORY_LOWER = 50                    # step5 히스토리 버퍼
N_HISTORY_ALL = 10                      # step1 히스토리 버퍼
N_FRAMES_LOWER = 10
N_FRAMES_ALL = 10
STEP_LOWER = 5
STEP_ALL = 1

# encoder 입력 차원 참조용
ENCODER_INPUT_DIM = 1762


@dataclass
class G1ObsData:
    """G1 로봇 관측 스냅샷."""

    # encoder 입력용
    joint_pos_lowerbody: np.ndarray             # (12,) MuJoCo 순서
    joint_vel_lowerbody: np.ndarray             # (12,)
    joint_pos_lowerbody_10f_s5: np.ndarray      # (120,) = 10×12, step5
    joint_vel_lowerbody_10f_s5: np.ndarray      # (120,)
    anchor_orientation_6d: np.ndarray           # (6,) 6D rotation

    # decoder 입력용
    joint_pos_all: np.ndarray                   # (29,) MuJoCo 순서
    joint_vel_all: np.ndarray                   # (29,)
    joint_pos_all_10f_s1: np.ndarray            # (290,) = 10×29, step1
    joint_vel_all_10f_s1: np.ndarray            # (290,)
    base_ang_vel: np.ndarray                    # (3,) robot frame
    base_ang_vel_10f_s1: np.ndarray             # (30,) = 10×3
    gravity_dir: np.ndarray                     # (3,) robot frame
    gravity_dir_10f_s1: np.ndarray              # (30,) = 10×3

    timestamp_monotonic: float


def _quat_to_6d_rowwise(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    스칼라-선행 쿼터니언 → 6D rotation (회전 행렬 1-2열, row-wise).

    C++ 코드와 동일한 레이아웃:
        [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]
    """
    rot = sRot.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    return np.array([
        rot[0, 0], rot[0, 1],
        rot[1, 0], rot[1, 1],
        rot[2, 0], rot[2, 1],
    ], dtype=np.float32)


def _gravity_dir_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    스칼라-선행 쿼터니언(world→body) → 로봇 frame 의 중력 방향 (단위벡터).

    geometry_msgs/Imu.orientation 은 body→world 변환이므로,
    world 기준 중력 벡터 [0, 0, -1] 을 body frame 으로 변환한다.
    """
    rot = sRot.from_quat(quat_wxyz, scalar_first=True)
    g_world = np.array([0.0, 0.0, -1.0])
    return rot.inv().apply(g_world).astype(np.float32)


class G1ObsProvider:
    """
    UnitreeG1Provider 로부터 encoder/decoder 입력용 관측값을 빌드하는 Provider.

    주기적으로 update() 를 호출해 히스토리 버퍼를 갱신해야 한다.

    사용법:
        g1 = UnitreeG1Provider(); g1.start()
        obs = G1ObsProvider(g1)
        obs.update()      # 제어 루프 매 스텝 호출
        data = obs.get_latest()
    """

    def __init__(self, g1_provider):
        self._g1 = g1_provider

        # 하체 step5 히스토리 (maxlen=50)
        self._pos_lower_history: deque = deque(maxlen=N_HISTORY_LOWER)
        self._vel_lower_history: deque = deque(maxlen=N_HISTORY_LOWER)

        # 전체 관절 step1 히스토리 (maxlen=10)
        self._pos_all_history: deque = deque(maxlen=N_HISTORY_ALL)
        self._vel_all_history: deque = deque(maxlen=N_HISTORY_ALL)

        # IMU step1 히스토리 (maxlen=10)
        self._ang_vel_history: deque = deque(maxlen=N_HISTORY_ALL)
        self._gravity_history: deque = deque(maxlen=N_HISTORY_ALL)

        self._joint_name_to_idx: Optional[dict] = None
        self._latest: Optional[G1ObsData] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self) -> bool:
        """
        UnitreeG1Provider 에서 최신 관절 상태를 읽어 히스토리를 갱신한다.

        반환: True = 새 데이터 추가됨, False = 데이터 없음
        """
        js_cache = self._g1.joint_state
        imu_cache = self._g1.imu_base

        if js_cache.last_seen_ts == 0.0 or js_cache.value is None:
            return False

        js = js_cache.value

        # JointState 이름→인덱스 매핑 캐시 (첫 호출 시 구축)
        if self._joint_name_to_idx is None:
            self._joint_name_to_idx = {name: i for i, name in enumerate(js.name)}
            missing = [n for n in ALL_JOINT_NAMES if n not in self._joint_name_to_idx]
            if missing:
                logger.warning("G1ObsProvider: JointState에 없는 관절: %s", missing)

        # 하체 12 관절 추출 (MuJoCo 순서)
        try:
            pos_lower = np.array([
                js.position[self._joint_name_to_idx[n]]
                for n in LOWER_BODY_JOINT_NAMES
            ], dtype=np.float32)
            vel_lower = np.array([
                js.velocity[self._joint_name_to_idx[n]]
                for n in LOWER_BODY_JOINT_NAMES
            ], dtype=np.float32)
        except (KeyError, IndexError):
            logger.debug("G1ObsProvider: 하체 관절 인덱스 오류 — JointState 재파싱")
            self._joint_name_to_idx = None
            return False

        # 전체 29 관절 추출 (MuJoCo 순서)
        try:
            pos_all = np.array([
                js.position[self._joint_name_to_idx[n]]
                if n in self._joint_name_to_idx else 0.0
                for n in ALL_JOINT_NAMES
            ], dtype=np.float32)
            vel_all = np.array([
                js.velocity[self._joint_name_to_idx[n]]
                if n in self._joint_name_to_idx else 0.0
                for n in ALL_JOINT_NAMES
            ], dtype=np.float32)
        except (KeyError, IndexError):
            pos_all = np.zeros(N_ALL, dtype=np.float32)
            vel_all = np.zeros(N_ALL, dtype=np.float32)

        # IMU 처리
        anchor_6d = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)
        gravity_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        if imu_cache.last_seen_ts != 0.0 and imu_cache.value is not None:
            try:
                imu = imu_cache.value
                q = imu.orientation
                quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)

                anchor_6d = _quat_to_6d_rowwise(quat_wxyz)
                gravity_dir = _gravity_dir_from_quat(quat_wxyz)

                av = imu.angular_velocity
                ang_vel = np.array([av.x, av.y, av.z], dtype=np.float32)
            except Exception:
                logger.debug("G1ObsProvider: IMU 파싱 오류, 기본값 사용")

        # 히스토리 갱신 (appendleft → 인덱스 0 이 최신)
        self._pos_lower_history.appendleft(pos_lower)
        self._vel_lower_history.appendleft(vel_lower)
        self._pos_all_history.appendleft(pos_all)
        self._vel_all_history.appendleft(vel_all)
        self._ang_vel_history.appendleft(ang_vel)
        self._gravity_history.appendleft(gravity_dir)

        # 10frame 히스토리 샘플링
        pos_10f_lower = self._sample_history(self._pos_lower_history, STEP_LOWER)
        vel_10f_lower = self._sample_history(self._vel_lower_history, STEP_LOWER)
        pos_10f_all   = self._sample_history(self._pos_all_history,   STEP_ALL)
        vel_10f_all   = self._sample_history(self._vel_all_history,   STEP_ALL)
        ang_vel_10f   = self._sample_history(self._ang_vel_history,   STEP_ALL)
        gravity_10f   = self._sample_history(self._gravity_history,   STEP_ALL)

        self._latest = G1ObsData(
            joint_pos_lowerbody=pos_lower,
            joint_vel_lowerbody=vel_lower,
            joint_pos_lowerbody_10f_s5=pos_10f_lower,
            joint_vel_lowerbody_10f_s5=vel_10f_lower,
            anchor_orientation_6d=anchor_6d,
            joint_pos_all=pos_all,
            joint_vel_all=vel_all,
            joint_pos_all_10f_s1=pos_10f_all,
            joint_vel_all_10f_s1=vel_10f_all,
            base_ang_vel=ang_vel,
            base_ang_vel_10f_s1=ang_vel_10f,
            gravity_dir=gravity_dir,
            gravity_dir_10f_s1=gravity_10f,
            timestamp_monotonic=time.monotonic(),
        )
        return True

    def get_latest(self) -> Optional[G1ObsData]:
        """가장 최근 update() 결과를 반환한다."""
        return self._latest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_history(self, buf: deque, step: int) -> np.ndarray:
        """
        deque[0]=최신 에서 step 간격으로 N_FRAMES_ALL(=10) 개를 샘플링해 flatten.

        버퍼가 충분하지 않으면 가장 오래된 프레임으로 padding.
        """
        buf_len = len(buf)
        frames = []
        for i in range(N_FRAMES_ALL):
            idx = i * step
            if idx < buf_len:
                frames.append(buf[idx])
            else:
                frames.append(buf[buf_len - 1])
        return np.concatenate(frames).astype(np.float32)
