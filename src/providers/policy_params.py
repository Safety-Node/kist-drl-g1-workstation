"""
G1 29-DOF policy constants (MuJoCo 순서 0-28).

출처: gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp

최종 관절 목표각:
    q_target[i] = default_angle[i] + action[i] * action_scale[i]

모터 PD 제어:
    τ = kp[i] * (q_target[i] - q[i]) + kd[i] * (0 - dq[i])
"""

import math
import numpy as np

# ── 모터 아마추어 상수 ──────────────────────────────────────────────────────
_A5020   = 0.003609725
_A7520_14 = 0.010177520
_A7520_22 = 0.025101925
_A4010   = 0.00425

# ── PID 계산 상수 ──────────────────────────────────────────────────────────
_W  = 10.0 * 2.0 * math.pi   # natural frequency (rad/s)
_Z  = 2.0                    # damping ratio

_K5020    = _A5020   * _W ** 2
_K7520_14 = _A7520_14 * _W ** 2
_K7520_22 = _A7520_22 * _W ** 2
_K4010    = _A4010   * _W ** 2

_D5020    = 2.0 * _Z * _A5020   * _W
_D7520_14 = 2.0 * _Z * _A7520_14 * _W
_D7520_22 = 2.0 * _Z * _A7520_22 * _W
_D4010    = 2.0 * _Z * _A4010   * _W

# ── effort limits ──────────────────────────────────────────────────────────
_E5020    = 25.0
_E7520_14 = 88.0
_E7520_22 = 139.0
_E4010    = 5.0


def _s(effort, stiff):
    return 0.25 * effort / stiff


# ── action_scale (MuJoCo 순서 0-28) ────────────────────────────────────────
ACTION_SCALE = np.array([
    _s(_E7520_22, _K7520_22),  # 0  left_hip_pitch_joint
    _s(_E7520_22, _K7520_22),  # 1  left_hip_roll_joint
    _s(_E7520_14, _K7520_14),  # 2  left_hip_yaw_joint
    _s(_E7520_22, _K7520_22),  # 3  left_knee_joint
    _s(_E5020,    _K5020),     # 4  left_ankle_pitch_joint
    _s(_E5020,    _K5020),     # 5  left_ankle_roll_joint
    _s(_E7520_22, _K7520_22),  # 6  right_hip_pitch_joint
    _s(_E7520_22, _K7520_22),  # 7  right_hip_roll_joint
    _s(_E7520_14, _K7520_14),  # 8  right_hip_yaw_joint
    _s(_E7520_22, _K7520_22),  # 9  right_knee_joint
    _s(_E5020,    _K5020),     # 10 right_ankle_pitch_joint
    _s(_E5020,    _K5020),     # 11 right_ankle_roll_joint
    _s(_E7520_14, _K7520_14),  # 12 waist_yaw_joint
    _s(_E5020,    _K5020),     # 13 waist_roll_joint
    _s(_E5020,    _K5020),     # 14 waist_pitch_joint
    _s(_E5020,    _K5020),     # 15 left_shoulder_pitch_joint
    _s(_E5020,    _K5020),     # 16 left_shoulder_roll_joint
    _s(_E5020,    _K5020),     # 17 left_shoulder_yaw_joint
    _s(_E5020,    _K5020),     # 18 left_elbow_joint
    _s(_E5020,    _K5020),     # 19 left_wrist_roll_joint
    _s(_E4010,    _K4010),     # 20 left_wrist_pitch_joint
    _s(_E4010,    _K4010),     # 21 left_wrist_yaw_joint
    _s(_E5020,    _K5020),     # 22 right_shoulder_pitch_joint
    _s(_E5020,    _K5020),     # 23 right_shoulder_roll_joint
    _s(_E5020,    _K5020),     # 24 right_shoulder_yaw_joint
    _s(_E5020,    _K5020),     # 25 right_elbow_joint
    _s(_E5020,    _K5020),     # 26 right_wrist_roll_joint
    _s(_E4010,    _K4010),     # 27 right_wrist_pitch_joint
    _s(_E4010,    _K4010),     # 28 right_wrist_yaw_joint
], dtype=np.float32)

# ── default standing angles (rad, MuJoCo 순서 0-28) ────────────────────────
DEFAULT_ANGLES = np.array([
    -0.312,  # 0  left_hip_pitch_joint
     0.000,  # 1  left_hip_roll_joint
     0.000,  # 2  left_hip_yaw_joint
     0.669,  # 3  left_knee_joint
    -0.363,  # 4  left_ankle_pitch_joint
     0.000,  # 5  left_ankle_roll_joint
    -0.312,  # 6  right_hip_pitch_joint
     0.000,  # 7  right_hip_roll_joint
     0.000,  # 8  right_hip_yaw_joint
     0.669,  # 9  right_knee_joint
    -0.363,  # 10 right_ankle_pitch_joint
     0.000,  # 11 right_ankle_roll_joint
     0.000,  # 12 waist_yaw_joint
     0.000,  # 13 waist_roll_joint
     0.000,  # 14 waist_pitch_joint
     0.200,  # 15 left_shoulder_pitch_joint
     0.200,  # 16 left_shoulder_roll_joint
     0.000,  # 17 left_shoulder_yaw_joint
     0.600,  # 18 left_elbow_joint
     0.000,  # 19 left_wrist_roll_joint
     0.000,  # 20 left_wrist_pitch_joint
     0.000,  # 21 left_wrist_yaw_joint
     0.200,  # 22 right_shoulder_pitch_joint
    -0.200,  # 23 right_shoulder_roll_joint
     0.000,  # 24 right_shoulder_yaw_joint
     0.600,  # 25 right_elbow_joint
     0.000,  # 26 right_wrist_roll_joint
     0.000,  # 27 right_wrist_pitch_joint
     0.000,  # 28 right_wrist_yaw_joint
], dtype=np.float32)

# ── Kp (MuJoCo 순서) ──────────────────────────────────────────────────────
KPS = np.array([
    _K7520_22,        # 0  left_hip_pitch_joint
    _K7520_22,        # 1  left_hip_roll_joint
    _K7520_14,        # 2  left_hip_yaw_joint
    _K7520_22,        # 3  left_knee_joint
    2.0 * _K5020,     # 4  left_ankle_pitch_joint
    2.0 * _K5020,     # 5  left_ankle_roll_joint
    _K7520_22,        # 6  right_hip_pitch_joint
    _K7520_22,        # 7  right_hip_roll_joint
    _K7520_14,        # 8  right_hip_yaw_joint
    _K7520_22,        # 9  right_knee_joint
    2.0 * _K5020,     # 10 right_ankle_pitch_joint
    2.0 * _K5020,     # 11 right_ankle_roll_joint
    _K7520_14,        # 12 waist_yaw_joint
    2.0 * _K5020,     # 13 waist_roll_joint
    2.0 * _K5020,     # 14 waist_pitch_joint
    _K5020,           # 15 left_shoulder_pitch_joint
    _K5020,           # 16 left_shoulder_roll_joint
    _K5020,           # 17 left_shoulder_yaw_joint
    _K5020,           # 18 left_elbow_joint
    _K5020,           # 19 left_wrist_roll_joint
    _K4010,           # 20 left_wrist_pitch_joint
    _K4010,           # 21 left_wrist_yaw_joint
    _K5020,           # 22 right_shoulder_pitch_joint
    _K5020,           # 23 right_shoulder_roll_joint
    _K5020,           # 24 right_shoulder_yaw_joint
    _K5020,           # 25 right_elbow_joint
    _K5020,           # 26 right_wrist_roll_joint
    _K4010,           # 27 right_wrist_pitch_joint
    _K4010,           # 28 right_wrist_yaw_joint
], dtype=np.float32)

# ── Kd (MuJoCo 순서) ──────────────────────────────────────────────────────
KDS = np.array([
    _D7520_22,        # 0  left_hip_pitch_joint
    _D7520_22,        # 1  left_hip_roll_joint
    _D7520_14,        # 2  left_hip_yaw_joint
    _D7520_22,        # 3  left_knee_joint
    2.0 * _D5020,     # 4  left_ankle_pitch_joint
    2.0 * _D5020,     # 5  left_ankle_roll_joint
    _D7520_22,        # 6  right_hip_pitch_joint
    _D7520_22,        # 7  right_hip_roll_joint
    _D7520_14,        # 8  right_hip_yaw_joint
    _D7520_22,        # 9  right_knee_joint
    2.0 * _D5020,     # 10 right_ankle_pitch_joint
    2.0 * _D5020,     # 11 right_ankle_roll_joint
    _D7520_14,        # 12 waist_yaw_joint
    2.0 * _D5020,     # 13 waist_roll_joint
    2.0 * _D5020,     # 14 waist_pitch_joint
    _D5020,           # 15 left_shoulder_pitch_joint
    _D5020,           # 16 left_shoulder_roll_joint
    _D5020,           # 17 left_shoulder_yaw_joint
    _D5020,           # 18 left_elbow_joint
    _D5020,           # 19 left_wrist_roll_joint
    _D4010,           # 20 left_wrist_pitch_joint
    _D4010,           # 21 left_wrist_yaw_joint
    _D5020,           # 22 right_shoulder_pitch_joint
    _D5020,           # 23 right_shoulder_roll_joint
    _D5020,           # 24 right_shoulder_yaw_joint
    _D5020,           # 25 right_elbow_joint
    _D5020,           # 26 right_wrist_roll_joint
    _D4010,           # 27 right_wrist_pitch_joint
    _D4010,           # 28 right_wrist_yaw_joint
], dtype=np.float32)
