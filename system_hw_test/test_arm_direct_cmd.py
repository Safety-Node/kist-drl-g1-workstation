"""
Test 1: 직접 팔 관절 명령 전송 — onboard motor controller 응답 확인

PICO 없이 동작. UnitreeG1Provider 로 /bridge/cmd/low 에 JointCmd 를 직접 전송.
로봇 팔이 DEFAULT_ANGLES → 약간 올린 자세로 천천히 이동하는지 확인.

실행:
    source env.sh && uv run system_hw_test/test_arm_direct_cmd.py
    source env.sh && uv run system_hw_test/test_arm_direct_cmd.py --hold   # 기본자세 유지만
"""

import os, sys, time, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

import numpy as np

# 팔 관절 (indices 12-28 in MuJoCo order)
ARM_NAMES = [
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
N_ARM = len(ARM_NAMES)  # 17

# DEFAULT_ANGLES[12:29]
ARM_DEFAULT = np.array([
    0.000,   # waist_yaw
    0.000,   # waist_roll
    0.000,   # waist_pitch
    0.200,   # left_shoulder_pitch
    0.200,   # left_shoulder_roll
    0.000,   # left_shoulder_yaw
    0.600,   # left_elbow
    0.000,   # left_wrist_roll
    0.000,   # left_wrist_pitch
    0.000,   # left_wrist_yaw
    0.200,   # right_shoulder_pitch
   -0.200,   # right_shoulder_roll
    0.000,   # right_shoulder_yaw
    0.600,   # right_elbow
    0.000,   # right_wrist_roll
    0.000,   # right_wrist_pitch
    0.000,   # right_wrist_yaw
], dtype=np.float32)

# 테스트 목표 자세: 양 어깨를 약간 올림 (shoulder_pitch +0.3)
ARM_TARGET = ARM_DEFAULT.copy()
ARM_TARGET[3]  += 0.3   # left_shoulder_pitch  0.2 → 0.5
ARM_TARGET[10] += 0.3   # right_shoulder_pitch 0.2 → 0.5

# PD 게인 (5020 모터 기준)
import math
_A5020 = 0.003609725; _A4010 = 0.00425
_W = 10.0 * 2.0 * math.pi; _Z = 2.0
_K5020 = _A5020 * _W**2;  _K4010 = _A4010 * _W**2
_D5020 = 2.0*_Z*_A5020*_W; _D4010 = 2.0*_Z*_A4010*_W
import numpy as np
from providers.policy_params import KPS, KDS
ARM_KPS = KPS[12:29]
ARM_KDS = KDS[12:29]


def _send(g1, q, chunk_id):
    cmd = {
        "joint_names": ARM_NAMES,
        "q":      q.tolist(),
        "dq":     [0.0] * N_ARM,
        "kp":     ARM_KPS.tolist(),
        "kd":     ARM_KDS.tolist(),
        "tau_ff": [0.0] * N_ARM,
        "mode":   1,
        "weight": 1.0,
        "chunk_id":   chunk_id,
        "step_index": 0,
    }
    g1.publish_joint_cmd_low(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", action="store_true", help="기본자세만 유지 (목표 자세 이동 없음)")
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    from providers.unitree_g1_provider import UnitreeG1Provider
    UnitreeG1Provider.reset()
    g1 = UnitreeG1Provider()
    g1.start()
    print("[INFO] UnitreeG1Provider 시작, DDS 대기 1s...")
    time.sleep(1.0)

    dt = 1.0 / args.hz
    chunk_id = 1

    # ── Phase 1: DEFAULT 자세 유지 3초 ────────────────────────────────────
    print("[INFO] Phase 1: DEFAULT 팔 자세 전송 (3s) — 로봇이 해당 자세로 이동해야 함")
    print(f"       left_shoulder_pitch={ARM_DEFAULT[3]:.3f}, right_shoulder_pitch={ARM_DEFAULT[10]:.3f}")
    deadline = time.monotonic()
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        deadline += dt
        _send(g1, ARM_DEFAULT, chunk_id)
        chunk_id = (chunk_id % 255) + 1 or 1
        sleep = deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    print("[PASS] Phase 1 완료")

    if args.hold:
        print("[INFO] --hold 모드: 종료")
        g1.stop()
        return

    # ── Phase 2: TARGET 자세로 이동 (5초에 걸쳐 선형 보간) ────────────────
    print(f"\n[INFO] Phase 2: TARGET 팔 자세로 5초 이동")
    print(f"       left_shoulder_pitch={ARM_TARGET[3]:.3f}, right_shoulder_pitch={ARM_TARGET[10]:.3f}")
    deadline = time.monotonic()
    t_start = time.monotonic()
    t_dur = 5.0
    while True:
        elapsed = time.monotonic() - t_start
        alpha = min(elapsed / t_dur, 1.0)
        q_now = ARM_DEFAULT + alpha * (ARM_TARGET - ARM_DEFAULT)

        deadline += dt
        _send(g1, q_now, chunk_id)
        chunk_id = (chunk_id % 255) + 1 or 1

        if alpha >= 1.0:
            break
        sleep = deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    print("[PASS] Phase 2 완료 — 어깨가 올라갔으면 onboard 정상")

    # ── Phase 3: DEFAULT 복귀 3초 ─────────────────────────────────────────
    print("\n[INFO] Phase 3: DEFAULT 자세 복귀 (3s)")
    deadline = time.monotonic()
    t_start = time.monotonic()
    t_dur = 3.0
    while True:
        elapsed = time.monotonic() - t_start
        alpha = min(elapsed / t_dur, 1.0)
        q_now = ARM_TARGET + alpha * (ARM_DEFAULT - ARM_TARGET)
        deadline += dt
        _send(g1, q_now, chunk_id)
        chunk_id = (chunk_id % 255) + 1 or 1
        if alpha >= 1.0:
            break
        sleep = deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    print("[PASS] Phase 3 완료 — 복귀 확인")

    g1.stop()
    print("\n[DONE] 테스트 완료")


if __name__ == "__main__":
    main()
