"""
ONNX dummy input 테스트 — 차렷 자세 기준

G1/PICO 연결 없이 ONNX encoder+decoder 에 직접 입력을 넣어
어떤 vr_3pt_orn 값이 action ≈ 0 (= DEFAULT_ANGLES 유지) 을 만드는지 찾는다.

--send 옵션: G1 에 실제로 명령 전송 (로봇이 해당 자세로 이동)
    Phase 1 (3s): DEFAULT_ANGLES 유지
    Phase 2 (5s): policy q_target 으로 보간
    Phase 3 (3s): DEFAULT_ANGLES 복귀

실행:
    source env.sh && uv run system_hw_test/test_onnx_dummy_input.py
    source env.sh && uv run system_hw_test/test_onnx_dummy_input.py --sweep-orn
    source env.sh && uv run system_hw_test/test_onnx_dummy_input.py --send
    source env.sh && uv run system_hw_test/test_onnx_dummy_input.py --send --orn identity
    source env.sh && uv run system_hw_test/test_onnx_dummy_input.py --send --orn measured
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from providers.policy_params import DEFAULT_ANGLES, ACTION_SCALE, KPS, KDS
from providers.g1_obs_provider import ALL_JOINT_NAMES

ARM_SHORT = [
    "wst_y", "wst_r", "wst_p",
    "L_shld_p", "L_shld_r", "L_shld_y", "L_elbow",
    "L_wr_r",  "L_wr_p",  "L_wr_y",
    "R_shld_p", "R_shld_r", "R_shld_y", "R_elbow",
    "R_wr_r",  "R_wr_p",  "R_wr_y",
]

# ── 측정된 차렷 자세 VR 위치 (yaw-only anchor 적용 후) ─────────────────────
VR_POS_ATTN = np.array([
    -0.110,  +0.366, -0.127,   # L-Wrist  (x, y, z)
    -0.079,  -0.336, -0.122,   # R-Wrist
     0.000,   0.000, +0.700,   # Neck
], dtype=np.float32)

OrnIdentity = np.array([
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
], dtype=np.float32)

OrnMeasured = np.array([
    0.863, 0.477, -0.141, -0.074,
    0.857, 0.494, -0.041, -0.122,
    1.0,   0.0,    0.0,    0.0,
], dtype=np.float32)


def _build_encoder_input(vr_pos, vr_orn, lower_pos_10f, lower_vel_10f, anchor_6d):
    enc = np.zeros(1762, dtype=np.float32)
    enc[0:4]     = [0, 1, 0, 0]
    enc[595:601] = anchor_6d
    enc[661:781] = lower_pos_10f
    enc[781:901] = lower_vel_10f
    enc[901:910] = vr_pos
    enc[910:922] = vr_orn
    return enc


def _build_decoder_input(token, pos_all_10f, vel_all_10f, gravity_10f):
    return np.concatenate([
        token,
        np.zeros(30, dtype=np.float32),   # ang_vel
        pos_all_10f,
        vel_all_10f,
        np.zeros(290, dtype=np.float32),  # action history
        gravity_10f,
    ]).astype(np.float32)


def run_policy(encoder, decoder, enc_input, dec_extra):
    """encoder → decoder → q_target (arm_only: 하체 q=0)."""
    token = encoder.run(None, {"obs_dict": enc_input[np.newaxis]})[0][0]
    dec_in = _build_decoder_input(token, *dec_extra)
    action = decoder.run(None, {"obs_dict": dec_in[np.newaxis]})[0][0]
    q_target = DEFAULT_ANGLES + action.astype(np.float32) * ACTION_SCALE
    q_target[:12] = 0.0   # arm_only: 하체 토크 없음
    return q_target, action


def print_result(label, q_target):
    arm = q_target[12:29]
    arm_def = DEFAULT_ANGLES[12:29]
    delta = arm - arm_def
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print("  " + "  ".join(f"{n:>8}" for n in ARM_SHORT))
    print("  " + "  ".join(f"{v:>8.3f}" for v in arm))
    print("Δ " + "  ".join(f"{d:>+8.3f}" for d in delta))
    big = [ARM_SHORT[i] for i, d in enumerate(delta) if abs(d) > 0.3]
    if big:
        print(f"  *** |delta|>0.3: {big}")


def _send(g1, q_target, chunk_id, hz=50.0):
    """q_target (29,) → publish_joint_cmd_low. 하체 kp=kd=0."""
    kps = KPS.copy(); kds = KDS.copy()
    kps[:12] = 0.0;   kds[:12] = 0.0
    cmd = {
        "joint_names": list(ALL_JOINT_NAMES),
        "q":      q_target.tolist(),
        "dq":     [0.0] * 29,
        "kp":     kps.tolist(),
        "kd":     kds.tolist(),
        "tau_ff": [0.0] * 29,
        "mode":   1,
        "weight": 1.0,
        "chunk_id":   chunk_id,
        "step_index": 0,
    }
    g1.publish_joint_cmd_low(cmd)


def send_to_robot(g1, q_target, hz=50.0):
    """
    Phase 1 (3s): DEFAULT_ANGLES 유지
    Phase 2 (5s): DEFAULT → q_target 선형 보간
    Phase 3 (3s): q_target → DEFAULT 선형 보간 (복귀)
    """
    dt = 1.0 / hz
    chunk_id = 1

    def tick(q):
        nonlocal chunk_id
        _send(g1, q, chunk_id)
        chunk_id = (chunk_id % 255) + 1 or 1

    arm_q_default = DEFAULT_ANGLES.copy()
    arm_q_default[:12] = 0.0

    # Phase 1
    print("[INFO] Phase 1: DEFAULT 자세 유지 (3s)")
    deadline = time.monotonic()
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        deadline += dt
        tick(arm_q_default)
        sleep = deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)

    # Phase 2
    print("[INFO] Phase 2: policy q_target 으로 이동 (5s)")
    deadline = time.monotonic()
    t_start = time.monotonic()
    while True:
        alpha = min((time.monotonic() - t_start) / 5.0, 1.0)
        q_now = arm_q_default + alpha * (q_target - arm_q_default)
        deadline += dt
        tick(q_now)
        if alpha >= 1.0:
            break
        sleep = deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    print("[PASS] Phase 2 완료")

    # Phase 3
    print("[INFO] Phase 3: DEFAULT 복귀 (3s)")
    deadline = time.monotonic()
    t_start = time.monotonic()
    while True:
        alpha = min((time.monotonic() - t_start) / 3.0, 1.0)
        q_now = q_target + alpha * (arm_q_default - q_target)
        deadline += dt
        tick(q_now)
        if alpha >= 1.0:
            break
        sleep = deadline - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    print("[PASS] Phase 3 완료 — 복귀")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-orn", action="store_true",
                        help="identity→measured 사이 10단계 sweep")
    parser.add_argument("--send", action="store_true",
                        help="G1 에 실제 명령 전송 (로봇 움직임)")
    parser.add_argument("--orn", choices=["identity", "measured"], default="identity",
                        help="--send 시 사용할 orientation (기본 identity)")
    parser.add_argument("--hz", type=float, default=50.0)
    args = parser.parse_args()

    import onnxruntime as ort
    model_dir = os.path.join(os.path.dirname(__file__), "..", "src", "policy")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    encoder = ort.InferenceSession(os.path.join(model_dir, "model_encoder.onnx"), opts)
    decoder = ort.InferenceSession(os.path.join(model_dir, "model_decoder.onnx"), opts)
    print("[INFO] 모델 로드 완료")

    # decoder 입력: DEFAULT_ANGLES 기준 standing 자세
    pos_all_10f   = np.tile(DEFAULT_ANGLES, 10).astype(np.float32)
    vel_all_10f   = np.zeros(290, dtype=np.float32)
    gravity_10f   = np.tile([0.0, 0.0, -1.0], 10).astype(np.float32)
    anchor_6d     = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
    lower_pos_10f = np.tile(DEFAULT_ANGLES[:12], 10).astype(np.float32)
    lower_vel_10f = np.zeros(120, dtype=np.float32)
    dec_extra     = (pos_all_10f, vel_all_10f, gravity_10f)

    # Case 1: identity
    enc1 = _build_encoder_input(VR_POS_ATTN, OrnIdentity, lower_pos_10f, lower_vel_10f, anchor_6d)
    q1, _ = run_policy(encoder, decoder, enc1, dec_extra)
    print_result("identity orn", q1)

    # Case 2: measured
    enc2 = _build_encoder_input(VR_POS_ATTN, OrnMeasured, lower_pos_10f, lower_vel_10f, anchor_6d)
    q2, _ = run_policy(encoder, decoder, enc2, dec_extra)
    print_result("measured orn", q2)

    # Case 3: baseline (zero vr)
    enc3 = _build_encoder_input(np.zeros(9, dtype=np.float32), OrnIdentity,
                                lower_pos_10f, lower_vel_10f, anchor_6d)
    q3, _ = run_policy(encoder, decoder, enc3, dec_extra)
    print_result("zero vr (baseline)", q3)

    if args.sweep_orn:
        print("\n\n[SWEEP] identity → measured")
        print(f"  {'alpha':>6}  " + "  ".join(f"{n:>8}" for n in ARM_SHORT))
        for i in range(11):
            alpha = i / 10.0
            orn_s = np.zeros(12, dtype=np.float32)
            for j in range(3):
                q_id = np.array([1, 0, 0, 0], dtype=np.float32)
                q_ms = OrnMeasured[j*4:(j+1)*4]
                q_lerp = sRot.from_quat(
                    np.stack([q_id, q_ms]), scalar_first=True
                ).mean(weights=[1-alpha, alpha]).as_quat(scalar_first=True)
                orn_s[j*4:(j+1)*4] = q_lerp.astype(np.float32)
            enc_s = _build_encoder_input(VR_POS_ATTN, orn_s, lower_pos_10f, lower_vel_10f, anchor_6d)
            q_s, _ = run_policy(encoder, decoder, enc_s, dec_extra)
            d = q_s[12:29] - DEFAULT_ANGLES[12:29]
            print(f"  {alpha:>6.1f}  " + "  ".join(f"{d_:>+8.3f}" for d_ in d))

    if args.send:
        orn_map = {"identity": OrnIdentity, "measured": OrnMeasured}
        orn = orn_map[args.orn]
        enc = _build_encoder_input(VR_POS_ATTN, orn, lower_pos_10f, lower_vel_10f, anchor_6d)
        q_send, _ = run_policy(encoder, decoder, enc, dec_extra)

        print(f"\n[SEND] orn={args.orn} 로 로봇 전송 시작")
        print_result(f"전송할 q_target ({args.orn})", q_send)
        print("\n로봇이 움직입니다. 3초 후 시작...")
        time.sleep(3.0)

        from providers.unitree_g1_provider import UnitreeG1Provider
        UnitreeG1Provider.reset()
        g1 = UnitreeG1Provider(); g1.start()
        print("[INFO] UnitreeG1Provider 시작, DDS 대기 1s...")
        time.sleep(1.0)

        try:
            send_to_robot(g1, q_send, hz=args.hz)
        except KeyboardInterrupt:
            print("\n[INFO] 중단됨 — DEFAULT 복귀 중...")
            q_def = DEFAULT_ANGLES.copy(); q_def[:12] = 0.0
            deadline = time.monotonic()
            for _ in range(int(3.0 * args.hz)):
                deadline += 1.0 / args.hz
                _send(g1, q_def, 1)
                sleep = deadline - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            g1.stop()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
