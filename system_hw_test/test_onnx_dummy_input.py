"""
ONNX dummy input 테스트 — 차렷 자세 기준

G1/PICO 연결 없이 ONNX encoder+decoder 에 직접 입력을 넣어
어떤 vr_3pt_orn 값이 action ≈ 0 (= DEFAULT_ANGLES 유지) 을 만드는지 찾는다.

실행:
    source env.sh && uv run system_hw_test/test_onnx_dummy_input.py
    source env.sh && uv run system_hw_test/test_onnx_dummy_input.py --sweep-orn
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from providers.policy_params import DEFAULT_ANGLES, ACTION_SCALE

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

# ── 기본 wrist orientation ──────────────────────────────────────────────────
OrnIdentity = np.array([
    1, 0, 0, 0,   # L-Wrist identity
    1, 0, 0, 0,   # R-Wrist identity
    1, 0, 0, 0,   # Neck   identity
], dtype=np.float32)

# ── 실측 wrist orientation (차렷 자세, yaw-only anchor 후) ─────────────────
OrnMeasured = np.array([
    0.863, 0.477, -0.141, -0.074,   # L-Wrist (measured)
    0.857, 0.494, -0.041, -0.122,   # R-Wrist (measured)
    1.0,   0.0,    0.0,    0.0,     # Neck    identity
], dtype=np.float32)


def _build_encoder_input(vr_pos, vr_orn, lower_pos_10f, lower_vel_10f, anchor_6d):
    """1762-dim encoder input 조립."""
    enc = np.zeros(1762, dtype=np.float32)
    enc[0:4]    = [0, 1, 0, 0]       # teleop mode
    enc[595:601] = anchor_6d
    enc[661:781] = lower_pos_10f
    enc[781:901] = lower_vel_10f
    enc[901:910] = vr_pos
    enc[910:922] = vr_orn
    return enc


def _build_decoder_input(token, pos_all_10f, vel_all_10f, gravity_10f):
    """994-dim decoder input 조립."""
    ang_vel_10f  = np.zeros(30, dtype=np.float32)
    action_10f   = np.zeros(290, dtype=np.float32)
    return np.concatenate([
        token,        # [0:64]
        ang_vel_10f,  # [64:94]
        pos_all_10f,  # [94:384]
        vel_all_10f,  # [384:674]
        action_10f,   # [674:964]
        gravity_10f,  # [964:994]
    ]).astype(np.float32)


def run_policy(encoder, decoder, enc_input, dec_extra):
    """encoder → token → decoder → action → q_target (arm_only 오버라이트 적용)."""
    token = encoder.run(None, {"obs_dict": enc_input[np.newaxis]})[0][0]
    dec_in = _build_decoder_input(token, *dec_extra)
    action = decoder.run(None, {"obs_dict": dec_in[np.newaxis]})[0][0]
    q_target = DEFAULT_ANGLES + action.astype(np.float32) * ACTION_SCALE

    # arm_only 오버라이트: 하체(0-11) q=0 (kp=kd=0 이므로 실제 토크 없음)
    q_target[:12] = 0.0

    return q_target, action


def print_result(label, q_target):
    arm = q_target[12:29]
    arm_def = DEFAULT_ANGLES[12:29]
    delta = arm - arm_def
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    header = "  " + "  ".join(f"{n:>8}" for n in ARM_SHORT)
    print(header)
    print("  " + "  ".join(f"{v:>8.3f}" for v in arm))
    print("  " + "  ".join(f"{d:>+8.3f}" for d in delta))
    big = [ARM_SHORT[i] for i, d in enumerate(delta) if abs(d) > 0.3]
    if big:
        print(f"  *** |delta|>0.3: {big}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-orn", action="store_true",
                        help="L/R wrist orn을 identity→measured 사이 sweep")
    args = parser.parse_args()

    import onnxruntime as ort
    model_dir = os.path.join(os.path.dirname(__file__), "..", "src", "policy")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    encoder = ort.InferenceSession(os.path.join(model_dir, "model_encoder.onnx"), opts)
    decoder = ort.InferenceSession(os.path.join(model_dir, "model_decoder.onnx"), opts)
    print("[INFO] 모델 로드 완료")

    # ── G1 default 자세 기준 decoder 입력 ────────────────────────────────────
    # 실제 서 있는 로봇과 동일하게: DEFAULT_ANGLES 를 10 프레임 반복
    pos_all_10f  = np.tile(DEFAULT_ANGLES, 10).astype(np.float32)   # (290,)
    vel_all_10f  = np.zeros(290, dtype=np.float32)
    gravity_10f  = np.tile([0.0, 0.0, -1.0], 10).astype(np.float32)
    anchor_6d    = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)   # identity
    lower_pos_10f = np.tile(DEFAULT_ANGLES[:12], 10).astype(np.float32)
    lower_vel_10f = np.zeros(120, dtype=np.float32)

    dec_extra = (pos_all_10f, vel_all_10f, gravity_10f)

    # ── Case 1: identity orientation ─────────────────────────────────────────
    enc1 = _build_encoder_input(VR_POS_ATTN, OrnIdentity, lower_pos_10f, lower_vel_10f, anchor_6d)
    q1, _ = run_policy(encoder, decoder, enc1, dec_extra)
    print_result("identity orn (차렷 pos + orn=[1,0,0,0])", q1)

    # ── Case 2: measured orientation ─────────────────────────────────────────
    enc2 = _build_encoder_input(VR_POS_ATTN, OrnMeasured, lower_pos_10f, lower_vel_10f, anchor_6d)
    q2, _ = run_policy(encoder, decoder, enc2, dec_extra)
    print_result("measured orn (차렷 실측값)", q2)

    # ── Case 3: all-zero vr input (baseline) ─────────────────────────────────
    enc3 = _build_encoder_input(
        np.zeros(9, dtype=np.float32), OrnIdentity,
        lower_pos_10f, lower_vel_10f, anchor_6d)
    q3, _ = run_policy(encoder, decoder, enc3, dec_extra)
    print_result("zero vr pos + identity orn (baseline)", q3)

    if args.sweep_orn:
        print("\n\n[SWEEP] identity → measured 사이 10단계")
        print(f"  {'alpha':>6}  " + "  ".join(f"{n:>8}" for n in ARM_SHORT))
        for i in range(11):
            alpha = i / 10.0
            orn_sweep = np.zeros(12, dtype=np.float32)
            for j in range(3):
                q_id = np.array([1, 0, 0, 0], dtype=np.float32)
                q_ms = OrnMeasured[j*4:(j+1)*4]
                q_lerp = sRot.from_quat(
                    np.stack([q_id, q_ms]), scalar_first=True
                ).mean(weights=[1-alpha, alpha]).as_quat(scalar_first=True)
                orn_sweep[j*4:(j+1)*4] = q_lerp.astype(np.float32)

            enc_s = _build_encoder_input(VR_POS_ATTN, orn_sweep, lower_pos_10f, lower_vel_10f, anchor_6d)
            q_s, _ = run_policy(encoder, decoder, enc_s, dec_extra)
            arm_d = q_s[12:29] - DEFAULT_ANGLES[12:29]
            print(f"  {alpha:>6.1f}  " + "  ".join(f"{d:>+8.3f}" for d in arm_d))

    print("\n[DONE]")


if __name__ == "__main__":
    main()
