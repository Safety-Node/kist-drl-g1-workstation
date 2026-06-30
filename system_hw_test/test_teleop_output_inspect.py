"""
Test 2: teleop 파이프라인 출력값 점검 — 명령 전송 없음

PICO + (선택) G1 연결 상태에서 policy 추론 결과 q_target 을 출력.
실제로 /bridge/cmd/low 에는 아무것도 보내지 않는다.

확인 항목:
  - q_target 이 NaN/Inf 가 아닌지
  - q_target 이 DEFAULT_ANGLES 에서 합리적 범위 내인지 (|delta| < 0.5 rad)
  - VR 입력이 실제로 들어오는지 (vr_ok)
  - G1 관절 상태가 들어오는지 (g1_ok)

실행:
    source env.sh && uv run system_hw_test/test_teleop_output_inspect.py
    source env.sh && uv run system_hw_test/test_teleop_output_inspect.py --count 20  # 20회만
"""

import os, sys, time, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

import numpy as np

ARM_SHORT = [
    "wst_y", "wst_r", "wst_p",
    "L_shld_p", "L_shld_r", "L_shld_y", "L_elbow", "L_wr_r", "L_wr_p", "L_wr_y",
    "R_shld_p", "R_shld_r", "R_shld_y", "R_elbow", "R_wr_r", "R_wr_p", "R_wr_y",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=0, help="출력 횟수 (0=무한)")
    parser.add_argument("--hz", type=float, default=2.0, help="출력 주기 (기본 2Hz = 0.5s 간격)")
    args = parser.parse_args()

    from providers.pico_reader_provider import PicoReaderProvider
    from providers.vr_coord_provider import VRCoordProvider
    from providers.g1_obs_provider import G1ObsProvider
    from providers.teleop_encoder_input_provider import TeleopEncoderInputProvider
    from providers.teleop_policy_provider import TeleopPolicyProvider
    from providers.policy_params import DEFAULT_ANGLES

    # ── PICO ──────────────────────────────────────────────────────────────
    PicoReaderProvider.reset()
    pico = PicoReaderProvider(); pico.start()
    vr_coord = VRCoordProvider()

    print("[INFO] PICO 연결 대기 (최대 10s)...")
    t_end = time.monotonic() + 10.0
    while time.monotonic() < t_end and not pico.connected:
        time.sleep(0.2)
    if not pico.connected:
        print("[FAIL] PICO 연결 실패")
        pico.stop(); sys.exit(1)
    print("[PASS] PICO 연결됨")

    # ── G1 (선택) ─────────────────────────────────────────────────────────
    g1_obs = None
    try:
        from providers.unitree_g1_provider import UnitreeG1Provider
        UnitreeG1Provider.reset()
        g1 = UnitreeG1Provider(); g1.start()
        time.sleep(1.0)
        g1_obs = G1ObsProvider(g1)
        print("[INFO] UnitreeG1Provider 연결됨")
    except Exception as e:
        print(f"[WARN] G1 연결 실패 ({e}) — 더미 obs 사용")
        class _DummyG1:
            joint_state = type("C", (), {"last_seen_ts": 0.0, "value": None})()
            imu_base    = type("C", (), {"last_seen_ts": 0.0, "value": None})()
        g1_obs = G1ObsProvider(_DummyG1())

    # ── Policy ────────────────────────────────────────────────────────────
    enc_prov = TeleopEncoderInputProvider(vr_coord, g1_obs)
    policy = TeleopPolicyProvider(enc_prov, g1_obs)
    print("[INFO] 모델 로드 완료\n")

    dt = 1.0 / args.hz
    step = 0
    deadline = time.monotonic()

    HEADER = f"{'step':>5}  {'vr':>3} {'g1':>3}  " + "  ".join(f"{n:>8}" for n in ARM_SHORT)
    SEP    = "-" * len(HEADER)

    try:
        while args.count == 0 or step < args.count:
            deadline += dt
            g1_obs.update()
            out = policy.build()

            if out is None:
                print(f"  step={step:4d}  policy.build() returned None")
            else:
                arm_q   = out.q_target[12:29]
                arm_def = DEFAULT_ANGLES[12:29]
                delta   = arm_q - arm_def
                bad_nan = not np.all(np.isfinite(arm_q))
                bad_big = np.any(np.abs(delta) > 0.8)

                if step % 20 == 0:
                    print(SEP)
                    print(HEADER)
                    print(f"{'':>5}  {'':>3} {'':>3}  " + "  ".join(f"{'default':>8}" for _ in ARM_SHORT))
                    print(f"{'':>5}  {'':>3} {'':>3}  " + "  ".join(f"{v:>8.3f}" for v in arm_def))
                    print(SEP)

                flag = " *** NaN/Inf!" if bad_nan else (" *** |delta|>0.8!" if bad_big else "")
                vr_s = "OK" if out.vr_ok else "--"
                g1_s = "OK" if out.g1_ok else "--"
                print(f"  {step:4d}  {vr_s:>3} {g1_s:>3}  " +
                      "  ".join(f"{v:>8.3f}" for v in arm_q) + flag)

                # delta 행
                print(f"  {'Δ':>4}  {'':>3} {'':>3}  " +
                      "  ".join(f"{d:>+8.3f}" for d in delta))

            step += 1
            sleep = deadline - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass

    pico.stop()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
