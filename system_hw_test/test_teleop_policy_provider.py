"""
TeleopPolicyProvider 테스트 — PICO 연결 필요, G1 로봇 선택.

encoder (1762-dim) → token (64,) → decoder (994-dim) → action (29,) 전체 파이프라인 검증.

실행:
    python3 system_hw_test/test_teleop_policy_provider.py
"""
import os, sys, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def _r(label, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}" + (f" — {detail}" if detail else ""))
    return ok


def _sec(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    import numpy as np
    from providers.pico_reader_provider import PicoReaderProvider
    from providers.vr_coord_provider import VRCoordProvider
    from providers.g1_obs_provider import G1ObsProvider
    from providers.teleop_encoder_input_provider import TeleopEncoderInputProvider, ENCODER_INPUT_DIM
    from providers.teleop_policy_provider import TeleopPolicyProvider, DECODER_INPUT_DIM, ACTION_DIM

    # PICO 시작
    PicoReaderProvider.reset()
    pico = PicoReaderProvider(); pico.start()
    vr_coord = VRCoordProvider()

    # G1 — 선택적
    g1_provider = None
    g1_obs = None
    try:
        from providers.unitree_g1_provider import UnitreeG1Provider
        UnitreeG1Provider.reset()
        g1_provider = UnitreeG1Provider()
        g1_provider.start()
        g1_obs = G1ObsProvider(g1_provider)
        print(f"  {INFO} UnitreeG1Provider 시작됨")
    except Exception as e:
        print(f"  {WARN} G1 Provider 시작 실패 ({e}) — G1 없이 계속")
        class _DummyG1:
            joint_state = type("C", (), {"last_seen_ts": 0.0, "value": None})()
            imu_base    = type("C", (), {"last_seen_ts": 0.0, "value": None})()
        g1_obs = G1ObsProvider(_DummyG1())

    enc_prov = TeleopEncoderInputProvider(vr_coord, g1_obs)

    # T-1: PICO 연결 대기
    _sec("T-1: PICO 연결 대기")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not pico.connected:
        time.sleep(0.2)
    if not _r("PICO connected", pico.connected):
        pico.stop(); sys.exit(1)
    time.sleep(0.5)

    # T-2: ONNX 모델 로드
    _sec("T-2: ONNX 모델 로드")
    try:
        policy = TeleopPolicyProvider(enc_prov, g1_obs)
        _r("TeleopPolicyProvider 생성", True)
    except Exception as e:
        _r("TeleopPolicyProvider 생성", False, str(e))
        pico.stop(); sys.exit(1)

    # T-3: 단일 추론 검증
    _sec("T-3: 단일 추론 검증")
    g1_obs.update()
    out = policy.build()
    _r("build() is not None", out is not None)
    if out is None:
        pico.stop(); sys.exit(1)

    _r(f"action shape == ({ACTION_DIM},)",
       out.action.shape == (ACTION_DIM,),
       str(out.action.shape))
    _r(f"q_target shape == ({ACTION_DIM},)",
       out.q_target.shape == (ACTION_DIM,),
       str(out.q_target.shape))
    _r("action finite", bool(np.all(np.isfinite(out.action))))
    _r("q_target finite", bool(np.all(np.isfinite(out.q_target))))
    _r(f"token shape == (64,)", out.token.shape == (64,))
    _r("inference_time_ms > 0", out.inference_time_ms > 0)
    print(f"  {INFO} vr_ok={out.vr_ok}  g1_ok={out.g1_ok}  "
          f"inf={out.inference_time_ms:.1f}ms")

    # T-4: q_target 값 출력 (MuJoCo 순서)
    _sec("T-4: q_target 값 [DEFAULT_ANGLES + action * ACTION_SCALE]")
    from providers.policy_params import DEFAULT_ANGLES, ACTION_SCALE
    joint_names = [
        "L_hip_pitch","L_hip_roll","L_hip_yaw","L_knee","L_ankle_p","L_ankle_r",
        "R_hip_pitch","R_hip_roll","R_hip_yaw","R_knee","R_ankle_p","R_ankle_r",
        "waist_yaw","waist_roll","waist_pitch",
        "L_shld_p","L_shld_r","L_shld_y","L_elbow","L_wrist_r","L_wrist_p","L_wrist_y",
        "R_shld_p","R_shld_r","R_shld_y","R_elbow","R_wrist_r","R_wrist_p","R_wrist_y",
    ]
    for i, name in enumerate(joint_names):
        print(f"  {INFO} [{i:2d}] {name:<14}  "
              f"default={DEFAULT_ANGLES[i]:+.3f}  "
              f"action={out.action[i]:+.4f}  "
              f"q_target={out.q_target[i]:+.4f}")

    # T-5: 연속 추론 속도 (50회)
    _sec("T-5: 연속 추론 50회 속도")
    times = []
    errors = 0
    for _ in range(50):
        g1_obs.update()
        t0 = time.monotonic()
        o = policy.build()
        times.append((time.monotonic() - t0) * 1000.0)
        if o is None or o.action.shape != (ACTION_DIM,) or not np.all(np.isfinite(o.action)):
            errors += 1
        time.sleep(0.005)
    avg_ms = sum(times) / len(times)
    max_ms = max(times)
    _r("50회 연속 성공", errors == 0, f"오류={errors}")
    _r(f"평균 추론 < 20ms", avg_ms < 20.0, f"avg={avg_ms:.1f}ms  max={max_ms:.1f}ms")
    print(f"  {INFO} avg={avg_ms:.2f}ms  max={max_ms:.2f}ms")

    if g1_provider is not None:
        try: g1_provider.stop()
        except Exception: pass
    pico.stop()
    print()


if __name__ == "__main__":
    main()
