"""
TeleopEncoderInputProvider 테스트 — PICO 연결 필요, G1 로봇 선택.

G1 없을 때: G1 관측값은 zeros 로 채워진다 (g1_ok=False 표시).
G1 있을 때: 실제 하체 관절 상태로 채워진다.

실행:
    python3 system_hw_test/test_teleop_encoder_input.py
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
    from providers.teleop_encoder_input_provider import (
        TeleopEncoderInputProvider, ENCODER_INPUT_DIM,
    )

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

    # T-1: PICO 연결
    _sec("T-1: PICO 연결 대기")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not pico.connected:
        time.sleep(0.2)
    if not _r("PICO connected", pico.connected):
        pico.stop(); sys.exit(1)
    time.sleep(0.5)

    # T-2: encoder input 차원 검증
    _sec("T-2: encoder input 차원 검증")
    g1_obs.update()
    enc = enc_prov.build()
    _r("build() is not None", enc is not None)
    if enc is None:
        pico.stop(); sys.exit(1)

    _r(f"encoder_input shape == ({ENCODER_INPUT_DIM},)",
       enc.encoder_input.shape == (ENCODER_INPUT_DIM,),
       str(enc.encoder_input.shape))
    _r("encoder_input dtype float32", enc.encoder_input.dtype == np.float32)
    _r("encoder_input finite", bool(np.all(np.isfinite(enc.encoder_input))))
    print(f"  {INFO} vr_ok={enc.vr_ok}  g1_ok={enc.g1_ok}")

    # T-3: 활성 필드 검증
    _sec("T-3: 활성 필드(obs_dict) 검증")
    expected = {
        "encoder_mode_4":                                   4,
        "motion_anchor_orientation":                        6,
        "motion_joint_positions_lowerbody_10frame_step5":  120,
        "motion_joint_velocities_lowerbody_10frame_step5": 120,
        "vr_3point_local_target":                           9,
        "vr_3point_local_orn_target":                      12,
    }
    for key, dim in expected.items():
        arr = enc.obs_dict[key]
        ok = arr.shape == (dim,) and np.all(np.isfinite(arr))
        _r(f"{key} ({dim},)", ok, f"shape={arr.shape}")

    # T-4: mode one-hot 확인
    _sec("T-4: encoder_mode_4 = [0,1,0,0]")
    mode = enc.obs_dict["encoder_mode_4"]
    _r("mode = [0,1,0,0]", list(mode) == [0.0, 1.0, 0.0, 0.0], str(mode))

    # T-5: 비활성 필드(zeros) 검증
    _sec("T-5: 비활성 필드 zeros 검증")
    buf = enc.encoder_input
    inactive_slices = [
        ("motion_joint_positions_10frame_step5",    4,    294),
        ("motion_joint_velocities_10frame_step5",  294,   584),
        ("motion_root_z_position_10frame_step5",   584,   594),
        ("motion_root_z_position",                 594,   595),
        ("smpl_joints_10frame_step1",              922,  1642),
        ("smpl_anchor_orientation_10frame_step1", 1642,  1702),
        ("motion_joint_positions_wrists_10f_s1",  1702,  1762),
    ]
    for name, s, e in inactive_slices:
        ok = np.all(buf[s:e] == 0.0)
        _r(f"{name} zeros", ok, f"nonzero={np.count_nonzero(buf[s:e])}")

    # T-6: 연속 build() 안정성 (10회)
    _sec("T-6: 연속 build() 10회 안정성")
    errors = 0
    for _ in range(10):
        g1_obs.update()
        e2 = enc_prov.build()
        if e2 is None or e2.encoder_input.shape != (ENCODER_INPUT_DIM,):
            errors += 1
        time.sleep(0.02)
    _r("10회 연속 성공", errors == 0, f"오류={errors}")

    if g1_provider is not None:
        try: g1_provider.stop()
        except Exception: pass
    pico.stop()
    print()


if __name__ == "__main__":
    main()
