"""
PicoReaderProvider 하드웨어 연결 테스트

사전 조건:
    1. PICO 헤드셋 전원 ON + XRoboToolkit 앱 실행 중
    2. xrobotoolkit_sdk 설치 완료 (install_scripts/install_pico.sh 참고)
    3. /opt/apps/roboticsservice/runService.sh 실행 중

실행:
    python3 system_hw_test/test_pico_reader_provider.py
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

_CONNECT_TIMEOUT_S = 10.0


def _result(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}" + (f" — {detail}" if detail else ""))
    return ok


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# T-1: 연결 확인 (헤드셋 포즈 수신 기준)
# ---------------------------------------------------------------------------
def test_connection(provider) -> bool:
    _section("T-1: PICO 연결 확인 (headset pose 수신 기준)")
    print(f"  {INFO} 최대 {_CONNECT_TIMEOUT_S:.0f}s 대기 중 …")

    deadline = time.monotonic() + _CONNECT_TIMEOUT_S
    while time.monotonic() < deadline:
        if provider.connected:
            break
        time.sleep(0.2)

    return _result("connected == True", provider.connected,
                   "헤드셋 전원 / XRoboToolkit 앱 / PC Service 확인" if not provider.connected else "")


# ---------------------------------------------------------------------------
# T-2: 헤드셋 포즈 유효성
# ---------------------------------------------------------------------------
def test_headset_pose(provider) -> bool:
    _section("T-2: headset_pose 유효성")
    import numpy as np

    data = provider.get_latest()
    if data is None:
        return _result("데이터 수신", False, "T-1 실패")

    ok_shape  = _result("shape == (7,)", data.headset_pose.shape == (7,),
                        str(data.headset_pose.shape))
    ok_finite = _result("nan/inf 없음", bool(np.all(np.isfinite(data.headset_pose))))

    quat  = data.headset_pose[3:]
    qnorm = float(np.linalg.norm(quat))
    ok_quat = _result("쿼터니언 |q|≈1", abs(qnorm - 1.0) < 0.05, f"|q|={qnorm:.4f}")

    pos = data.headset_pose[:3]
    print(f"  {INFO} headset pos={np.round(pos, 3)}  quat={np.round(quat, 4)}")

    return ok_shape and ok_finite and ok_quat


# ---------------------------------------------------------------------------
# T-3: 컨트롤러 포즈 출력
# ---------------------------------------------------------------------------
def test_controller_poses(provider) -> bool:
    _section("T-3: 컨트롤러 포즈 (L / R)")
    import numpy as np

    data = provider.get_latest()
    if data is None:
        return _result("데이터 수신", False, "T-1 실패")

    results = []
    for name, pose in [("L-Controller", data.left_controller_pose),
                       ("R-Controller", data.right_controller_pose)]:
        ok_shape  = pose.shape == (7,)
        ok_finite = bool(np.all(np.isfinite(pose)))
        qnorm     = float(np.linalg.norm(pose[3:]))
        pos       = pose[:3]
        # 컨트롤러가 sleep 이면 zeros 일 수 있어 쿼터니언 체크는 WARN 으로만
        identity  = np.allclose(pose, [0, 0, 0, 0, 0, 0, 1], atol=1e-6)
        status    = "sleep/not held" if identity else f"pos={np.round(pos, 3)}  |q|={qnorm:.4f}"
        results.append(_result(f"{name} shape (7,) & finite", ok_shape and ok_finite, status))

    return all(results)


# ---------------------------------------------------------------------------
# T-4: FPS / 타임스탬프 단조 증가
# ---------------------------------------------------------------------------
def test_timing(provider) -> bool:
    _section("T-4: FPS & 타임스탬프 단조 증가")

    # EMA 안정화 대기
    time.sleep(1.0)

    d0 = provider.get_latest()
    if d0 is None:
        return _result("데이터 수신", False, "T-1 실패")

    time.sleep(0.3)
    d1 = provider.get_latest()

    ok_mono = _result("timestamp_ns 증가", d1.timestamp_ns > d0.timestamp_ns,
                      f"Δ={(d1.timestamp_ns - d0.timestamp_ns) * 1e-6:.1f} ms")
    ok_fps  = _result("20 ≤ fps ≤ 200", 20.0 <= d1.fps <= 200.0,
                      f"fps={d1.fps:.1f} Hz  dt={d1.dt*1000:.2f} ms")
    return ok_mono and ok_fps


# ---------------------------------------------------------------------------
# T-5: 컨트롤러 입력 범위
# ---------------------------------------------------------------------------
def test_controller_input(provider) -> bool:
    _section("T-5: ControllerData 값 범위")

    ctrl = provider.get_controller()
    if ctrl is None:
        return _result("컨트롤러 데이터 수신", False, "T-1 실패")

    results = []
    for label, val in [
        ("left_trigger",  ctrl.left_trigger),
        ("right_trigger", ctrl.right_trigger),
        ("left_grip",     ctrl.left_grip),
        ("right_grip",    ctrl.right_grip),
    ]:
        results.append(_result(f"{label} ∈ [0,1]", 0.0 <= val <= 1.0, f"{val:.3f}"))

    results.append(_result("left_joystick  2-tuple", len(ctrl.left_joystick) == 2,
                            str(ctrl.left_joystick)))
    results.append(_result("right_joystick 2-tuple", len(ctrl.right_joystick) == 2,
                            str(ctrl.right_joystick)))

    print(f"  {INFO} 버튼: A={ctrl.btn_a} B={ctrl.btn_b} X={ctrl.btn_x} Y={ctrl.btn_y} "
          f"L_menu={ctrl.left_menu} R_menu={ctrl.right_menu}")

    return all(results)


# ---------------------------------------------------------------------------
# T-6: 실시간 스트리밍 5초 모니터링
# ---------------------------------------------------------------------------
def test_streaming(provider) -> bool:
    _section("T-6: 실시간 스트리밍 5초 모니터링")
    import numpy as np

    print(f"  {INFO} 5초간 데이터 수신 확인 …")
    MONITOR_S      = 5.0
    PRINT_INTERVAL = 1.0

    frame_count = 0
    prev_ts_ns  = None
    last_print  = time.monotonic()
    deadline    = time.monotonic() + MONITOR_S

    while time.monotonic() < deadline:
        data = provider.get_latest()
        if data is None:
            time.sleep(0.01)
            continue

        if prev_ts_ns is None or data.timestamp_ns != prev_ts_ns:
            frame_count += 1
            prev_ts_ns = data.timestamp_ns

        now = time.monotonic()
        if now - last_print >= PRINT_INTERVAL:
            hpos = data.headset_pose[:3]
            lpos = data.left_controller_pose[:3]
            rpos = data.right_controller_pose[:3]
            print(f"  {INFO} fps={data.fps:.1f}  "
                  f"HMD={np.round(hpos,2)}  "
                  f"L={np.round(lpos,2)}  R={np.round(rpos,2)}")
            last_print = now

        time.sleep(0.005)

    avg_fps = frame_count / MONITOR_S
    ok = _result(f"5s 동안 프레임 수신 ({frame_count}개, 평균 {avg_fps:.1f} fps)",
                 frame_count > 50)
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 60)
    print("  PicoReaderProvider 하드웨어 테스트")
    print("=" * 60)

    try:
        import xrobotoolkit_sdk
    except ImportError:
        print(f"\n  {FAIL} xrobotoolkit_sdk 미설치 — 테스트 중단")
        print("  install_scripts/install_pico.sh 를 먼저 실행하세요.")
        sys.exit(1)

    from providers.pico_reader_provider import PicoReaderProvider
    PicoReaderProvider.reset()
    provider = PicoReaderProvider()
    provider.start()

    results: dict[str, bool] = {}
    try:
        results["T-1 connection"] = test_connection(provider)

        if not results["T-1 connection"]:
            print(f"\n  {FAIL} PICO 연결 실패 — 이후 테스트 중단")
        else:
            results["T-2 headset pose"]       = test_headset_pose(provider)
            results["T-3 controller poses"]   = test_controller_poses(provider)
            results["T-4 timing"]             = test_timing(provider)
            results["T-5 controller input"]   = test_controller_input(provider)
            results["T-6 streaming"]          = test_streaming(provider)
    finally:
        provider.stop()

    _section("SUMMARY")
    all_pass = True
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  {tag} {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"  {PASS} 모든 테스트 통과")
        sys.exit(0)
    else:
        print(f"  {FAIL} 일부 테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
