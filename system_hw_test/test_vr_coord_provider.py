"""
VRCoordProvider 테스트 — PICO 연결 필요, G1 로봇 불필요.

실행:
    python3 system_hw_test/test_vr_coord_provider.py
"""
import os, sys, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


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

    PicoReaderProvider.reset()
    pico = PicoReaderProvider(); pico.start()
    vr = VRCoordProvider()

    # T-1: 연결
    _sec("T-1: PICO 연결 대기")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not pico.connected:
        time.sleep(0.2)
    if not _r("PICO connected", pico.connected):
        pico.stop(); sys.exit(1)

    time.sleep(0.5)  # EMA 안정화

    # T-2: VRCoordData 구조
    _sec("T-2: VRCoordData 필드 검증")
    data = vr.get_latest()
    ok = _r("get_latest() is not None", data is not None)
    if not ok:
        pico.stop(); sys.exit(1)

    _r("vr_3point_local_target shape (9,)", data.vr_3point_local_target.shape == (9,))
    _r("vr_3point_local_orn_target shape (12,)", data.vr_3point_local_orn_target.shape == (12,))
    _r("vr_3point_local_target finite", bool(np.all(np.isfinite(data.vr_3point_local_target))))
    _r("vr_3point_local_orn_target finite", bool(np.all(np.isfinite(data.vr_3point_local_orn_target))))

    # 쿼터니언 단위벡터 검사 (scalar-first: [w,x,y,z])
    quats = data.vr_3point_local_orn_target.reshape(3, 4)
    norms = np.linalg.norm(quats, axis=1)
    _r("3개 쿼터니언 |q|≈1", bool(np.all(np.abs(norms - 1.0) < 0.05)),
       f"norms={np.round(norms,4)}")

    # T-3: 값 출력 (물리적 타당성 육안 확인)
    _sec("T-3: 3-point local 위치 출력")
    tgt = data.vr_3point_local_target.reshape(3, 3)
    orn = data.vr_3point_local_orn_target.reshape(3, 4)
    for name, pos, q in zip(["L-Wrist", "R-Wrist", "Neck   "], tgt, orn):
        print(f"  {INFO} {name}  pos={np.round(pos,3)}  q(wxyz)={np.round(q,3)}")

    print(f"\n  {INFO} headset_pos_robot = {np.round(data.headset_pos_robot, 3)}")
    print(f"  {INFO} pelvis_pos_robot  = {np.round(data.pelvis_pos_robot, 3)}")
    print(f"  {INFO} headset_to_pelvis_y = {vr._headset_to_pelvis_y:.3f} m")

    # T-4: 5초 실시간 스트리밍
    _sec("T-4: 5초 실시간 모니터링")
    prev_ts = None
    count = 0
    last_print = time.monotonic()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        d = vr.get_latest()
        if d and (prev_ts is None or d.timestamp_ns != prev_ts):
            count += 1
            prev_ts = d.timestamp_ns
        if time.monotonic() - last_print > 1.0 and d:
            tgt = d.vr_3point_local_target.reshape(3, 3)
            print(f"  {INFO} LW={np.round(tgt[0],2)}  RW={np.round(tgt[1],2)}  NK={np.round(tgt[2],2)}")
            last_print = time.monotonic()
        time.sleep(0.005)
    avg = count / 5.0
    _r(f"5s 프레임 수신 ({count}개, {avg:.1f} fps)", count > 50)

    pico.stop()


if __name__ == "__main__":
    main()
