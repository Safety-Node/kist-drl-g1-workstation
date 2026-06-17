"""
NavigationProvider 단독 테스트 — table 목적지로 제어 명령 출력 확인.

UnitreeG1Provider + NavigationProvider 만 기동하고 즉시 "table" 로
submit_nav_subtask() 를 호출한다. 이후 NavigationState 를 10Hz 로
출력하여 /bridge/cmd/vel 에 올바른 값이 나가는지 확인한다.

NX 측에서 동시에 확인:
    ros2 topic hz   /bridge/cmd/vel
    ros2 topic echo /bridge/cmd/vel

Usage:
    uv run system_hw_test/test_navigation.py [destination]

    destination : locations.json5 등록 키 (기본값: table)
"""

import signal
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import dotenv
dotenv.load_dotenv(dotenv_path=_ROOT / ".env")

from providers.unitree_g1_provider import UnitreeG1Provider
from providers.navigation_provider import NavigationProvider

PRINT_HZ = 10.0


def main() -> int:
    dest = sys.argv[1] if len(sys.argv) > 1 else "table"

    print(f"UnitreeG1Provider 시작 ...")
    g1 = UnitreeG1Provider()
    g1.start()

    print(f"NavigationProvider 시작 ...")
    nav = NavigationProvider()
    nav.start()

    stop = False
    def _on_sigint(sig, _):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _on_sigint)

    # 센서 데이터 대기
    print(f"센서 데이터 대기 ...")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if (not g1.occupancy.stale(now, 1.0) and
                not g1.location.stale(now, 1.0)):
            break
        time.sleep(0.2)
    else:
        print("[!] 타임아웃: 센서 데이터 미수신")
        nav.stop()
        g1.stop()
        return 1

    print(f"\n목적지 설정: {dest}")
    nav.submit_nav_subtask(dest)

    print(f"\n{'mode':<14} {'dist_m':>7} {'vx':>7} {'vy':>7} {'vyaw':>7}")
    print("-" * 50)

    period = 1.0 / PRINT_HZ
    while not stop:
        t0    = time.monotonic()
        state = nav.get_state()
        print(
            f"{state.mode:<14} "
            f"{state.dist_to_goal:>7.3f} "
            f"{state.vx:>7.3f} "
            f"{state.vy:>7.3f} "
            f"{state.vyaw:>7.3f}"
        )
        if state.mode == "ARRIVED":
            print(f"\n[도착] {dest}")
            break
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, period - elapsed))

    nav.stop()
    g1.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
