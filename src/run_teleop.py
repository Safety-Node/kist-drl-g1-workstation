"""
VR Teleop 진입점

PICO VR 헤드셋 + G1 관절/IMU → encoder(1762) → decoder(994) → q_target[29]
→ JointCmdChunk → /bridge/cmd/low → onboard motor_controller → Unitree SDK PD 제어

실행:
    python src/run_teleop.py
    python src/run_teleop.py --dry-run       # 연결만 확인, 제어 루프 미실행
    python src/run_teleop.py --hz 30         # 제어 주기 변경

종료: Ctrl-C
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="run_teleop")
    parser.add_argument("--hz", type=float, default=50.0, help="제어 주기 (기본 50 Hz)")
    parser.add_argument("--pico-timeout", type=float, default=15.0, help="PICO 연결 대기 타임아웃 (초)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", help="프로바이더 초기화만 하고 제어 루프 미실행")
    parser.add_argument("--arm-only", action="store_true", help="팔 관절(waist+arms, 17개)만 전송 — 하체는 loco SDK 담당")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    logger = logging.getLogger("run_teleop")

    from providers.pico_reader_provider import PicoReaderProvider
    from providers.vr_coord_provider import VRCoordProvider
    from providers.unitree_g1_provider import UnitreeG1Provider
    from providers.g1_obs_provider import G1ObsProvider
    from providers.teleop_control_loop import TeleopControlLoop

    # ── 1. PICO ───────────────────────────────────────────────────────
    logger.info("PICO 초기화 중...")
    PicoReaderProvider.reset()
    pico = PicoReaderProvider()
    pico.start()

    logger.info("PICO 연결 대기 (최대 %.0fs)...", args.pico_timeout)
    deadline = time.monotonic() + args.pico_timeout
    while time.monotonic() < deadline and not pico.connected:
        time.sleep(0.2)
    if not pico.connected:
        logger.error("PICO 연결 실패 — 헤드셋 전원 및 서비스 확인")
        pico.stop()
        return 1
    logger.info("PICO 연결됨")

    vr_coord = VRCoordProvider()

    # ── 2. G1 ─────────────────────────────────────────────────────────
    logger.info("UnitreeG1Provider 초기화 중...")
    UnitreeG1Provider.reset()
    g1 = UnitreeG1Provider()
    g1.start()
    time.sleep(1.0)   # DDS discovery 대기
    logger.info("UnitreeG1Provider 시작됨")

    g1_obs = G1ObsProvider(g1)

    # ── 3. dry-run ────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("dry-run: 모든 프로바이더 초기화 완료, 제어 루프 미실행")
        try:
            g1.stop()
        except Exception:
            pass
        pico.stop()
        return 0

    # ── 4. 제어 루프 ──────────────────────────────────────────────────
    loop = TeleopControlLoop(g1, vr_coord, g1_obs, control_hz=args.hz, arm_only=args.arm_only)

    def _on_signal(sig, _frame):
        logger.warning("신호 %d 수신 — 종료 중...", sig)
        loop.stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    logger.info("teleop 제어 루프 시작 (%.0f Hz) — Ctrl-C 로 종료", args.hz)
    loop.run()   # 블로킹, loop.stop() 또는 Ctrl-C 까지

    # ── 5. 정리 ───────────────────────────────────────────────────────
    logger.info("정리 중... (총 %d 스텝)", loop.step_count)
    try:
        g1.stop()
    except Exception:
        pass
    pico.stop()
    logger.info("종료 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
