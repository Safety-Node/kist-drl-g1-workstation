"""
Live STT verification (G1 provider 경유 버전): UnitreeG1Provider → STTProvider → text.

verify_stt_live.py 와 동일한 입출력이지만, **TASK-41 어댑터(_AudioBridgeNode)를
제거**하고 정식 경로로 검증한다. STTProvider 가 ``register_audio_callback`` 으로
UnitreeG1Provider 의 audio_pcm 콜백에 직접 붙는다 — verify_full_loop.py 와 동일한
오디오 경로. 둘 사이를 비교해 G1 provider 의 audio sub/디스패치가 정상인지 격리
진단하는 용도다.

대비표 (어댑터 버전 vs 본 스크립트):
    /bridge/sensors/audio_pcm 구독:
        verify_stt_live.py     → _AudioBridgeNode (메인 스레드 rclpy.spin)
        verify_stt_live_g1.py  → UnitreeG1Provider (MultiThreadedExecutor 데몬 스레드)
    STT 로 PCM 전달:
        verify_stt_live.py     → stt._on_audio_chunk() 직접 호출
        verify_stt_live_g1.py  → g1.register_audio_callback(stt._on_audio_chunk)
                                  (STTProvider.start() 가 자동 등록)

전제:
    source env.sh                       # Jetson ROS2 환경 (rclpy, g1_onboard_msgs)
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json   # GOOGLE_CLOUD 백엔드
실행:
    uv run python scripts/verify_stt_live_g1.py                  # 기본 google, ko-KR
    uv run python scripts/verify_stt_live_g1.py --backend dummy  # 파이프라인 연결만 점검
"""

import argparse
import logging
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import dotenv
dotenv.load_dotenv(dotenv_path=_ROOT / ".env")

from providers.stt_provider import STTProvider, STTConfig, STTBackend, TranscriptEvent
from providers.unitree_g1_provider import UnitreeG1Provider


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["google", "dummy"], default="google")
    ap.add_argument("--lang", default="ko-KR")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    backend = STTBackend.GOOGLE_CLOUD if args.backend == "google" else STTBackend.DUMMY

    # ── 와이어링 (run.py 와 동일한 순서: G1 FIRST → STT 가 같은 싱글턴을 본다) ──
    # singleton 잔재 방지
    UnitreeG1Provider.reset()  # type: ignore[attr-defined]
    STTProvider.reset()        # type: ignore[attr-defined]

    g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig(
        backend=backend,
        language_code=args.lang,
        sample_rate_hz=16000,
        interim_results=False,     # is_final 만 방출
    ))

    # ── 진단 probe — 첫 audio 청크 / 누적 카운트 (G1 audio path 가 살아있는지) ──
    chunk = {"n": 0, "bytes": 0, "first_ts": None, "last_ts": None}

    def _audio_probe(pcm: bytes, ts: float) -> None:
        n = chunk["n"] + 1
        chunk["n"] = n
        chunk["bytes"] += len(pcm)
        if chunk["first_ts"] is None:
            chunk["first_ts"] = ts
            logging.info(
                "G1 first audio chunk reached STT path: %d bytes (audio path OK)",
                len(pcm),
            )
        chunk["last_ts"] = ts

    g1.register_audio_callback(_audio_probe)

    # ── 검증 지점: transcript 가 실제로 나오는지 + 타이밍 ──────────────
    count = {"n": 0}
    tr_times = []          # 첫 오디오 기준 transcript 도착 시각(s)

    def on_transcript(ev: TranscriptEvent) -> None:
        count["n"] += 1
        now = time.monotonic()
        rel = (now - chunk["first_ts"]) if chunk["first_ts"] is not None else 0.0
        tr_times.append(rel)
        logging.info(
            "TRANSCRIPT #%d  t=+%.2fs (final=%s, conf=%s): %r",
            count["n"], rel, ev.is_final, ev.confidence, ev.text,
        )

    stt.register_transcript_callback(on_transcript)

    # ── 기동 (G1 → STT) ──────────────────────────────────────────────────
    g1.start()
    stt.start()
    if not getattr(stt, "_audio_cb_registered", False):
        logging.error(
            "전제 미충족: UnitreeG1Provider.register_audio_callback 미구현(TASK-41). "
            "이 스크립트는 G1 provider 경유 전용 — TASK-41 머지 전에는 "
            "verify_stt_live.py (어댑터 버전) 를 사용하세요.")
        g1.stop()
        return 2

    logging.info(
        "STT started (backend=%s, state=%s) — 말해보세요. Ctrl-C 종료.",
        args.backend, stt.state.value,
    )

    # ── 메인: 그냥 대기 (G1 provider 가 자기 데몬 스레드에서 spin 중) ─────
    stop_event = threading.Event()
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stt.stop()
        g1.stop()
        # ── 타이밍 요약 ───────────────────────────────────────────────
        logging.info("---- stt timing ----")
        if chunk["first_ts"] is not None and chunk["last_ts"] is not None:
            span = chunk["last_ts"] - chunk["first_ts"]
            rate = (chunk["n"] - 1) / span if span > 0 else 0.0
            logging.info("  수신 오디오   : %d청크 / %d bytes, 실측 %.1fHz (기대 ~50Hz)",
                         chunk["n"], chunk["bytes"], rate)
        else:
            logging.warning(
                "  수신 오디오   : 0 — G1 provider audio_pcm 콜백 미동작 "
                "(comm_bridge / CYCLONEDDS_URI / 도메인 확인)")
        logging.info("  transcript    : 총 %d건", count["n"])
        if len(tr_times) >= 2:
            gaps = [tr_times[i] - tr_times[i - 1] for i in range(1, len(tr_times))]
            logging.info("  transcript 간격: 평균 %.2fs (첫 발화→첫 transcript +%.2fs)",
                         statistics.mean(gaps), tr_times[0])
        elif tr_times:
            logging.info("  첫 발화→첫 transcript: +%.2fs", tr_times[0])
        logging.info("done. total transcripts=%d, final state=%s",
                     count["n"], stt.state.value)
    return 0 if count["n"] > 0 else 1   # 한 건도 못 받으면 비정상 종료


if __name__ == "__main__":
    raise SystemExit(main())
