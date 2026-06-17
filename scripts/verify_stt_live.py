"""
Live STT verification: /bridge/sensors/audio_pcm -> STTProvider -> text.

UnitreeG1Provider 가 /bridge/sensors/audio_pcm 를 단일 구독하고,
STTProvider 가 register_audio_callback() 으로 fan-out 받는다.
이전의 _AudioBridgeNode (직접 구독 scaffold) 는 TASK-41 완료로 제거됨.

전제:
    source env.sh
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json   # GOOGLE_CLOUD 백엔드
실행:
    uv run python scripts/verify_stt_live.py
    uv run python scripts/verify_stt_live.py --backend dummy
    uv run python scripts/verify_stt_live.py --local          # 로봇 없이 로컬 마이크 테스트
"""

import argparse
import logging
import os
import statistics
import sys
import time
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from providers.unitree_g1_provider import UnitreeG1Provider
from providers.stt_provider import STTProvider, STTConfig, STTBackend, TranscriptEvent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["google", "dummy"], default="google")
    ap.add_argument("--lang", default="ko-KR")
    ap.add_argument("--local", action="store_true",
                    help="로봇 없이 로컬 mic_publisher 와 함께 실행. eno2 CycloneDDS 설정 무시.")
    args = ap.parse_args()

    if args.local:
        os.environ.pop("CYCLONEDDS_URI", None)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    backend = STTBackend.GOOGLE_CLOUD if args.backend == "google" else STTBackend.DUMMY

    # 싱글턴 초기화 — 재실행 시 이전 인스턴스 정리
    UnitreeG1Provider.reset()
    STTProvider.reset()

    g1  = UnitreeG1Provider()
    stt = STTProvider(STTConfig(
        backend=backend,
        language_code=args.lang,
        sample_rate_hz=16000,
        interim_results=False,
    ))

    # ── 오디오 수신 통계 (UnitreeG1Provider 콜백에 추가) ──────────────────
    audio_stats: dict = {"chunks": 0, "bytes": 0, "first_ts": None, "last_ts": None}

    def _on_audio_stats(pcm: bytes, ts: float) -> None:
        audio_stats["chunks"] += 1
        audio_stats["bytes"]  += len(pcm)
        if audio_stats["first_ts"] is None:
            audio_stats["first_ts"] = ts
        audio_stats["last_ts"] = ts

    g1.register_audio_callback(_on_audio_stats)

    # ── transcript 수신 통계 ──────────────────────────────────────────────
    count    = {"n": 0}
    tr_times: list[float] = []

    def on_transcript(ev: TranscriptEvent) -> None:
        count["n"] += 1
        now = time.monotonic()
        first = audio_stats["first_ts"]
        rel   = (now - first) if first else 0.0
        tr_times.append(rel)
        logging.info(
            "TRANSCRIPT #%d  t=+%.2fs (final=%s, conf=%s): %r",
            count["n"], rel, ev.is_final, ev.confidence, ev.text,
        )

    stt.register_transcript_callback(on_transcript)

    # ── 시작 ──────────────────────────────────────────────────────────────
    # g1.start() : rclpy.init() + DDS 노드 생성 + spin thread 시작
    # stt.start(): register_audio_callback() → UnitreeG1Provider fan-out 등록
    g1.start()
    stt.start()

    logging.info(
        "STT started (backend=%s, state=%s) — 말해보세요. Ctrl-C 종료.",
        args.backend, stt.state.value,
    )

    stop_event = threading.Event()
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stt.stop()
        g1.unregister_audio_callback(_on_audio_stats)
        g1.stop()

        # ── 타이밍 요약 ──────────────────────────────────────────────────
        logging.info("---- stt timing ----")
        first = audio_stats["first_ts"]
        last  = audio_stats["last_ts"]
        if first and last:
            span = last - first
            rate = (audio_stats["chunks"] - 1) / span if span > 0 else 0.0
            logging.info(
                "  수신 오디오   : %d청크 / %d bytes, 실측 %.1fHz (기대 ~50Hz)",
                audio_stats["chunks"], audio_stats["bytes"], rate,
            )
        else:
            logging.warning("  수신 오디오   : 0 — mic/comm_bridge 확인")
        logging.info("  transcript    : 총 %d건", count["n"])
        if len(tr_times) >= 2:
            gaps = [tr_times[i] - tr_times[i - 1] for i in range(1, len(tr_times))]
            logging.info(
                "  transcript 간격: 평균 %.2fs (첫 발화→첫 transcript +%.2fs)",
                statistics.mean(gaps), tr_times[0],
            )
        elif tr_times:
            logging.info("  첫 발화→첫 transcript: +%.2fs", tr_times[0])
        logging.info("done. total transcripts=%d, final state=%s",
                     count["n"], stt.state.value)

    return 0 if count["n"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
