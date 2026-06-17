"""
Live TTS verification (G1 provider 경유 버전): text → TTSProvider → G1 → NX speaker.

verify_tts_live.py 와 동일한 입출력이지만, **TASK-41 어댑터(_AudioOutBridgeNode)
+ monkey patch 를 제거**하고 정식 경로로 검증한다. TTSProvider 는 native 로
``self._unitree_g1.publish_audio_out(pcm)`` 를 호출해 G1 provider 의 publisher 가
``/bridge/cmd/audio_out`` 으로 내보낸다 — verify_full_loop.py 와 동일한 publish
경로. speaker_state 도 G1 provider 의 ``speaker_state`` TopicCache 로 폴링한다.

대비표 (어댑터 버전 vs 본 스크립트):
    /bridge/cmd/audio_out publish:
        verify_tts_live.py    → _AudioOutBridgeNode (스크립트 직접 publish)
                                 + tts._unitree_g1.publish_audio_out monkey patch
        verify_tts_live_g1.py → g1.publish_audio_out() native (monkey patch 없음)
    /bridge/audio/speaker_state 구독:
        verify_tts_live.py    → _AudioOutBridgeNode 가 subscriber 생성 + spin
        verify_tts_live_g1.py → g1.speaker_state TopicCache 폴링

데이터 흐름:
    text → TTSProvider.synthesize() → (Clova REST → 16k PCM)
         → tts._publish() → g1.publish_audio_out()
         → comm_bridge inbound_relay (RELIABLE) → /onboard/audio/playback
         → NX speaker_node → AudioClient.PlayStream → 🔊

전제:
    source env.sh                        # ROS2 + g1_onboard_msgs
    export NCP_CLOVA_CLIENT_ID=...        # Naver Clova Voice 자격증명 (.env 자동로드 가능)
    export NCP_CLOVA_CLIENT_SECRET=...
    (NX 측 comm_bridge + speaker_node 가 떠 있어야 실제 소리가 남)
실행:
    uv run python scripts/verify_tts_live_g1.py "냉장고에서 오이 가져올게요"
    uv run python scripts/verify_tts_live_g1.py --voice nara "안녕하세요"
"""

import argparse
import asyncio
import logging
import math
import sys
import time
from array import array
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import dotenv
dotenv.load_dotenv(dotenv_path=_ROOT / ".env")

from providers.tts_provider import TTSProvider, TTSConfig, TTSBackend
from providers.unitree_g1_provider import UnitreeG1Provider

# 파이프라인 LOCKED (AudioPCM.msg)
SAMPLE_RATE = 16000
CHANNELS = 1
BIT_DEPTH = 16
BYTES_PER_SAMPLE = 2

# G1 provider publish_audio_out 의 한도 (msg payload 65500 B = ~2.04s @ 16k mono int16)
_G1_MAX_PAYLOAD = 65500


def make_tone(freq: float, seconds: float, amp: float = 0.3) -> bytes:
    """16kHz mono int16 사인파 PCM (--tone 모드: Clova 없이 speaker 경로만 검증)."""
    n = int(SAMPLE_RATE * seconds)
    peak = int(max(0.0, min(1.0, amp)) * 32767)
    buf = array("h")
    for i in range(n):
        buf.append(int(peak * math.sin(2.0 * math.pi * freq * i / SAMPLE_RATE)))
    return buf.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="", help="합성할 텍스트 (--tone 시 불필요)")
    ap.add_argument("--tone", action="store_true",
                    help="Clova 없이 사인파 톤으로 speaker 경로만 검증")
    ap.add_argument("--freq", type=float, default=440.0, help="--tone 주파수(Hz)")
    ap.add_argument("--tone-seconds", type=float, default=1.5, help="--tone 길이(초)")
    ap.add_argument("--voice", default="nara", help="Clova voice id")
    ap.add_argument("--speed", type=int, default=0,
                    help="Clova 말하기 속도 [-5~+5] (음수=빠르게, 양수=느리게)")
    ap.add_argument("--listen-sec", type=float, default=6.0,
                    help="발행 후 speaker_state 전이를 듣는 시간(초)")
    ap.add_argument("--wait-bridge-sec", type=float, default=3.0,
                    help="g1.start() 후 comm_bridge heartbeat 가 잡힐 때까지 대기(초)")
    args = ap.parse_args()
    if not args.tone and not args.text:
        ap.error("text 인자가 필요합니다 (또는 --tone 사용)")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    # ── 와이어링 (run.py 순서: G1 FIRST → TTS 가 같은 싱글턴을 본다) ──
    UnitreeG1Provider.reset()  # type: ignore[attr-defined]
    TTSProvider.reset()        # type: ignore[attr-defined]

    g1 = UnitreeG1Provider()

    tts = None
    stage_ms = {}
    if not args.tone:
        tts = TTSProvider(TTSConfig(
            backend=TTSBackend.NAVER_CLOVA,
            voice=args.voice,
            speed=args.speed,
            sample_rate_hz=SAMPLE_RATE,
        ))
        # --- 단계별 타이밍 래퍼 (TTSProvider 로직은 그대로, 시간만 측정) -----
        _orig_clova = tts._http_post_clova
        _orig_decode = tts._decode_wav
        _orig_resample = tts._resample_to_wire
        _orig_publish = tts._publish

        async def _timed_clova(text):
            t = time.monotonic()
            r = await _orig_clova(text)
            stage_ms["clova"] = (time.monotonic() - t) * 1000
            return r

        def _timed_decode(wav):
            t = time.monotonic()
            r = _orig_decode(wav)
            stage_ms["decode"] = (time.monotonic() - t) * 1000
            return r

        def _timed_resample(pcm, src, ch):
            t = time.monotonic()
            r = _orig_resample(pcm, src, ch)
            stage_ms["resample"] = (time.monotonic() - t) * 1000
            return r

        # _publish 시점 (T3) — g1 으로 내려가기 직전
        publish_ts = {"t": None, "bytes": 0}

        def _timed_publish(pcm):
            if publish_ts["t"] is None:
                publish_ts["t"] = time.monotonic()
            publish_ts["bytes"] += len(pcm)
            return _orig_publish(pcm)

        tts._http_post_clova = _timed_clova       # type: ignore[assignment]
        tts._decode_wav = _timed_decode           # type: ignore[assignment]
        tts._resample_to_wire = _timed_resample   # type: ignore[assignment]
        tts._publish = _timed_publish             # type: ignore[assignment]
    else:
        publish_ts = {"t": None, "bytes": 0}

    # ── 기동 ──────────────────────────────────────────────────────────
    g1.start()
    if tts is not None:
        tts.start()   # 자격증명 없으면 WARNING 후 진행(synthesize 가 log+drop)

    # comm_bridge 가 살아나길 잠시 대기 (g1.publish_audio_out 이 alive 체크함)
    deadline = time.monotonic() + max(0.1, args.wait_bridge_sec)
    while time.monotonic() < deadline:
        if g1.comm_bridge_alive():
            break
        time.sleep(0.1)
    if not g1.comm_bridge_alive():
        logging.warning(
            "comm_bridge heartbeat 미수신 (%.1fs) — publish_audio_out 가 dropping 할 수 있음. "
            "그래도 시도는 진행.", args.wait_bridge_sec,
        )

    t_synth_start = time.monotonic()   # T0
    try:
        if args.tone:
            pcm = make_tone(args.freq, args.tone_seconds)
            logging.info("tone: %.0fHz %.1fs (Clova 없이 speaker 경로만 검증)",
                         args.freq, args.tone_seconds)
            # g1.publish_audio_out 은 65500 B 한도 — tone 은 직접 chunk 해서 보냄
            for i in range(0, len(pcm), _G1_MAX_PAYLOAD):
                slab = pcm[i:i + _G1_MAX_PAYLOAD]
                if publish_ts["t"] is None:
                    publish_ts["t"] = time.monotonic()
                publish_ts["bytes"] += len(slab)
                g1.publish_audio_out(slab)
        else:
            logging.info("synthesize: %r (voice=%s, speed=%d)",
                         args.text, args.voice, args.speed)
            asyncio.run(tts.synthesize(args.text))
    except KeyboardInterrupt:
        pass
    finally:
        # 재생 동안 g1.speaker_state TopicCache 가 갱신되는지 폴링
        logging.info("listening %.1fs for /bridge/audio/speaker_state ...", args.listen_sec)
        observed = 0
        first_play_ts = None
        last_seen = None
        end = time.monotonic() + max(1.0, args.listen_sec)
        while time.monotonic() < end:
            cache = g1.speaker_state
            seen_ts = cache.last_seen_ts
            val = cache.value
            if seen_ts > 0.0 and seen_ts != last_seen:
                last_seen = seen_ts
                observed += 1
                playing = bool(getattr(val, "playing", False)) if val is not None else False
                chunk_id = getattr(val, "current_chunk_id", "?") if val is not None else "?"
                qd = getattr(val, "queue_depth", "?") if val is not None else "?"
                logging.info(
                    "SPEAKER_STATE  playing=%s  chunk_id=%s  queue_depth=%s",
                    playing, chunk_id, qd,
                )
                if playing and first_play_ts is None and publish_ts["t"] is not None:
                    first_play_ts = time.monotonic()
                    logging.info(
                        "  [timing] end-to-end→speaker : %.0f ms (publish → playing=True)",
                        (first_play_ts - publish_ts["t"]) * 1000,
                    )
            time.sleep(0.05)
        if tts is not None:
            tts.stop()
        g1.stop()

    # --- 타이밍 요약 -------------------------------------------------------
    if publish_ts["t"] is not None:
        audio_sec = publish_ts["bytes"] / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        logging.info("---- timing ----")
        if not args.tone:
            clova = stage_ms.get("clova", 0.0)
            dec = stage_ms.get("decode", 0.0)
            res = stage_ms.get("resample", 0.0)
            synth_pub = (publish_ts["t"] - t_synth_start) * 1000   # T3 - T0
            rtf = (synth_pub / 1000.0) / audio_sec if audio_sec > 0 else float("nan")
            logging.info("  clova(A)           : %6.0f ms", clova)
            logging.info("  decode+resample(B) : %6.0f ms", dec + res)
            logging.info("  synth→publish(T3-T0): %6.0f ms   (audio %.2fs, RTF=%.2f)",
                         synth_pub, audio_sec, rtf)
        else:
            logging.info("  tone published     : audio %.2fs", audio_sec)
        if first_play_ts is not None:
            logging.info("  end-to-end→speaker : %6.0f ms   (publish → playing=True)",
                         (first_play_ts - publish_ts["t"]) * 1000)
        else:
            logging.info("  end-to-end→speaker :     -- (speaker_state playing=True 미수신)")
    else:
        logging.warning("타이밍 없음 — 오디오를 발행하지 않음 "
                        "(자격증명/E-STOP/네트워크/comm_bridge 확인)")

    if observed > 0:
        logging.info("done. speaker_state %d건 수신 — speaker_node 재생 확인됨.", observed)
        return 0
    logging.warning(
        "done. speaker_state 0건 — speaker_node/comm_bridge(outbound) 미동작이거나 "
        "오디오가 NX까지 도달 못 했을 수 있음. ros2 topic hz /onboard/audio/playback 확인.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
