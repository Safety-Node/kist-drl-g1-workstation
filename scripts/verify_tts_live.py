"""
Live TTS verification: text -> TTSProvider -> /bridge/cmd/audio_out -> NX speaker.

TASK-41(UnitreeG1Provider.publish_audio_out)이 아직 미구현이므로, 이 스크립트가
직접 ROS2 노드로 /bridge/cmd/audio_out 에 AudioPCM 을 publish 한다. TTSProvider 는
Clova 합성 → WAV 디코드 → 16kHz mono int16 리샘플까지 정상 동작하고, 마지막
publish 단계만 막혀 있으므로(NotImplementedError), 그 지점을 어댑터 publish 로
가로챈다(_AudioOutBridgeNode). TASK-41 머지 후에는 이 노드를 버리고 TTSProvider 가
g1.publish_audio_out 으로 직접 publish 하면 된다.

데이터 흐름:
    text → TTSProvider.synthesize() → (Clova REST → 16k PCM)
         → [어댑터] publish AudioPCM on /bridge/cmd/audio_out
         → comm_bridge inbound_relay (RELIABLE) → /onboard/audio/playback
         → NX speaker_node → AudioClient.PlayStream → 🔊

전제:
    source env.sh                        # ROS2 + g1_onboard_msgs
    export NCP_CLOVA_CLIENT_ID=...        # Naver Clova Voice 자격증명
    export NCP_CLOVA_CLIENT_SECRET=...
    (NX 측 comm_bridge + speaker_node 가 떠 있어야 실제 소리가 남)
실행:
    uv run python scripts/verify_tts_live.py "냉장고에서 오이 가져올게요"
    uv run python scripts/verify_tts_live.py --voice nara --chunk-ms 200 "안녕하세요"
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

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from g1_onboard_msgs.msg import AudioPCM, SpeakerState

from providers.tts_provider import TTSProvider, TTSConfig, TTSBackend

# 파이프라인 LOCKED (AudioPCM.msg)
SAMPLE_RATE = 16000
CHANNELS = 1
BIT_DEPTH = 16
BYTES_PER_SAMPLE = 2


def make_tone(freq: float, seconds: float, amp: float = 0.3) -> bytes:
    """16kHz mono int16 사인파 PCM (--tone 모드: Clova 없이 speaker 경로만 검증)."""
    n = int(SAMPLE_RATE * seconds)
    peak = int(max(0.0, min(1.0, amp)) * 32767)
    buf = array("h")
    for i in range(n):
        buf.append(int(peak * math.sin(2.0 * math.pi * freq * i / SAMPLE_RATE)))
    return buf.tobytes()


class _AudioOutBridgeNode(Node):
    """/bridge/cmd/audio_out 으로 AudioPCM 을 publish 하는 임시 어댑터 (TASK-41 대행)."""

    def __init__(self, chunk_ms: int = 0):
        super().__init__("verify_tts_live")
        self._chunk_ms = chunk_ms
        # comm_bridge inbound_relay 가 RELIABLE 강제 → publisher 도 RELIABLE.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._pub = self.create_publisher(AudioPCM, "/bridge/cmd/audio_out", qos)
        # NX speaker_node 가 재생하며 발행하는 상태 피드백을 구독(검증용).
        # /onboard/audio/speaker_state → comm_bridge outbound → /bridge/audio/speaker_state
        self._state_count = 0
        # 타이밍 측정용
        self.publish_ts = None        # 첫 publish 시각 (T3)
        self.published_bytes = 0      # 발행한 PCM 총 바이트 (RTF·audio 길이 계산)
        self.first_play_ts = None     # speaker_state playing=True 최초 수신 (T4)
        self.create_subscription(
            SpeakerState, "/bridge/audio/speaker_state", self._on_speaker_state, qos)
        self.get_logger().info(
            "publisher: /bridge/cmd/audio_out (RELIABLE)  |  "
            "subscriber: /bridge/audio/speaker_state")

    def _on_speaker_state(self, msg: SpeakerState) -> None:
        self._state_count += 1
        # end-to-end: publish(T3) → 실제 재생 시작(playing=True, T4)
        if msg.playing and self.first_play_ts is None and self.publish_ts is not None:
            self.first_play_ts = time.monotonic()
            self.get_logger().info(
                f"  [timing] end-to-end→speaker : "
                f"{(self.first_play_ts - self.publish_ts) * 1000:.0f} ms "
                f"(publish → playing=True)")
        self.get_logger().info(
            f"SPEAKER_STATE  playing={msg.playing}  "
            f"chunk_id={msg.current_chunk_id}  queue_depth={msg.queue_depth}")

    def wait_for_subscriber(self, timeout_s: float = 5.0) -> bool:
        """comm_bridge(또는 speaker) 가 구독을 붙일 때까지 대기 (RELIABLE 첫 메시지 유실 방지)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._pub.get_subscription_count() > 0

    def _make_msg(self, pcm: bytes) -> AudioPCM:
        msg = AudioPCM()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tts"
        msg.sample_rate = SAMPLE_RATE
        msg.channels = CHANNELS
        msg.bit_depth = BIT_DEPTH
        msg.data = list(pcm)   # uint8[]
        return msg

    def publish_pcm(self, pcm: bytes) -> None:
        """TTSProvider 의 publish 지점에서 호출됨. 통째로 보내거나 chunk_ms 로 쪼개 pace."""
        if not pcm:
            return
        if self.publish_ts is None:
            self.publish_ts = time.monotonic()   # T3
        self.published_bytes += len(pcm)
        if self._chunk_ms <= 0:
            # 한 발화 = AudioPCM 1개 = speaker 큐 청크 1개 (오버플로우 위험 없음)
            self._pub.publish(self._make_msg(pcm))
            self.get_logger().info(f"published 1 msg, {len(pcm)} bytes")
            return
        # chunk_ms 단위로 쪼개고 실시간보다 약간 빠르게 pace (큐 50 오버플로우 방지)
        step = SAMPLE_RATE * BYTES_PER_SAMPLE * self._chunk_ms // 1000  # bytes/chunk
        step -= step % BYTES_PER_SAMPLE
        n = 0
        for i in range(0, len(pcm), step):
            self._pub.publish(self._make_msg(pcm[i:i + step]))
            n += 1
            time.sleep(self._chunk_ms / 1000.0 * 0.9)
        self.get_logger().info(f"published {n} chunks, {len(pcm)} bytes total")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="", help="합성할 텍스트 (--tone 시 불필요)")
    ap.add_argument("--tone", action="store_true",
                    help="Clova 없이 사인파 톤으로 speaker 경로만 검증(데스크탑→speaker 단독)")
    ap.add_argument("--freq", type=float, default=440.0, help="--tone 주파수(Hz)")
    ap.add_argument("--tone-seconds", type=float, default=1.5, help="--tone 길이(초)")
    ap.add_argument("--voice", default="nara", help="Clova voice id")
    ap.add_argument("--speed", type=int, default=0,
                    help="Clova 말하기 속도 [-5~+5] (음수=빠르게, 양수=느리게)")
    ap.add_argument("--chunk-ms", type=int, default=0,
                    help="0=한 메시지로 전송 / >0=해당 ms 단위로 쪼개 pace")
    ap.add_argument("--listen-sec", type=float, default=6.0,
                    help="발행 후 speaker_state 전이를 듣는 시간(초)")
    args = ap.parse_args()
    if not args.tone and not args.text:
        ap.error("text 인자가 필요합니다 (또는 --tone 사용)")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    rclpy.init()
    node = _AudioOutBridgeNode(chunk_ms=args.chunk_ms)

    tts = None
    stage_ms = {}

    if not args.tone:
        # TTSProvider 는 @singleton — reset 후 생성. publish 지점을 어댑터로 가로챈다.
        TTSProvider.reset()  # type: ignore[attr-defined]
        tts = TTSProvider(TTSConfig(
            backend=TTSBackend.NAVER_CLOVA,
            voice=args.voice,
            speed=args.speed,
            sample_rate_hz=SAMPLE_RATE,
        ))
        # TASK-41 자리(publish_audio_out)를 어댑터 publish 로 채움.
        tts._unitree_g1.publish_audio_out = node.publish_pcm  # type: ignore[assignment]

        # --- 단계별 타이밍 래퍼 (TTSProvider 로직은 그대로, 시간만 측정) -----
        _orig_clova = tts._http_post_clova
        _orig_decode = tts._decode_wav
        _orig_resample = tts._resample_to_wire

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

        tts._http_post_clova = _timed_clova       # type: ignore[assignment]
        tts._decode_wav = _timed_decode           # type: ignore[assignment]
        tts._resample_to_wire = _timed_resample   # type: ignore[assignment]
        tts.start()   # 자격증명 없으면 WARNING 후 진행(synthesize 가 log+drop)

    if not node.wait_for_subscriber(5.0):
        node.get_logger().warn(
            "구독자 없음 — comm_bridge(/bridge/cmd/audio_out) 가 떠 있는지 확인. "
            "그래도 시도는 함(RELIABLE이라 첫 메시지 유실 가능).")

    t_synth_start = time.monotonic()   # T0
    try:
        if args.tone:
            pcm = make_tone(args.freq, args.tone_seconds)
            logging.info("tone: %.0fHz %.1fs (Clova 없이 speaker 경로만 검증)",
                         args.freq, args.tone_seconds)
            node.publish_pcm(pcm)
        else:
            logging.info("synthesize: %r (voice=%s, speed=%d)",
                         args.text, args.voice, args.speed)
            asyncio.run(tts.synthesize(args.text))
    except KeyboardInterrupt:
        pass
    finally:
        # 재생 동안 speaker_state 전이(playing false→true→false)를 수신하며 대기
        logging.info("listening %.1fs for /bridge/audio/speaker_state ...", args.listen_sec)
        end = time.monotonic() + max(1.0, args.listen_sec)
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        observed = node._state_count
        if tts is not None:
            tts.stop()

    # --- 타이밍 요약 -------------------------------------------------------
    if node.publish_ts is not None:
        audio_sec = node.published_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        logging.info("---- timing ----")
        if not args.tone:
            clova = stage_ms.get("clova", 0.0)
            dec = stage_ms.get("decode", 0.0)
            res = stage_ms.get("resample", 0.0)
            synth_pub = (node.publish_ts - t_synth_start) * 1000   # T3 - T0
            rtf = (synth_pub / 1000.0) / audio_sec if audio_sec > 0 else float("nan")
            logging.info("  clova(A)           : %6.0f ms", clova)
            logging.info("  decode+resample(B) : %6.0f ms", dec + res)
            logging.info("  synth→publish(T3-T0): %6.0f ms   (audio %.2fs, RTF=%.2f)",
                         synth_pub, audio_sec, rtf)
        else:
            logging.info("  tone published     : audio %.2fs", audio_sec)
        if node.first_play_ts is not None:
            logging.info("  end-to-end→speaker : %6.0f ms   (publish → playing=True)",
                         (node.first_play_ts - node.publish_ts) * 1000)
        else:
            logging.info("  end-to-end→speaker :     -- (speaker_state playing=True 미수신)")
    else:
        logging.warning("타이밍 없음 — 오디오를 발행하지 않음 "
                        "(자격증명/E-STOP/네트워크 확인)")

    node.destroy_node()
    rclpy.shutdown()
    if observed > 0:
        logging.info("done. speaker_state %d건 수신 — speaker_node 재생 확인됨.", observed)
        return 0
    logging.warning(
        "done. speaker_state 0건 — speaker_node/comm_bridge(outbound) 미동작이거나 "
        "오디오가 NX까지 도달 못 했을 수 있음. ros2 topic hz /onboard/audio/playback 확인.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
