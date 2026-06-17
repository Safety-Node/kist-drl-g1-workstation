"""
Live STT verification: /bridge/sensors/audio_pcm -> STTProvider -> text.

TASK-41(register_audio_callback)이 아직 미구현이므로, 이 스크립트가 직접
ROS2 노드(_AudioBridgeNode)로 /bridge/sensors/audio_pcm 을 구독해서
STTProvider._on_audio_chunk() 에 PCM 을 push 한다. 이 어댑터 노드는 본래
UnitreeG1Provider 가 할 일(/bridge/* 토픽의 단일 소유자)을 TASK-41 이 구현될
때까지 임시로 대신하는 scaffold 다. TASK-41 머지 후에는 이 노드를 버리고
STTProvider 가 register_audio_callback 으로 직접 붙으면 된다.

speaker_state / estop 은 NX 온보드(speaker_node / safety_monitor)가 아직
없으므로 구독하지 않는다. 따라서 에코 캔슬 / E-STOP 게이트는 발동하지 않고
마이크 입력이 항상 STT 로 흐른다(현 단계 text 변환 검증에는 무방).

전제:
    source env.sh                       # Jetson ROS2 환경 (rclpy, g1_onboard_msgs)
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json   # GOOGLE_CLOUD 백엔드
실행:
    uv run python scripts/verify_stt_live.py                  # 기본 google, ko-KR
    uv run python scripts/verify_stt_live.py --backend dummy  # 파이프라인 연결만 점검
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from g1_onboard_msgs.msg import AudioPCM

from providers.stt_provider import STTProvider, STTConfig, STTBackend, TranscriptEvent


class _AudioBridgeNode(Node):
    """audio_pcm 만 구독해 STTProvider 로 push 하는 임시 어댑터 (TASK-41 대행)."""

    def __init__(self, stt: STTProvider):
        super().__init__("verify_stt_live")
        self._stt = stt
        self._first = True
        # 타이밍 측정용 (전부 desktop monotonic)
        self.chunks = 0
        self.bytes_in = 0
        self.first_audio_ts = None
        self.last_audio_ts = None

        # mic_node publish QoS 와 일치해야 매칭됨 (audio_pcm 은 BestEffort).
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            AudioPCM, "/bridge/sensors/audio_pcm", self._on_audio, sensor_qos
        )
        self.get_logger().info("subscribed: /bridge/sensors/audio_pcm")

    def _on_audio(self, msg: AudioPCM) -> None:
        if self._first:
            self._first = False
            # 포맷 sanity check — LOCKED: 16000 / 1 / 16
            self.get_logger().info(
                f"first AudioPCM: rate={msg.sample_rate} ch={msg.channels} "
                f"depth={msg.bit_depth} bytes={len(msg.data)}"
            )
            if (msg.sample_rate, msg.channels, msg.bit_depth) != (16000, 1, 16):
                self.get_logger().warning(
                    "format != locked(16000/1/16) — STT 품질 저하 가능"
                )
        now = time.monotonic()
        self.chunks += 1
        self.bytes_in += len(msg.data)
        if self.first_audio_ts is None:
            self.first_audio_ts = now
        self.last_audio_ts = now
        # uint8[] data == int16 LE packed PCM. 그대로 STT 로 push.
        self._stt._on_audio_chunk(bytes(msg.data), now)


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

    # STTProvider 는 @singleton — 재실행 안전하게 reset 후 생성.
    # UnitreeG1Provider 는 STTProvider.__init__ 내부에서 참조하지만 .start() 를
    # 부르지 않으므로 inert(구독 0). speaker_state 가 항상 None → 에코캔슬 미발동.
    STTProvider.reset()  # type: ignore[attr-defined]
    stt = STTProvider(STTConfig(
        backend=backend,
        language_code=args.lang,
        sample_rate_hz=16000,
        interim_results=False,     # is_final 만 방출
    ))

    # ── 검증 지점: transcript 가 실제로 나오는지 + 타이밍 ──────────────
    count = {"n": 0}
    tr_times = []          # 첫 오디오 기준 transcript 도착 시각(s)
    node_ref = {"node": None}

    def on_transcript(ev: TranscriptEvent) -> None:
        count["n"] += 1
        now = time.monotonic()
        node = node_ref["node"]
        rel = (now - node.first_audio_ts) if (node and node.first_audio_ts) else 0.0
        tr_times.append(rel)
        logging.info(
            "TRANSCRIPT #%d  t=+%.2fs (final=%s, conf=%s): %r",
            count["n"], rel, ev.is_final, ev.confidence, ev.text,
        )

    stt.register_transcript_callback(on_transcript)

    # start(): register_audio_callback 은 NotImplementedError → WARNING 후 진행.
    # 실제 오디오는 아래 _AudioBridgeNode 가 _on_audio_chunk 로 직접 먹인다.
    stt.start()
    logging.info(
        "STT started (backend=%s, state=%s) — 말해보세요. Ctrl-C 종료.",
        args.backend, stt.state.value,
    )

    rclpy.init()
    node = _AudioBridgeNode(stt)
    node_ref["node"] = node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stt.stop()
        # ── 타이밍 요약 (전부 desktop monotonic) ──────────────────────
        logging.info("---- stt timing ----")
        if node.first_audio_ts and node.last_audio_ts:
            span = node.last_audio_ts - node.first_audio_ts
            rate = (node.chunks - 1) / span if span > 0 else 0.0
            logging.info("  수신 오디오   : %d청크 / %d bytes, 실측 %.1fHz (기대 ~50Hz)",
                         node.chunks, node.bytes_in, rate)
        else:
            logging.warning("  수신 오디오   : 0 — mic/comm_bridge 확인")
        logging.info("  transcript    : 총 %d건", count["n"])
        if len(tr_times) >= 2:
            gaps = [tr_times[i] - tr_times[i - 1] for i in range(1, len(tr_times))]
            logging.info("  transcript 간격: 평균 %.2fs (첫 발화→첫 transcript +%.2fs)",
                         statistics.mean(gaps), tr_times[0])
        elif tr_times:
            logging.info("  첫 발화→첫 transcript: +%.2fs", tr_times[0])
        node.destroy_node()
        rclpy.shutdown()
        logging.info("done. total transcripts=%d, final state=%s",
                     count["n"], stt.state.value)
    return 0 if count["n"] > 0 else 1   # 한 건도 못 받으면 비정상 종료


if __name__ == "__main__":
    raise SystemExit(main())
