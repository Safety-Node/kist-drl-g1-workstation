"""
오디오 필터 테스트 노드.

/bridge/sensors/audio_pcm 구독 → HPF(120Hz) + LPF(3800Hz) IIR 필터 적용
→ /bridge/sensors/audio_pcm_filtered 발행.

필터링 결과를 visualize_audio_pcm_stft.py로 확인:
    uv run python scripts/visualize_audio_pcm_stft.py --topic /bridge/sensors/audio_pcm_filtered

실행:
    source env.sh
    uv run python scripts/test_audio_filter.py
    uv run python scripts/test_audio_filter.py --highpass 150 --lowpass 3600
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import os
os.environ.pop("CYCLONEDDS_URI", None)

from providers.stt_provider import StreamingSpeechFilter

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from g1_onboard_msgs.msg import AudioPCM

SAMPLE_RATE = 16000


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20.0 * np.log10(rms / 32768.0 + 1e-12)


class AudioFilterNode(Node):
    def __init__(self, highpass_hz: float, lowpass_hz: float):
        super().__init__("audio_filter_node")

        self._filter = StreamingSpeechFilter(SAMPLE_RATE, highpass_hz, lowpass_hz)
        self._hp = highpass_hz
        self._lp = lowpass_hz

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        self._sub = self.create_subscription(
            AudioPCM, "/bridge/sensors/audio_pcm", self._on_audio, qos_be
        )
        self._pub = self.create_publisher(
            AudioPCM, "/bridge/sensors/audio_pcm_filtered", qos_be
        )

        self._chunks  = 0
        self._t_start = time.monotonic()
        self._t_last_log = self._t_start

        self.get_logger().info(
            f"audio_filter: HPF={highpass_hz}Hz  LPF={lowpass_hz}Hz  "
            f"raw → /bridge/sensors/audio_pcm_filtered"
        )

    def _on_audio(self, msg: AudioPCM) -> None:
        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        if samples.size == 0:
            return

        filtered = self._filter.process(samples)

        out = AudioPCM()
        out.sample_rate = msg.sample_rate
        out.channels    = msg.channels
        out.bit_depth   = msg.bit_depth
        out.data        = filtered.tobytes()
        self._pub.publish(out)

        self._chunks += 1

        now = time.monotonic()
        if now - self._t_last_log >= 5.0:
            elapsed = now - self._t_start
            rate    = self._chunks / elapsed if elapsed > 0 else 0.0
            rms_in  = _rms_dbfs(samples)
            rms_out = _rms_dbfs(filtered)
            self.get_logger().info(
                f"chunks={self._chunks:6d}  rate={rate:5.1f}Hz  "
                f"RMS in={rms_in:+6.1f}dBFS  out={rms_out:+6.1f}dBFS  "
                f"Δ={rms_out - rms_in:+5.1f}dB"
            )
            self._t_last_log = now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--highpass", type=float, default=120.0, help="HPF cutoff Hz (default 120)")
    ap.add_argument("--lowpass",  type=float, default=3800.0, help="LPF cutoff Hz (default 3800)")
    args = ap.parse_args()

    rclpy.init()
    node = AudioFilterNode(args.highpass, args.lowpass)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
