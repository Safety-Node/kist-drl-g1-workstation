"""
로컬 마이크 → /bridge/sensors/audio_pcm 발행.

로봇 없이 workstation 마이크로 STT를 테스트할 때 사용.
verify_stt_live.py --local 와 함께 실행하면 로봇 mic_node 없이 전체 파이프라인 검증 가능.

장치가 16kHz를 지원하지 않으면 (e.g. 44100Hz 전용 USB 인터페이스) 자동으로 캡처 후
scipy로 16kHz 리샘플링한다.

실행:
    source env.sh
    uv run python scripts/mic_publisher.py              # 기본 장치
    uv run python scripts/mic_publisher.py --list       # 장치 목록 확인
    uv run python scripts/mic_publisher.py --device 11  # 장치 번호 지정 (e.g. NEVA UNO)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# mic_publisher is local-only; drop the NX-specific CycloneDDS config (eno2 interface)
# so rclpy.init() uses auto-detected loopback/WiFi interface instead.
os.environ.pop("CYCLONEDDS_URI", None)

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from g1_onboard_msgs.msg import AudioPCM

TARGET_RATE  = 16000
CHUNK_MS     = 20


def _resample_gcd(src_rate: int, dst_rate: int) -> tuple[int, int]:
    """Return (up, down) coprime integers for resample_poly."""
    g = math.gcd(src_rate, dst_rate)
    return dst_rate // g, src_rate // g


class MicPublisher(Node):
    def __init__(self, device, topic: str):
        super().__init__("mic_publisher")

        # Detect native sample rate of the device
        dev_info   = sd.query_devices(device, "input") if device is not None else sd.query_devices(sd.default.device[0], "input")
        native_sr  = int(dev_info["default_samplerate"])
        self._native_sr = native_sr
        self._resample  = native_sr != TARGET_RATE

        if self._resample:
            self._up, self._down = _resample_gcd(native_sr, TARGET_RATE)
            self.get_logger().info(
                f"Device native rate={native_sr}Hz; will resample ×{self._up}/{self._down} → {TARGET_RATE}Hz"
            )

        # blocksize in native samples so output is ~CHUNK_MS ms at TARGET_RATE
        self._blocksize = native_sr * CHUNK_MS // 1000

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self._pub    = self.create_publisher(AudioPCM, topic, qos)
        self._topic  = topic
        self._chunks = 0
        self._t_start = time.monotonic()

        self._stream = sd.InputStream(
            device=device,
            samplerate=native_sr,
            channels=1,
            dtype="int16",
            blocksize=self._blocksize,
            callback=self._on_audio,
        )
        self.get_logger().info(
            f"mic_publisher: device={device!r}  {native_sr}Hz mono int16 "
            f"→ {topic}  (chunk={CHUNK_MS}ms)"
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            self.get_logger().warning(f"sounddevice status: {status}")

        samples = indata[:, 0]  # mono

        if self._resample:
            resampled = resample_poly(samples.astype(np.float32), self._up, self._down)
            samples   = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)

        msg = AudioPCM()
        msg.sample_rate = TARGET_RATE
        msg.channels    = 1
        msg.bit_depth   = 16
        msg.data        = samples.tobytes()
        self._pub.publish(msg)

        self._chunks += 1
        if self._chunks % 250 == 0:   # ~5초마다
            elapsed = time.monotonic() - self._t_start
            hz = self._chunks / elapsed
            self.get_logger().info(f"published {self._chunks} chunks  rate={hz:.1f}Hz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list",   action="store_true", help="사용 가능한 장치 목록 출력 후 종료")
    ap.add_argument("--device", default=None,        help="sounddevice 장치 번호 또는 이름 (기본: 시스템 default)")
    ap.add_argument("--topic",  default="/bridge/sensors/audio_pcm")
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices())
        return 0

    device = int(args.device) if (args.device and args.device.isdigit()) else args.device

    rclpy.init()
    node = MicPublisher(device=device, topic=args.topic)
    node.start()

    print("마이크 입력 중 — Ctrl-C 로 종료")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
