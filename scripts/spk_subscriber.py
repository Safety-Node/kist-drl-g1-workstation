"""
로컬 스피커 재생 노드.

/bridge/cmd/audio_out (AudioPCM) 구독 → paplay 로 PulseAudio default sink 재생.

로봇 없이 로컬에서 full-loop 테스트할 때 사용:
    mic_publisher.py      → /bridge/sensors/audio_pcm  (마이크 입력)
    speaker_player.py     → /bridge/cmd/audio_out     (TTS 출력 재생)
    verify_full_loop.py --local  (STT → LLM → TTS 파이프라인)

실행:
    source env.sh
    uv run python scripts/speaker_player.py
    uv run python scripts/speaker_player.py --device alsa_output.usb-Generic_AB13X_USB_Audio...  # 특정 싱크
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

# Local-only: drop NX-specific CycloneDDS config (eno2 interface)
os.environ.pop("CYCLONEDDS_URI", None)

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from g1_onboard_msgs.msg import AudioPCM

TOPIC = "/bridge/cmd/audio_out"


class SpeakerPlayerNode(Node):
    def __init__(self, pa_sink: str | None):
        super().__init__("speaker_player")

        self._pa_sink = pa_sink
        self._q: queue.Queue[bytes] = queue.Queue()
        self._chunks = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._sub = self.create_subscription(AudioPCM, TOPIC, self._on_audio, qos)

        self._player = threading.Thread(target=self._player_thread, daemon=True, name="speaker_player")
        self._player.start()

        self.get_logger().info(
            f"speaker_player: {TOPIC} → paplay"
            + (f" --device={pa_sink}" if pa_sink else " (default sink)")
        )

    def _on_audio(self, msg: AudioPCM) -> None:
        pcm = bytes(msg.data)
        if pcm:
            self._q.put(pcm)
            self._chunks += 1
            if self._chunks % 50 == 0:
                self.get_logger().info(f"received {self._chunks} audio chunks")

    def _player_thread(self) -> None:
        """Collect chunks until a 150ms gap, then play via paplay."""
        while True:
            # Wait for first chunk
            try:
                first = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            chunks = [first]
            import time
            deadline = time.monotonic() + 0.15
            while time.monotonic() < deadline:
                try:
                    chunks.append(self._q.get_nowait())
                except queue.Empty:
                    time.sleep(0.01)

            data = b"".join(chunks)
            cmd = ["pacat", "--playback", "--format=s16le", "--rate=16000", "--channels=1"]
            if self._pa_sink:
                cmd += [f"--device={self._pa_sink}"]

            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                _, err = proc.communicate(input=data, timeout=30)
                if proc.returncode != 0 and err:
                    self.get_logger().warning(f"paplay: {err.decode().strip()}")
            except Exception as exc:
                self.get_logger().warning(f"speaker playback error: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="PulseAudio sink name (기본: default sink). "
                         "pactl list short sinks 로 확인")
    args = ap.parse_args()

    rclpy.init()
    node = SpeakerPlayerNode(pa_sink=args.device)

    print(f"스피커 대기 중 — {TOPIC} 수신 시 재생. Ctrl-C 종료.")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
