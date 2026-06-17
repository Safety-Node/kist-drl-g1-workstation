"""
Live audio_pcm visualization: mic → comm_bridge → desktop 경로 검증용.

`/bridge/sensors/audio_pcm` 을 직접 구독 (G1 provider 우회 — desktop 도착
지점을 단독 검증). 마이크에 말하면 파형이 출렁이고, 안 말하면 평탄해야 함.
파형이 평탄하면 mic_node → comm_bridge → desktop 경로 어딘가가 깨진 것.

화면 구성 (matplotlib live plot, ~20Hz refresh):
    상단: 최근 2초 waveform (32k samples @ 16kHz)
    중단: 최근 30초 chunk-별 RMS amplitude (db로 변환)
    하단: 텍스트 통계 (수신 chunks, 실측 Hz, peak, RMS)

전제:
    source env.sh                       # ROS2 + g1_onboard_msgs
실행:
    uv run python scripts/visualize_audio_pcm.py
    uv run python scripts/visualize_audio_pcm.py --topic /bridge/sensors/audio_pcm
"""

import argparse
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

import matplotlib
matplotlib.use("TkAgg")  # 다른 backend 도 가능 — TkAgg 가 가장 흔히 됨
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from g1_onboard_msgs.msg import AudioPCM

# 파이프라인 LOCKED (AudioPCM.msg)
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

WAVEFORM_SEC = 2.0       # 상단 waveform 창 길이 (초)
AMPLITUDE_SEC = 30.0     # 중단 amplitude history 길이 (초)
EXPECTED_HZ = 50.0       # mic_node 발행 주기 (20ms chunks)
SILENCE_DB = -50.0       # 이 아래면 "조용함" 으로 간주


class AudioPCMTap(Node):
    """audio_pcm 만 구독해 thread-safe 버퍼로 옮기는 노드."""

    def __init__(self, topic: str):
        super().__init__("visualize_audio_pcm")
        # mic_node publish QoS 와 일치 (BestEffort). depth 는 시각화이므로 넉넉히.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.create_subscription(AudioPCM, topic, self._on_audio, sensor_qos)

        self._lock = threading.Lock()
        # waveform: 최근 N samples 의 rolling buffer (int16)
        self._wave_capacity = int(SAMPLE_RATE * WAVEFORM_SEC)
        self._wave = np.zeros(self._wave_capacity, dtype=np.int16)
        # amplitude history: (ts, rms_db) 튜플 deque
        self._amp = deque(maxlen=int(EXPECTED_HZ * AMPLITUDE_SEC))
        # 통계
        self.chunks = 0
        self.bytes_in = 0
        self.first_ts = None
        self.last_ts = None
        self.peak = 0
        self.first_format_logged = False

        self.get_logger().info(f"subscribed: {topic}")

    def _on_audio(self, msg: AudioPCM) -> None:
        now = time.monotonic()

        if not self.first_format_logged:
            self.first_format_logged = True
            self.get_logger().info(
                f"first AudioPCM: rate={msg.sample_rate} ch={msg.channels} "
                f"depth={msg.bit_depth} bytes={len(msg.data)}"
            )
            if (msg.sample_rate, msg.channels, msg.bit_depth) != (16000, 1, 16):
                self.get_logger().warning(
                    "format != locked(16000/1/16) — 시각화 스케일이 맞지 않을 수 있음"
                )

        # uint8[] → int16 LE PCM
        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        if samples.size == 0:
            return

        # RMS 계산 (전체 chunk 평균 진폭, dBFS)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        rms_db = 20.0 * np.log10(rms / 32768.0 + 1e-12)
        peak = int(np.max(np.abs(samples)))

        with self._lock:
            n = samples.size
            if n >= self._wave_capacity:
                self._wave = samples[-self._wave_capacity:].copy()
            else:
                # rolling: 앞쪽 버리고 뒤에 append
                self._wave = np.roll(self._wave, -n)
                self._wave[-n:] = samples
            self._amp.append((now, rms_db))
            self.chunks += 1
            self.bytes_in += len(msg.data)
            if self.first_ts is None:
                self.first_ts = now
            self.last_ts = now
            if peak > self.peak:
                self.peak = peak

    def snapshot(self):
        """Plot updater 가 부르는 thread-safe snapshot."""
        with self._lock:
            return {
                "wave": self._wave.copy(),
                "amp": list(self._amp),
                "chunks": self.chunks,
                "bytes": self.bytes_in,
                "first_ts": self.first_ts,
                "last_ts": self.last_ts,
                "peak": self.peak,
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/bridge/sensors/audio_pcm",
                    help="구독할 audio_pcm 토픽 (기본: /bridge/sensors/audio_pcm)")
    ap.add_argument("--refresh-hz", type=float, default=20.0,
                    help="plot 갱신 주기 (기본: 20Hz)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    rclpy.init()
    node = AudioPCMTap(args.topic)

    # rclpy.spin 은 별도 데몬 스레드에서. matplotlib 은 main 스레드 점유.
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), name="rclpy_spin", daemon=True,
    )
    spin_thread.start()

    # ── matplotlib figure 셋업 ─────────────────────────────────────────
    fig, (ax_wave, ax_amp) = plt.subplots(2, 1, figsize=(11, 6.5))
    fig.canvas.manager.set_window_title("audio_pcm live (mic → bridge → desktop)")

    # 상단: waveform
    wave_x = np.arange(node._wave_capacity) / SAMPLE_RATE - WAVEFORM_SEC
    (wave_line,) = ax_wave.plot(wave_x, np.zeros(node._wave_capacity), lw=0.6)
    ax_wave.set_ylim(-32768, 32768)
    ax_wave.set_xlim(-WAVEFORM_SEC, 0.0)
    ax_wave.set_xlabel("seconds (relative to now)")
    ax_wave.set_ylabel("int16 sample")
    ax_wave.set_title(f"waveform — last {WAVEFORM_SEC:.1f}s @ {SAMPLE_RATE}Hz")
    ax_wave.grid(True, alpha=0.3)
    ax_wave.axhline(0, color="gray", lw=0.5)

    # 중단: amplitude (dB) over time
    (amp_line,) = ax_amp.plot([], [], lw=1.0, color="tab:orange")
    ax_amp.set_xlim(-AMPLITUDE_SEC, 0.0)
    ax_amp.set_ylim(-80, 0)
    ax_amp.set_xlabel("seconds (relative to now)")
    ax_amp.set_ylabel("RMS amplitude (dBFS)")
    ax_amp.set_title(f"per-chunk RMS — last {AMPLITUDE_SEC:.0f}s")
    ax_amp.grid(True, alpha=0.3)
    ax_amp.axhline(SILENCE_DB, color="gray", ls="--", lw=0.5, label=f"silence ≈ {SILENCE_DB:.0f}dB")
    ax_amp.legend(loc="upper right", fontsize=8)

    # 통계 텍스트 (figure 하단)
    stats_text = fig.text(
        0.5, 0.01,
        "waiting for first chunk...",
        ha="center", va="bottom",
        family="monospace", fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    def update(_frame):
        snap = node.snapshot()
        wave = snap["wave"]
        amp = snap["amp"]

        wave_line.set_ydata(wave)

        now = time.monotonic()
        if amp:
            ts = np.array([t - now for (t, _) in amp])
            db = np.array([d for (_, d) in amp])
            amp_line.set_data(ts, db)

        # 통계 라인
        if snap["first_ts"] is not None and snap["last_ts"] is not None:
            span = snap["last_ts"] - snap["first_ts"]
            rate = (snap["chunks"] - 1) / span if span > 0 else 0.0
            last_db = amp[-1][1] if amp else -np.inf
            quiet_marker = "QUIET" if last_db < SILENCE_DB else "ACTIVE"
            stats_text.set_text(
                f"chunks={snap['chunks']:6d}   "
                f"bytes={snap['bytes']:8d}   "
                f"rate={rate:5.1f}Hz (expect ~{EXPECTED_HZ:.0f})   "
                f"peak={snap['peak']:6d}   "
                f"last_rms={last_db:6.1f}dB   "
                f"[{quiet_marker}]"
            )
        else:
            stats_text.set_text(
                "no audio yet — mic_node / comm_bridge 확인. "
                "토픽 확인: ros2 topic hz /bridge/sensors/audio_pcm"
            )

        return wave_line, amp_line, stats_text

    interval_ms = int(1000.0 / max(1.0, args.refresh_hz))
    _anim = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
