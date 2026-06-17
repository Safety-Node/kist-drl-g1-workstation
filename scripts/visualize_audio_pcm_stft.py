"""
Live STFT visualization: audio_pcm 주파수 분석용.

`/bridge/sensors/audio_pcm` 구독 → 실시간 FFT 스펙트럼 + spectrogram.
AGC/노이즈게이트 진단, 주파수별 노이즈 파악 등에 사용.

화면 구성 (matplotlib live plot, ~20Hz refresh):
    상단: 최근 2초 waveform (raw int16)
    중단: 실시간 FFT 스펙트럼 (현재 chunk, Hann window, dBFS)
    하단: spectrogram (최근 10초, time×frequency heatmap)

전제:
    source env.sh                       # ROS2 + g1_onboard_msgs
실행:
    uv run python scripts/visualize_audio_pcm_stft.py
    uv run python scripts/visualize_audio_pcm_stft.py --topic /bridge/sensors/audio_pcm
"""

import argparse
import os
import threading
import time
from collections import deque
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

os.environ.pop("CYCLONEDDS_URI", None)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from g1_onboard_msgs.msg import AudioPCM

# 파이프라인 LOCKED (AudioPCM.msg)
SAMPLE_RATE    = 16000
CHUNK_SAMPLES  = 320          # 20ms × 16kHz
N_FREQ_BINS    = CHUNK_SAMPLES // 2 + 1   # 161 (0 ~ 8000Hz, Δf=50Hz)

WAVEFORM_SEC    = 2.0
SPECTROGRAM_SEC = 10.0
EXPECTED_HZ     = 50.0
DB_FLOOR        = -80.0


def compute_spectrum(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    int16 samples → (freqs Hz, power_dBFS) with Hann window.

    정규화:
      - 2 / sum(window) : one-sided 스펙트럼 진폭 보정 (음수 주파수 몫)
      - DC/Nyquist bin  : 미러링 없으므로 ×2 하지 않음
      - / 32768         : int16 full-scale 기준 dBFS
    """
    N = len(samples)
    window = np.hanning(N)
    windowed = samples.astype(np.float32) * window
    X = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N, 1.0 / SAMPLE_RATE)

    win_sum = float(np.sum(window)) + 1e-12
    amplitude = 2.0 * np.abs(X) / win_sum
    amplitude[0]  /= 2.0   # DC
    amplitude[-1] /= 2.0   # Nyquist

    power_db = 20.0 * np.log10(amplitude / 32768.0 + 1e-12)
    return freqs, power_db


class AudioSTFTTap(Node):
    def __init__(self, topic: str):
        super().__init__("visualize_audio_pcm_stft")
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.create_subscription(AudioPCM, topic, self._on_audio, sensor_qos)

        self._lock = threading.Lock()

        # 상단 waveform rolling buffer
        self._wave_capacity = int(SAMPLE_RATE * WAVEFORM_SEC)
        self._wave = np.zeros(self._wave_capacity, dtype=np.int16)

        # 중단 스펙트럼 (최신 chunk)
        self._freqs: np.ndarray | None = None
        self._spectrum: np.ndarray | None = None

        # 하단 spectrogram: deque of (timestamp, power_db ndarray)
        self._spec_maxlen = int(EXPECTED_HZ * SPECTROGRAM_SEC)
        self._spec_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=self._spec_maxlen)

        # 통계
        self.chunks    = 0
        self.bytes_in  = 0
        self.first_ts: float | None = None
        self.last_ts:  float | None = None

        self.get_logger().info(f"subscribed: {topic}")

    def _on_audio(self, msg: AudioPCM) -> None:
        now = time.monotonic()
        samples = np.frombuffer(bytes(msg.data), dtype=np.int16)
        if samples.size == 0:
            return

        freqs, power_db = compute_spectrum(samples)

        with self._lock:
            n = samples.size
            if n >= self._wave_capacity:
                self._wave = samples[-self._wave_capacity:].copy()
            else:
                self._wave = np.roll(self._wave, -n)
                self._wave[-n:] = samples

            self._freqs    = freqs
            self._spectrum = power_db
            self._spec_frames.append((now, power_db.copy()))

            self.chunks   += 1
            self.bytes_in += len(msg.data)
            if self.first_ts is None:
                self.first_ts = now
            self.last_ts = now

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "wave":       self._wave.copy(),
                "freqs":      self._freqs.copy()    if self._freqs    is not None else None,
                "spectrum":   self._spectrum.copy() if self._spectrum is not None else None,
                "spec_frames": list(self._spec_frames),
                "chunks":     self.chunks,
                "bytes":      self.bytes_in,
                "first_ts":   self.first_ts,
                "last_ts":    self.last_ts,
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/bridge/sensors/audio_pcm")
    ap.add_argument("--refresh-hz", type=float, default=20.0)
    args = ap.parse_args()

    rclpy.init()
    node = AudioSTFTTap(args.topic)

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), name="rclpy_spin", daemon=True
    )
    spin_thread.start()

    fig, (ax_wave, ax_spec, ax_sgram) = plt.subplots(3, 1, figsize=(12, 9))
    fig.canvas.manager.set_window_title("audio_pcm STFT (mic → bridge → desktop)")

    # ── 상단: waveform ────────────────────────────────────────────────────
    wave_x = np.arange(node._wave_capacity) / SAMPLE_RATE - WAVEFORM_SEC
    (wave_line,) = ax_wave.plot(wave_x, np.zeros(node._wave_capacity), lw=0.6)
    ax_wave.set_ylim(-32768, 32768)
    ax_wave.set_xlim(-WAVEFORM_SEC, 0.0)
    ax_wave.set_xlabel("seconds (relative to now)")
    ax_wave.set_ylabel("int16 sample")
    ax_wave.set_title(f"waveform — last {WAVEFORM_SEC:.1f}s @ {SAMPLE_RATE}Hz")
    ax_wave.grid(True, alpha=0.3)
    ax_wave.axhline(0, color="gray", lw=0.5)

    # ── 중단: FFT 스펙트럼 ────────────────────────────────────────────────
    (spec_line,) = ax_spec.plot([], [], lw=1.0, color="tab:blue")
    ax_spec.set_xlim(0, SAMPLE_RATE / 2)
    ax_spec.set_ylim(DB_FLOOR, 0)
    ax_spec.set_xlabel("frequency (Hz)")
    ax_spec.set_ylabel("dBFS")
    ax_spec.set_title(f"FFT spectrum — current chunk (Hann window, Δf={SAMPLE_RATE/CHUNK_SAMPLES:.0f}Hz/bin)")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.axvspan(300, 3400, alpha=0.06, color="green", label="speech band (300–3400Hz)")
    ax_spec.legend(loc="upper right", fontsize=8)

    # ── 하단: spectrogram ────────────────────────────────────────────────
    spec_maxlen = node._spec_maxlen
    sgram_data  = np.full((N_FREQ_BINS, spec_maxlen), DB_FLOOR, dtype=np.float32)
    sgram_img   = ax_sgram.imshow(
        sgram_data,
        aspect="auto",
        origin="lower",
        extent=[-SPECTROGRAM_SEC, 0.0, 0, SAMPLE_RATE / 2],
        vmin=DB_FLOOR,
        vmax=0,
        cmap="inferno",
        interpolation="nearest",
    )
    fig.colorbar(sgram_img, ax=ax_sgram, label="dBFS")
    ax_sgram.set_xlabel("seconds (relative to now)")
    ax_sgram.set_ylabel("frequency (Hz)")
    ax_sgram.set_title(f"spectrogram — last {SPECTROGRAM_SEC:.0f}s")
    ax_sgram.axhline(300,  color="lime", lw=0.6, ls="--", alpha=0.6)
    ax_sgram.axhline(3400, color="lime", lw=0.6, ls="--", alpha=0.6)

    stats_text = fig.text(
        0.5, 0.002,
        "waiting for first chunk...",
        ha="center", va="bottom",
        family="monospace", fontsize=8,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))

    def update(_frame):
        snap = node.snapshot()

        # waveform
        wave_line.set_ydata(snap["wave"])

        # FFT 스펙트럼
        if snap["freqs"] is not None and snap["spectrum"] is not None:
            spec_line.set_data(snap["freqs"], snap["spectrum"])

        # spectrogram: deque → (N_FREQ_BINS, spec_maxlen) matrix
        frames = snap["spec_frames"]
        if frames:
            n_frames = len(frames)
            # shape (n_frames, N_FREQ_BINS) → transpose to (N_FREQ_BINS, n_frames)
            powers = np.stack([p for (_, p) in frames if p.size == N_FREQ_BINS], axis=0)
            if powers.ndim == 2 and powers.shape[0] > 0:
                matrix = np.full((N_FREQ_BINS, spec_maxlen), DB_FLOOR, dtype=np.float32)
                n = powers.shape[0]
                matrix[:, spec_maxlen - n:] = powers.T
                sgram_img.set_data(matrix)

        # 통계
        if snap["first_ts"] is not None and snap["last_ts"] is not None:
            span = snap["last_ts"] - snap["first_ts"]
            rate = (snap["chunks"] - 1) / span if span > 0 else 0.0
            stats_text.set_text(
                f"chunks={snap['chunks']:6d}   "
                f"rate={rate:5.1f}Hz (expect ~{EXPECTED_HZ:.0f})   "
                f"bytes={snap['bytes']:8d}"
            )

        return wave_line, spec_line, sgram_img, stats_text

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
