"""
Exercise STTProvider (DUMMY backend) — CONV-009 log-based verification.

Tests the filter chain and state machine without a live mic, ROS 2, or
Google Cloud credentials.  UnitreeG1Provider callbacks (register_audio_callback /
register_estop_callback) are still NotImplementedError (TASK-41 pending);
this script calls _on_audio_chunk() and _on_estop() directly.

Run from repo root:
    uv run python scripts/exercise_stt.py

Expected PASS lines:
  S5a: initial state IDLE
  S5b: state STREAMING after start
  S1:  canned PCM (UTF-8 text bytes) → callback fires
  S4:  p50 feed→callback latency < 500 ms
  S2a: muted while speaker playing
  S2b: still muted in tail-off window
  S2c: resumed after tail-off expires
  S3a: blocked by E-STOP
  S3b: resumed after E-STOP cleared
  S5c: state IDLE after stop
"""

import logging
import statistics
import sys
import time
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# ---------------------------------------------------------------------------
# Minimal ROS 2 stub — UnitreeG1Provider imports rclpy at module level but
# defers all rclpy calls to .start(), which we never call here.
# ---------------------------------------------------------------------------
def _stub_module(name: str, **attrs):
    """Create and register a lightweight stub module."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

if "rclpy" not in sys.modules:
    _stub_module("rclpy", ok=lambda: False, init=lambda **kw: None,
                 shutdown=lambda: None)
    _stub_module("rclpy.node", Node=object)
    _stub_module("rclpy.executors", MultiThreadedExecutor=object)
    _stub_module("rclpy.qos", HistoryPolicy=object, QoSProfile=object,
                 ReliabilityPolicy=object)
    _stub_module("sensor_msgs")
    _stub_module("sensor_msgs.msg", Imu=object)

from providers.unitree_g1_provider import UnitreeG1Provider, TopicCache  # noqa: E402
from providers.stt_provider import (  # noqa: E402
    STTProvider,
    STTConfig,
    STTBackend,
    TranscriptEvent,
)


# ---------------------------------------------------------------------------
# Fake speaker state for echo-cancel testing
# ---------------------------------------------------------------------------

class _FakeSpeakerState:
    """Minimal speaker state value with a ``playing`` bool."""

    def __init__(self, playing: bool = False):
        self.playing = playing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(label: str, ok: bool) -> None:
    if ok:
        logging.info("PASS: %s", label)
    else:
        logging.error("FAIL: %s", label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all STTProvider (DUMMY backend) verification cases."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    # Reset singletons so this script is reentrant.
    UnitreeG1Provider.reset()  # type: ignore[attr-defined]
    STTProvider.reset()  # type: ignore[attr-defined]

    g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig(
        backend=STTBackend.DUMMY,
        echo_cancel_tail_ms=200,
        echo_cancel_lead_ms=0,
    ))

    # Collected (event, callback_ts) pairs from the subscriber.
    received = []

    def _on_transcript(event: TranscriptEvent) -> None:
        received.append((event, time.monotonic()))

    stt.register_transcript_callback(_on_transcript)

    # ------------------------------------------------------------------
    # S5a — initial state IDLE
    # ------------------------------------------------------------------
    _check("S5a: initial state IDLE", stt.state.value == "idle")

    # ------------------------------------------------------------------
    # Start (DUMMY — no credentials needed)
    # ------------------------------------------------------------------
    stt.start()
    _check("S5b: state STREAMING after start", stt.state.value == "streaming")

    # ------------------------------------------------------------------
    # S1 — canned PCM (UTF-8 text bytes) → callback fires
    # ------------------------------------------------------------------
    before = len(received)
    feed_ts = time.monotonic()
    stt._on_audio_chunk("냉장고에서 오이 가져와".encode("utf-8"), feed_ts)
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and len(received) == before:
        time.sleep(0.005)
    _check("S1: canned PCM → callback fires", len(received) > before)
    if len(received) > before:
        logging.info("S1: transcript text=%r", received[-1][0].text)

    # ------------------------------------------------------------------
    # S4 — p50 feed→callback latency < 500 ms
    # ------------------------------------------------------------------
    latencies = []
    for i in range(12):
        r_before = len(received)
        t_feed = time.monotonic()
        stt._on_audio_chunk(f"latency sample {i}".encode("utf-8"), t_feed)
        end = time.monotonic() + 0.5
        while time.monotonic() < end and len(received) == r_before:
            time.sleep(0.002)
        if len(received) > r_before:
            latencies.append((received[-1][1] - t_feed) * 1000)
    if latencies:
        p50 = statistics.median(latencies)
        logging.info("S4: p50=%.1f ms over %d samples (min=%.1f, max=%.1f)",
                     p50, len(latencies), min(latencies), max(latencies))
        _check(f"S4: p50 latency {p50:.1f} ms < 500 ms", p50 < 500.0)
    else:
        logging.error("FAIL: S4: no latency samples collected")

    # ------------------------------------------------------------------
    # S2 — echo-cancel muting + tail-off
    # ------------------------------------------------------------------
    # Set speaker to playing=True
    g1._speaker_state = TopicCache(
        value=_FakeSpeakerState(playing=True),
        last_seen_ts=time.monotonic(),
    )
    t_mute_start = time.monotonic()
    before = len(received)
    stt._on_audio_chunk("muted chunk".encode("utf-8"), t_mute_start)
    time.sleep(0.1)
    _check("S2a: muted while speaker playing", len(received) == before)

    # Clear speaker state (playing=False) — tail-off window still active
    g1._speaker_state = TopicCache(
        value=_FakeSpeakerState(playing=False),
        last_seen_ts=time.monotonic(),
    )
    # 50 ms after mute start → still within 200 ms tail window
    ts_in_tail = t_mute_start + 0.05
    before = len(received)
    stt._on_audio_chunk("in tail window".encode("utf-8"), ts_in_tail)
    time.sleep(0.1)
    _check("S2b: still muted in tail-off window", len(received) == before)

    # 300 ms after mute start → beyond 200 ms tail window
    ts_after_tail = t_mute_start + 0.30
    before = len(received)
    stt._on_audio_chunk("after tail expires".encode("utf-8"), ts_after_tail)
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline and len(received) == before:
        time.sleep(0.005)
    _check("S2c: resumed after tail-off expires", len(received) > before)

    # ------------------------------------------------------------------
    # S3 — E-STOP block / clear
    # ------------------------------------------------------------------
    stt._on_estop(True, time.monotonic())
    before = len(received)
    stt._on_audio_chunk("estop blocked".encode("utf-8"), time.monotonic())
    time.sleep(0.1)
    _check("S3a: blocked by E-STOP", len(received) == before)

    stt._on_estop(False, time.monotonic())
    before = len(received)
    stt._on_audio_chunk("after estop clear".encode("utf-8"), time.monotonic())
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline and len(received) == before:
        time.sleep(0.005)
    _check("S3b: resumed after E-STOP cleared", len(received) > before)

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------
    stt.stop()
    _check("S5c: state IDLE after stop", stt.state.value == "idle")

    logging.info("exercise_stt: all cases complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
