"""
Exercise SoundSensor — CONV-009 log-based verification.

Wires SoundSensor with lightweight stubs so the full filter chain
(confidence / empty / dedupe / stop) runs in-process without needing a
live STT backend or robot.

Run from repo root:
    uv run python scripts/exercise_sound_sensor.py

Expected PASS lines in order:
  P1a: started flag
  P1b: callback registered with FakeSTT
  P1c: normal event → on_audio reached
  P2:  drop[confidence]
  P3:  drop[empty]
  P4:  drop[dedupe] (duplicate within window)
  P5a: started flag cleared after stop
  P5b: callback unregistered
  P5c: no on_audio after stop
"""

import logging
import sys
import types
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# ---------------------------------------------------------------------------
# Minimal ROS 2 stub — allows importing UnitreeG1Provider in environments
# where rclpy is not installed (import-time only; no rclpy calls are made
# because UnitreeG1Provider.__init__ defers rclpy.init() to .start()).
# ---------------------------------------------------------------------------
def _stub_module(name: str, **attrs):
    """Create and register a lightweight stub module."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

if "rclpy" not in sys.modules:
    _rclpy = _stub_module("rclpy", ok=lambda: False, init=lambda **kw: None,
                          shutdown=lambda: None)
    _stub_module("rclpy.node", Node=object)
    _stub_module("rclpy.executors", MultiThreadedExecutor=object)
    _stub_module("rclpy.qos", HistoryPolicy=object, QoSProfile=object,
                 ReliabilityPolicy=object)
    _stub_module("sensor_msgs", )
    _stub_module("sensor_msgs.msg", Imu=object)

from providers.unitree_g1_provider import UnitreeG1Provider  # noqa: E402
from providers.task_srv_provider import TaskSrvProvider, TaskSrvConfig  # noqa: E402
from providers.stt_provider import STTProvider, STTConfig, TranscriptEvent  # noqa: E402
from inputs.plugins.sound_sensor import SoundSensor, SoundSensorConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeSTT:
    """Minimal STT stub: tracks registered callbacks and fires events."""

    def __init__(self):
        self._cbs = []

    def register_transcript_callback(self, cb):
        self._cbs.append(cb)
        logging.debug("FakeSTT: registered %s", getattr(cb, "__qualname__", cb))

    def unregister_transcript_callback(self, cb):
        try:
            self._cbs.remove(cb)
        except ValueError:
            pass

    def fire(self, event: TranscriptEvent):
        for cb in list(self._cbs):
            cb(event)


class _FakeTaskSrv:
    """Minimal TaskSrv stub: records on_audio calls."""

    def __init__(self):
        self.calls = []

    def on_audio(self, text: str, ts: Optional[float] = None):
        self.calls.append((text, ts))
        logging.info("[FakeTaskSrv] on_audio: %r (ts=%.3f)", text, ts or 0.0)


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
    """Run all SoundSensor filter-chain cases and log PASS/FAIL for each."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    # Reset all @singletons so this script is reentrant (e.g. REPL).
    UnitreeG1Provider.reset()  # type: ignore[attr-defined]
    STTProvider.reset()  # type: ignore[attr-defined]
    TaskSrvProvider.reset()  # type: ignore[attr-defined]

    # Construct minimal real singletons (none started — no rclpy.init or
    # STT backend; their __init__ only stores config).
    UnitreeG1Provider()
    TaskSrvProvider(TaskSrvConfig())
    STTProvider(STTConfig())

    # SoundSensor fetches the singletons above in its __init__.
    sensor = SoundSensor(SoundSensorConfig(min_confidence=0.5, dedupe_window_s=1.5))

    # Replace singleton deps with isolated stubs.
    fake_stt = _FakeSTT()
    fake_task = _FakeTaskSrv()
    sensor._stt = fake_stt
    sensor._task_srv = fake_task

    # ------------------------------------------------------------------
    # P1 — start() + normal event
    # ------------------------------------------------------------------
    sensor.start()
    _check("P1a: started flag", sensor.started)
    _check("P1b: callback registered with FakeSTT", len(fake_stt._cbs) == 1)

    evt = TranscriptEvent(text="냉장고에서 오이 가져와", ts=time.monotonic(), confidence=0.9)
    fake_stt.fire(evt)
    _check("P1c: normal event → on_audio reached", len(fake_task.calls) == 1)

    # ------------------------------------------------------------------
    # P2 — drop[confidence]
    # ------------------------------------------------------------------
    before = len(fake_task.calls)
    fake_stt.fire(TranscriptEvent(text="잡음", ts=time.monotonic(), confidence=0.2))
    _check("P2: drop[confidence] (not forwarded)", len(fake_task.calls) == before)

    # ------------------------------------------------------------------
    # P3 — drop[empty]
    # ------------------------------------------------------------------
    before = len(fake_task.calls)
    fake_stt.fire(TranscriptEvent(text="   ", ts=time.monotonic(), confidence=None))
    _check("P3: drop[empty] (not forwarded)", len(fake_task.calls) == before)

    # ------------------------------------------------------------------
    # P4 — drop[dedupe] — inject fresh text then same text within window
    # ------------------------------------------------------------------
    before = len(fake_task.calls)
    t0 = time.monotonic()
    fresh = TranscriptEvent(text="냉장고 문 열어줘", ts=t0, confidence=None)
    dup = TranscriptEvent(text="냉장고 문 열어줘", ts=t0 + 0.3, confidence=None)
    fake_stt.fire(fresh)   # passes (different text from previous _last_event)
    fake_stt.fire(dup)     # dropped (same text, 0.3s < 1.5s window)
    _check("P4: drop[dedupe] (dup not forwarded)", len(fake_task.calls) == before + 1)

    # ------------------------------------------------------------------
    # P5 — stop() → callback unregistered → no further on_audio
    # ------------------------------------------------------------------
    sensor.stop()
    _check("P5a: started flag cleared", not sensor.started)
    _check("P5b: callback unregistered from FakeSTT", len(fake_stt._cbs) == 0)

    before = len(fake_task.calls)
    fake_stt.fire(TranscriptEvent(text="post stop", ts=time.monotonic(), confidence=None))
    _check("P5c: no on_audio after stop", len(fake_task.calls) == before)

    logging.info("exercise_sound_sensor: all cases complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
