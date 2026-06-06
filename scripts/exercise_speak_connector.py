"""
Exercise SpeakConnector [TASK-45] — log-based verification.

Verifies the fire-and-forget contract without a live TTS backend:
  - connect() forwards SpeakInput.action to TTSProvider.synthesize.
  - connect() swallows synthesis exceptions (NEVER re-raises) so a failed
    announcement cannot crash TaskSrvProvider's dispatch path.
  - asyncio.CancelledError IS re-raised (cooperative cancellation).

The connector fetches TTSProvider() as a @singleton in __init__; we swap in
a fake afterwards so no real Clova call happens.

Run from repo root:
    uv run python scripts/exercise_speak_connector.py

Expected PASS lines:
  P1: connect → TTS.synthesize called with the action text
  P2: TTS.synthesize raises → connect does NOT re-raise (swallowed + logged)
  P3: TTS.synthesize raises CancelledError → connect DOES re-raise
"""

import asyncio
import logging
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def _stub_module(name: str, **attrs):
    """Create and register a lightweight stub module."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        _stub_module("rclpy", ok=lambda: False, init=lambda **kw: None, shutdown=lambda: None)
        _stub_module("rclpy.node", Node=object)
        _stub_module("rclpy.executors", MultiThreadedExecutor=object)
        _stub_module("rclpy.qos", HistoryPolicy=object, QoSProfile=object, ReliabilityPolicy=object)
        _stub_module("sensor_msgs")
        _stub_module("sensor_msgs.msg", Imu=object)

from actions.base import ActionConfig  # noqa: E402
from actions.speak.interface import SpeakInput  # noqa: E402
from actions.speak.connector.speak_connector import SpeakConnector  # noqa: E402
from providers.tts_provider import TTSProvider  # noqa: E402


class _FakeTTS:
    """Records synthesize() calls; optionally raises to test the swallow path."""

    def __init__(self):
        self.calls = []
        self.raise_exc = None  # set to an exception instance to raise

    async def synthesize(self, text: str) -> None:
        self.calls.append(text)
        if self.raise_exc is not None:
            raise self.raise_exc


def _check(label: str, ok: bool) -> None:
    if ok:
        logging.info("PASS: %s", label)
    else:
        logging.error("FAIL: %s", label)


async def main() -> int:
    """Run SpeakConnector fire-and-forget contract verification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    TTSProvider.reset()  # type: ignore[attr-defined]
    connector = SpeakConnector(ActionConfig())

    fake = _FakeTTS()
    connector._tts = fake  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # P1 — forwards text to synthesize
    # ------------------------------------------------------------------
    await connector.connect(SpeakInput(action="냉장고 앞으로 갑니다."))
    _check("P1: connect → synthesize called with action text",
           fake.calls == ["냉장고 앞으로 갑니다."])

    # ------------------------------------------------------------------
    # P2 — synthesize raises → connect swallows (no re-raise)
    # ------------------------------------------------------------------
    fake.raise_exc = RuntimeError("simulated Clova failure")
    swallowed = True
    try:
        await connector.connect(SpeakInput(action="실패 케이스"))
    except Exception:
        swallowed = False
    _check("P2: synthesize error swallowed (connect did not re-raise)", swallowed)

    # ------------------------------------------------------------------
    # P3 — CancelledError IS re-raised (cooperative cancellation)
    # ------------------------------------------------------------------
    fake.raise_exc = asyncio.CancelledError()
    reraised = False
    try:
        await connector.connect(SpeakInput(action="취소 케이스"))
    except asyncio.CancelledError:
        reraised = True
    _check("P3: CancelledError re-raised (not swallowed)", reraised)

    logging.info("exercise_speak_connector: all cases complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
