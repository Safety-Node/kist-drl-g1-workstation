"""
Exercise TaskSrvProvider standalone — no UnitreeG1 / STT / VLA backends.

Wires TaskSrvProvider + TaskSrvBg with stubbed dependencies and injects a
real demo trigger ("냉장고에서 오이 가져와") so the full state machine —
trigger match → activate → dispatch → cancel → IDLE — runs end-to-end and
shows up in the logs.

Run from repo root:
    uv run python scripts/exercise_task_srv.py

What you should see in the output (in order):
  - TaskSrvProvider loads scenarios from config.scenarios.ALL
  - TaskSrvBg tick loop entering (period=0.100s)
  - Trigger '냉장고에서 오이' matched scenario 'refrigerator_pickup'
  - [MOVE] connect(action='Walk to the refrigerator and face the door.')
  - ... (criterion never True; FakeG1 has no pose data)
  - Cancel keyword '취소' matched → aborting active scenario
  - [SPEAK] connect(action='취소했습니다.')
  - state back to IDLE
  - Clean shutdown on stop event

Why this exists: most Providers are still NotImplementedError, so running
``src/run.py`` aborts at the first .start() (or skips them via
--scaffold-loop but then nothing triggers TaskSrvProvider since SoundSensor
also doesn't start). This script bypasses that scaffolding gap and
exercises TaskSrvProvider directly via stubs.
"""

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any


# Make `providers.*` / `backgrounds.*` (under src/) AND `config.scenarios`
# (project root) importable from the scripts/ subdirectory.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))             # for `config.scenarios`
sys.path.insert(0, str(_ROOT / "src"))     # for `providers.*` / `backgrounds.*`


from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig  # noqa: E402
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs — replace the non-singleton Connectors. The Provider singletons
# (UnitreeG1Provider) are real instances; their default TopicCache fields
# (value=None, last_seen_ts=0.0) already behave as "never received" so
# TaskSrvProvider's tick() sees the same thing a FakeG1 would produce.
# ---------------------------------------------------------------------------


class FakeConnector:
    """ActionConnector-shaped fake: connect() logs the dispatched action."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def connect(self, output_interface: Any) -> None:
        logging.info("[%s] connect(action=%r)", self._name, output_interface.action)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    # Reset singleton in case this script was imported before (REPL etc.).
    TaskSrvProvider.reset()  # type: ignore[attr-defined]

    # Cancel keywords let us short-circuit the 25 s sub-task timeout below.
    # bind() now only takes the non-singleton Connectors (CONV-010);
    # UnitreeG1Provider is fetched as a @singleton inside TaskSrvProvider's
    # __init__ — its default-empty TopicCaches behave as "never received",
    # which keeps every success-criterion evaluating False forever.
    ts = TaskSrvProvider(TaskSrvConfig(cancel_keywords=["취소", "그만"]))
    ts.bind(
        move_connector=FakeConnector("MOVE"),
        speak_connector=FakeConnector("SPEAK"),
    )
    ts.start()  # loads config.scenarios.ALL — refrigerator_pickup at minimum

    bg = TaskSrvBg(TaskSrvBgConfig(tick_rate_hz=10.0))
    stop_event = threading.Event()
    bg.set_stop_event(stop_event)
    bg_thread = threading.Thread(target=bg.run, name="TaskSrvBg", daemon=True)
    bg_thread.start()

    def _on_sigint(_signum: int, _frame: Any) -> None:
        logging.info(">>> SIGINT — stopping")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    # ------------------------------------------------------------------
    # Timeline of injected events (Ctrl+C aborts at any point)
    # ------------------------------------------------------------------
    try:
        logging.info(">>> T+0.0s: provider IDLE (state=%s)", ts.state.value)
        time.sleep(1.0)

        logging.info(">>> T+1.0s: inject trigger '냉장고에서 오이 가져와'")
        ts.on_audio("냉장고에서 오이 가져와", ts=time.monotonic())

        # Sub-task 0 timeout_s=25.0 — wait a few seconds to watch tick noop,
        # then trigger the cancel path so we observe the full state cycle.
        time.sleep(3.0)

        logging.info(">>> T+4.0s: state=%s, active=%r",
                     ts.state.value, ts.active_scenario_name)

        logging.info(">>> T+4.0s: inject cancel keyword '취소'")
        ts.on_audio("취소", ts=time.monotonic())

        time.sleep(1.5)
        logging.info(">>> T+5.5s: state=%s (should be IDLE)", ts.state.value)

    except KeyboardInterrupt:
        logging.info(">>> KeyboardInterrupt — stopping")
    finally:
        logging.info(">>> Clean shutdown")
        stop_event.set()
        bg_thread.join(timeout=2.0)
        if bg_thread.is_alive():
            logging.warning("TaskSrvBg did not stop within 2s")
        ts.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
