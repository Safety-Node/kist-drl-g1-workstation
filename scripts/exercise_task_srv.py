"""
Exercise TaskSrvProvider standalone — no STT / VLA / DDS backends.

Drives the LINEAR, consume-once ``move_test`` scenario end-to-end with
FakeConnectors and a fake NavigationProvider. Each leg = two sub-tasks:
① wait for a voice command (voice_keyword) → ② Move + nav ARRIVED
(nav_state, mode=ARRIVED). No loop, so a consumed command keyword does NOT
re-fire — re-issuing "냉장고로 가" after the fridge leg succeeds is ignored
because the active sub-task only listens for 식탁.

Run from repo root:
    uv run python scripts/exercise_task_srv.py

Expected log order:
  - loads scenarios from config/scenarios/*.json5 ; tick loop entering
  - trigger '이동 테스트 시작' → on_scenario_start [SPEAK] 이동 테스트를 시작합니다...
  - command '냉장고로 가' → go_fridge: [SPEAK] 냉장고로 갑니다. + [MOVE] Walk to the refrigerator...
  - inject nav ARRIVED → [SPEAK] 냉장고에 도착했습니다.   (now current = await_table_cmd)
  - RE-ISSUE '냉장고로 가' → IGNORED (await_table_cmd only listens for 식탁); state stays active
  - command '식탁으로 가' → go_table: [SPEAK] 식탁으로 갑니다. + [MOVE] Walk to the table.
  - inject nav ARRIVED → [SPEAK] 식탁에 도착했습니다. (current = await_home_cmd)
  - command '집으로 가' → go_home: [SPEAK] 처음 위치로 돌아갑니다. + [MOVE] Walk to the home position.
  - inject nav ARRIVED → [SPEAK] 처음 위치에 도착했습니다. → on_scenario_end → state IDLE
  - Clean shutdown on stop event
"""

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any


# Make `providers.*` / `backgrounds.*` (src/) AND `config.scenarios` (root)
# importable from scripts/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig  # noqa: E402
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider  # noqa: E402
from providers.unitree_g1_provider import TopicCache, UnitreeG1Provider  # noqa: E402


class FakeConnector:
    """ActionConnector-shaped fake: connect() logs the dispatched action."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def connect(self, output_interface: Any) -> None:
        logging.info("[%s] connect(action=%r)", self._name, output_interface.action)


def _inject_uwb_pose(x: float, y: float, yaw: float) -> None:
    """Replace UnitreeG1Provider's UWB cache so a UwbPose criterion can pass."""
    g1 = UnitreeG1Provider()
    g1._uwb_pose = TopicCache(value={"x": x, "y": y, "yaw": yaw}, last_seen_ts=time.monotonic())


class _FakeNavProvider:
    """Stand-in for NavigationProvider — only ``get_state()`` is called by the
    NavState criterion. ``set(mode)`` drives transitions from the test script.
    """

    def __init__(self) -> None:
        from providers.navigation_provider import NavigationState
        self._NavigationState = NavigationState
        self._state = NavigationState(t_monotonic=time.monotonic(), mode="IDLE")

    def get_state(self) -> Any:
        return self._state

    def set(self, mode: str) -> None:
        self._state = self._NavigationState(t_monotonic=time.monotonic(), mode=mode)


def _inject_nav_state(mode: str) -> None:
    """Plug a fake NavigationProvider into TaskSrvProvider and set its mode.

    Satisfies a ``nav_state`` criterion without running the real navigation
    stack. First call installs the fake on the TaskSrvProvider singleton;
    subsequent calls just mutate the mode.
    """
    ts = TaskSrvProvider()
    nav = getattr(ts, "_nav_provider", None)
    if not isinstance(nav, _FakeNavProvider):
        nav = _FakeNavProvider()
        ts._nav_provider = nav
    nav.set(mode)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    TaskSrvProvider.reset()  # type: ignore[attr-defined]
    UnitreeG1Provider.reset()  # type: ignore[attr-defined]

    # cancel_keywords intentionally excludes "그만" so it doesn't shadow the
    # move_test exit keyword in this demo.
    ts = TaskSrvProvider(TaskSrvConfig(cancel_keywords=["취소"]))
    ts.bind(move_connector=FakeConnector("MOVE"), speak_connector=FakeConnector("SPEAK"))
    ts.start()  # loads config/scenarios/*.yaml (move_test + refrigerator_pickup)

    bg = TaskSrvBg(TaskSrvBgConfig())   # loop tuning lives on TaskSrvConfig now
    stop_event = threading.Event()
    bg.set_stop_event(stop_event)
    bg_thread = threading.Thread(target=bg.run, name="TaskSrvBg", daemon=True)
    bg_thread.start()

    def _on_sigint(_signum: int, _frame: Any) -> None:
        logging.info(">>> SIGINT — stopping")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        logging.info(">>> T+0.0s: IDLE (state=%s)", ts.state.value)
        time.sleep(1.0)

        logging.info(">>> T+1.0s: trigger '이동 테스트 시작'")
        ts.on_audio("이동 테스트 시작", ts=time.monotonic())
        time.sleep(1.0)

        # ── fridge leg ─────────────────────────────────────────────────
        logging.info(">>> T+2.0s: command '냉장고로 가'")
        ts.on_audio("냉장고로 가", ts=time.monotonic())
        time.sleep(2.0)   # move action has delay=1.0; give it time to fire too

        logging.info(">>> T+4.0s: inject nav.mode=ARRIVED (fridge)")
        _inject_nav_state("ARRIVED")
        time.sleep(1.0)

        # Consume-once proof: fridge leg is done; current sub-task is now
        # await_table_cmd, which only listens for 식탁. Re-issuing 냉장고 must
        # be ignored (no [SPEAK]/[MOVE], state stays active on move_test).
        logging.info(">>> T+5.0s: RE-ISSUE '냉장고로 가' (should be IGNORED) — sub-task='%s'",
                     ts.active_sub_task.name if ts.active_sub_task else None)
        ts.on_audio("냉장고로 가", ts=time.monotonic())
        time.sleep(1.0)
        logging.info(">>> T+6.0s: state=%s sub-task=%s (still awaiting 식탁)",
                     ts.state.value, ts.active_sub_task.name if ts.active_sub_task else None)

        # ── table leg ──────────────────────────────────────────────────
        # The fake nav still says mode=ARRIVED from the fridge leg, but with
        # an old t_monotonic — go_table's criterion must reject it as stale.
        # Only the next _inject_nav_state("ARRIVED") (fresh ts) succeeds.
        logging.info(">>> T+6.0s: command '식탁으로 가'")
        ts.on_audio("식탁으로 가", ts=time.monotonic())
        time.sleep(2.0)

        logging.info(">>> T+8.0s: inject nav.mode=ARRIVED (table)")
        _inject_nav_state("ARRIVED")
        time.sleep(1.0)

        # ── home leg ───────────────────────────────────────────────────
        logging.info(">>> T+9.0s: command '집으로 가'  — sub-task='%s'",
                     ts.active_sub_task.name if ts.active_sub_task else None)
        ts.on_audio("집으로 가", ts=time.monotonic())
        time.sleep(2.0)

        logging.info(">>> T+11.0s: inject nav.mode=ARRIVED (home)")
        _inject_nav_state("ARRIVED")
        time.sleep(1.5)

        logging.info(">>> T+12.5s: state=%s (scenario should have ended → idle)", ts.state.value)

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
