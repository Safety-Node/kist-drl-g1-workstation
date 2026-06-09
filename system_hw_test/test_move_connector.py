"""
Unit tests for MoveConnector — TASK-44 / CONV-009 verification.

No ROS2 / NX required. All Provider deps are replaced with Mock objects.

Test cases
----------
1. loco path  : "stand up" / "sit down" / "damp" / "balance stand"
               → publish_loco_cmd called with correct name
2. nav path   : "go to the fridge" / "walk to the door" / etc.
               → submit_nav_subtask called
3. VLA path   : generic prompt ("grab the bottle")
               → vla.infer awaited
4. exception swallow : VLA.infer raises RuntimeError
               → connect() does NOT re-raise (fire-and-forget contract)
5. cancelled  : CancelledError propagates (asyncio task lifecycle)
6. priority   : loco wins over nav when both keywords match

Run:
    cd ~/kist-drl-g1-workstation
    source env.sh
    python3 system_hw_test/test_move_connector.py
"""

import asyncio
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def _result(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}" + (f" — {detail}" if detail else ""))
    return ok


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _make_connector():
    """Build a MoveConnector with all Provider deps mocked."""
    with patch("providers.unitree_g1_provider.UnitreeG1Provider"), \
         patch("providers.vla_provider.VLAProvider"), \
         patch("providers.navigation_provider.NavigationProvider"):

        # Import after patch so singleton constructors don't run
        from actions.move.connector.move_connector import MoveConnector
        from actions.base import ActionConfig

        conn = MoveConnector(ActionConfig())

    conn._unitree_g1 = MagicMock()
    conn._vla = MagicMock()
    conn._vla.infer = AsyncMock(return_value=1)
    conn._navigation = MagicMock()
    conn._navigation.submit_nav_subtask = MagicMock()
    return conn


def test_loco_path() -> bool:
    _section("1. Loco path — discrete LocoClient presets")
    from actions.move.interface import MoveInput

    cases = [
        ("stand up",       "StandUp"),
        ("Stand Up",       "StandUp"),   # case-insensitive
        ("sit down",       "SitDown"),
        ("damp",           "Damp"),
        ("balance stand",  "BalanceStand"),
        ("please stand up now", "StandUp"),  # substring match
    ]
    all_ok = True
    for prompt, expected_name in cases:
        conn = _make_connector()
        asyncio.run(conn.connect(MoveInput(action=prompt)))
        called = conn._unitree_g1.publish_loco_cmd.called
        if called:
            got = conn._unitree_g1.publish_loco_cmd.call_args[0][0]["name"]
            ok = got == expected_name
        else:
            ok = False
            got = "(not called)"
        all_ok &= _result(f"'{prompt}' → {expected_name}", ok, f"got={got!r}")
    return all_ok


def test_nav_path() -> bool:
    _section("2. Nav path — NavigationProvider.submit_nav_subtask")
    from actions.move.interface import MoveInput

    cases = [
        "go to the fridge",
        "walk to the door",
        "move to the table",
        "navigate to waypoint A",
        "이동해줘",
        "접근해",
    ]
    all_ok = True
    for prompt in cases:
        conn = _make_connector()
        asyncio.run(conn.connect(MoveInput(action=prompt)))
        ok = conn._navigation.submit_nav_subtask.called
        all_ok &= _result(f"'{prompt}' → submit_nav_subtask", ok)
    return all_ok


def test_vla_path() -> bool:
    _section("3. VLA path — VLAProvider.infer")
    from actions.move.interface import MoveInput

    cases = [
        "grab the bottle",
        "open the drawer",
        "pick up the cup",
        "냉장고 문 열어줘",
    ]
    all_ok = True
    for prompt in cases:
        conn = _make_connector()
        asyncio.run(conn.connect(MoveInput(action=prompt)))
        ok = conn._vla.infer.called
        all_ok &= _result(f"'{prompt}' → vla.infer", ok)
    return all_ok


def test_exception_swallow() -> bool:
    _section("4. Exception swallow — fire-and-forget contract")
    from actions.move.interface import MoveInput

    conn = _make_connector()
    conn._vla.infer = AsyncMock(side_effect=RuntimeError("VLA boom"))

    raised = False
    try:
        asyncio.run(conn.connect(MoveInput(action="grab the bottle")))
    except Exception:
        raised = True

    ok = not raised
    _result("RuntimeError from vla.infer does not propagate", ok)

    # nav path exception
    conn2 = _make_connector()
    conn2._navigation.submit_nav_subtask = MagicMock(side_effect=RuntimeError("nav boom"))
    raised2 = False
    try:
        asyncio.run(conn2.connect(MoveInput(action="go to the fridge")))
    except Exception:
        raised2 = True
    ok2 = not raised2
    _result("RuntimeError from submit_nav_subtask does not propagate", ok2)

    # loco path exception
    conn3 = _make_connector()
    conn3._unitree_g1.publish_loco_cmd = MagicMock(side_effect=RuntimeError("loco boom"))
    raised3 = False
    try:
        asyncio.run(conn3.connect(MoveInput(action="stand up")))
    except Exception:
        raised3 = True
    ok3 = not raised3
    _result("RuntimeError from publish_loco_cmd does not propagate", ok3)

    return ok and ok2 and ok3


def test_cancelled_propagates() -> bool:
    _section("5. CancelledError propagates")
    from actions.move.interface import MoveInput

    conn = _make_connector()
    conn._vla.infer = AsyncMock(side_effect=asyncio.CancelledError())

    async def _run():
        task = asyncio.ensure_future(conn.connect(MoveInput(action="grab the bottle")))
        try:
            await task
            return False  # should have raised
        except asyncio.CancelledError:
            return True

    ok = asyncio.run(_run())
    _result("CancelledError re-raised from connect()", ok)
    return ok


def test_loco_wins_over_nav() -> bool:
    _section("6. Priority — loco wins when both keywords match")
    from actions.move.interface import MoveInput

    # "stand up and walk to the door" — loco keyword wins
    conn = _make_connector()
    asyncio.run(conn.connect(MoveInput(action="stand up and go to the door")))
    loco_called = conn._unitree_g1.publish_loco_cmd.called
    nav_called = conn._navigation.submit_nav_subtask.called
    ok = loco_called and not nav_called
    _result("loco wins over nav on overlap", ok,
            f"loco={loco_called}, nav={nav_called}")
    return ok


def main() -> None:
    print("\n" + "=" * 60)
    print("  MoveConnector Unit Test — TASK-44")
    print("=" * 60)

    results = {
        "loco_path":            test_loco_path(),
        "nav_path":             test_nav_path(),
        "vla_path":             test_vla_path(),
        "exception_swallow":    test_exception_swallow(),
        "cancelled_propagates": test_cancelled_propagates(),
        "loco_wins_over_nav":   test_loco_wins_over_nav(),
    }

    _section("SUMMARY")
    all_pass = True
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  {tag} {name}")
        all_pass = all_pass and ok

    print()
    if all_pass:
        print(f"  {PASS} All tests passed")
        sys.exit(0)
    else:
        print(f"  {FAIL} Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
