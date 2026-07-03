"""
Live test for PlannerStreamer.

NavigationProvider is NOT started — a mock is injected via singleton slot.
Requires PicoVR headset connected.

Steps:
  [1] Idle        — no joystick, no nav        → _compute_default
  [2] Controller  — move joystick, no nav      → _compute_from_controller
  [3] Nav only    — dummy nav, idle joystick   → _compute_from_nav
  [4] Nav + ctrl  — dummy nav + joystick       → _compute_from_controller (interrupt)

Run:
  uv run src/pipeline/gearsonic/planner/test/test_streamer_live.py
"""

import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.boot_manager.service_manager import ServiceManager
from src.pipeline.pico_vr.reader import PicoVRReader
from src.pipeline.gearsonic.planner.streamer import LocomotionMode, PlannerCommand, PlannerStreamer
from src.providers.navigation_provider import NavigationProvider, NavVelCmd

PASS = "\033[92mPASS\033[0m"

DUMMY_NAV = NavVelCmd(vx=0.3, vy=0.0, vyaw=0.2)
PRINT_HZ  = 10


class _MockNav:
    def __init__(self):
        self._vel_cmd: Optional[NavVelCmd] = None

    @property
    def vel_cmd(self) -> Optional[NavVelCmd]:
        return self._vel_cmd


def _fmt(cmd: PlannerCommand) -> str:
    return (
        f"mode={cmd.mode:2d}({LocomotionMode(cmd.mode).name:<16})  "
        f"vel={cmd.target_vel:6.3f}  "
        f"move=[{cmd.movement_direction[0]:6.3f}, {cmd.movement_direction[1]:6.3f}]  "
        f"face=[{cmd.facing_direction[0]:6.3f}, {cmd.facing_direction[1]:6.3f}]"
    )


def _run_step(
    streamer: PlannerStreamer,
    mock_nav: _MockNav,
    nav_vel: Optional[NavVelCmd],
    label: str,
    duration: float,
) -> None:
    mock_nav._vel_cmd = nav_vel
    print(f"\n{label}")
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        cmd = streamer.command
        if cmd is not None:
            print(f"\r  {_fmt(cmd)}", end="", flush=True)
        time.sleep(1.0 / PRINT_HZ)
    print()


def main() -> None:
    mock_nav = _MockNav()
    NavigationProvider._singleton_class._singleton_instance = mock_nav

    reader   = PicoVRReader()
    streamer = PlannerStreamer()

    manager = ServiceManager()
    manager.register(reader)

    try:
        manager.start()
        streamer.start()

        print("Waiting for VR headset ... ", end="", flush=True)
        while not reader.connected:
            time.sleep(0.2)
        print(PASS)

        _run_step(streamer, mock_nav, None,       "[1] Idle       (no joystick, no nav)     5s",  5.0)
        _run_step(streamer, mock_nav, None,       "[2] Controller (move joystick)           10s", 10.0)
        _run_step(streamer, mock_nav, DUMMY_NAV,  "[3] Nav only   (dummy nav, idle stick)   10s", 10.0)
        _run_step(streamer, mock_nav, DUMMY_NAV,  "[4] Nav + ctrl (dummy nav + joystick)    10s", 10.0)

        print("\nDone.")

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        manager.stop()
        NavigationProvider.reset()


if __name__ == "__main__":
    main()
