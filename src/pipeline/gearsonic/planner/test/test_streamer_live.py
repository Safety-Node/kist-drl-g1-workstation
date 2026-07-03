"""
Live test for PlannerStreamer.

NavigationProvider is NOT started — NavVelCmd is injected as dummy.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.boot_manager.service_manager import ServiceManager
from src.pipeline.pico_vr.reader import PicoVRReader
from src.pipeline.gearsonic.planner.streamer import LocomotionMode, PlannerCommand, PlannerStreamer
from src.providers.navigation_provider import NavVelCmd

PASS = "\033[92mPASS\033[0m"

DUMMY_NAV = NavVelCmd(vx=0.3, vy=0.0, vyaw=0.2)
PRINT_HZ  = 10


def _fmt(cmd: PlannerCommand, input_mode: int) -> str:
    return (
        f"input={input_mode:+d}  "
        f"mode={cmd.mode:2d}({LocomotionMode(cmd.mode).name:<16})  "
        f"vel={cmd.target_vel:6.3f}  "
        f"move=[{cmd.movement_direction[0]:6.3f}, {cmd.movement_direction[1]:6.3f}]  "
        f"face=[{cmd.facing_direction[0]:6.3f}, {cmd.facing_direction[1]:6.3f}]"
    )


def _run_step(
    streamer: PlannerStreamer,
    reader: PicoVRReader,
    nav_vel: "NavVelCmd | None",
    label: str,
    duration: float,
) -> None:
    print(f"\n{label}")
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        controller  = reader.controller
        mode        = streamer._input_mode(controller, nav_vel)
        if mode == 1:
            cmd = streamer._compute_from_controller(controller)
        elif mode == 0:
            cmd = streamer._compute_from_nav(nav_vel)
        else:
            cmd = streamer._compute_default()
        print(f"\r  {_fmt(cmd, mode)}", end="", flush=True)
        time.sleep(1.0 / PRINT_HZ)
    print()


def main() -> None:
    reader   = PicoVRReader()
    streamer = PlannerStreamer()

    manager = ServiceManager()
    manager.register(reader)

    try:
        manager.start()

        print("Waiting for VR headset ... ", end="", flush=True)
        while not reader.connected:
            time.sleep(0.2)
        print(PASS)

        _run_step(streamer, reader, None,       "[1] Idle       (no joystick, no nav)     5s", 5.0)
        _run_step(streamer, reader, None,       "[2] Controller (move joystick)           10s", 10.0)
        _run_step(streamer, reader, DUMMY_NAV,  "[3] Nav only   (dummy nav, idle stick)   10s", 10.0)
        _run_step(streamer, reader, DUMMY_NAV,  "[4] Nav + ctrl (dummy nav + joystick)    10s", 10.0)

        print("\nDone.")

    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
