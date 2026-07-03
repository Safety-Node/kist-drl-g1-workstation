"""
Live test for PlannerStreamer.

NavigationProvider is NOT started — a mock is injected via singleton slot.
Requires PicoVR headset connected.

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
from src.providers.nav_types import NavVelCmd

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

DUMMY_NAV = NavVelCmd(vx=0.3, vy=0.0, vyaw=0.2)
PRINT_HZ  = 10


def _fmt(cmd: PlannerCommand, mode: Optional[int]) -> str:
    mode_str = f"{mode:+d}" if mode is not None else " ?"
    return (
        f"input={mode_str}  "
        f"mode={cmd.mode:2d}({LocomotionMode(cmd.mode).name:<16})  "
        f"vel={cmd.target_vel:6.3f}  "
        f"move=[{cmd.movement_direction[0]:6.3f}, {cmd.movement_direction[1]:6.3f}]  "
        f"face=[{cmd.facing_direction[0]:6.3f}, {cmd.facing_direction[1]:6.3f}]"
    )


class _MockNav:
    def __init__(self):
        self._vel_cmd: Optional[NavVelCmd] = None

    @property
    def vel_cmd(self) -> Optional[NavVelCmd]:
        return self._vel_cmd


# ------------------------------------------------------------------
# Steps
# ------------------------------------------------------------------

def test_connection(reader: PicoVRReader) -> bool:
    print("[0] Waiting for VR headset ... ", end="", flush=True)
    while True:
        if reader.connected:
            print(PASS)
            return True
        time.sleep(0.2)


def test_idle(streamer: PlannerStreamer, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = None
    print("[1] Idle (no joystick, no nav) ... ", end="", flush=True)
    while True:
        if streamer.input_mode == -1:
            print(PASS)
            return True
        time.sleep(0.05)


def test_controller(streamer: PlannerStreamer, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = None
    print("[2] Controller — move left joystick ... ", end="", flush=True)
    while True:
        if streamer.input_mode == 1:
            print(PASS)
            return True
        time.sleep(0.05)


def test_nav_only(streamer: PlannerStreamer, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = DUMMY_NAV
    print("[3] Nav only — release joystick ... ", end="", flush=True)
    while True:
        if streamer.input_mode == 0:
            print(PASS)
            return True
        time.sleep(0.05)


def test_nav_ctrl(streamer: PlannerStreamer, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = DUMMY_NAV
    print("[4] Nav + ctrl — move joystick to override nav ... ", end="", flush=True)
    while True:
        if streamer.input_mode == 1:
            print(PASS)
            return True
        time.sleep(0.05)


def test_mode_up(streamer: PlannerStreamer, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = None
    print("[5] Mode up — press A+B ... ", end="", flush=True)
    while True:
        cmd = streamer.command
        if cmd is not None and cmd.mode == LocomotionMode.SLOW_WALK:
            print(PASS)
            return True
        time.sleep(0.05)


def test_mode_down(streamer: PlannerStreamer, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = None
    print("[6] Mode down — press X+Y ... ", end="", flush=True)
    while True:
        cmd = streamer.command
        if cmd is not None and cmd.mode == LocomotionMode.IDLE:
            print(PASS)
            return True
        time.sleep(0.05)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    mock_nav = _MockNav()

    reader   = PicoVRReader()
    streamer = PlannerStreamer()
    streamer.set_nav_source(mock_nav)

    manager = ServiceManager()
    manager.register(reader)

    try:
        manager.start()
        streamer.start()

        steps = [
            ("Connection",  lambda: test_connection(reader)),
            ("Idle",        lambda: test_idle(streamer, mock_nav)),
            ("Controller",  lambda: test_controller(streamer, mock_nav)),
            ("Nav only",    lambda: test_nav_only(streamer, mock_nav)),
            ("Nav + ctrl",  lambda: test_nav_ctrl(streamer, mock_nav)),
            ("Mode up",     lambda: test_mode_up(streamer, mock_nav)),
            ("Mode down",   lambda: test_mode_down(streamer, mock_nav)),
        ]

        results = {}
        for name, fn in steps:
            ok = fn()
            results[name] = ok
            if not ok:
                print(f"\nStopped at: {name}")
                break

        print("\n--- Results ---")
        for name, ok in results.items():
            print(f"  {PASS if ok else FAIL}  {name}")
        passed = sum(results.values())
        print(f"\n{passed}/{len(results)} passed")

        if passed == len(results):
            print("\nLive (Ctrl+C to exit)")
            while True:
                cmd  = streamer.command
                mode = streamer.input_mode
                if cmd is not None:
                    print(f"\r  {_fmt(cmd, mode)}", end="", flush=True)
                time.sleep(1.0 / PRINT_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        manager.stop()


if __name__ == "__main__":
    main()
