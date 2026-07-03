"""
Live test for PlannerStreamer.

NavigationProvider is NOT started — a mock is injected via singleton slot.
Requires PicoVR headset connected.

Run:
  uv run src/pipeline/gearsonic/planner/test/test_streamer_live.py
"""

import math
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
FAIL = "\033[91mFAIL\033[0m"

DUMMY_NAV       = NavVelCmd(vx=0.3, vy=0.0, vyaw=0.2)
PRINT_HZ        = 10
JOYSTICK_DEADZONE = 0.15  # matches streamer_config.yaml


class _MockNav:
    def __init__(self):
        self._vel_cmd: Optional[NavVelCmd] = None

    @property
    def vel_cmd(self) -> Optional[NavVelCmd]:
        return self._vel_cmd


def _joystick_active(reader: PicoVRReader) -> bool:
    ctrl = reader.controller
    if ctrl is None:
        return False
    lx, ly = ctrl.left_joystick
    return math.hypot(lx, ly) > JOYSTICK_DEADZONE


def _fmt(cmd: PlannerCommand) -> str:
    return (
        f"mode={cmd.mode:2d}({LocomotionMode(cmd.mode).name:<16})  "
        f"vel={cmd.target_vel:6.3f}  "
        f"move=[{cmd.movement_direction[0]:6.3f}, {cmd.movement_direction[1]:6.3f}]  "
        f"face=[{cmd.facing_direction[0]:6.3f}, {cmd.facing_direction[1]:6.3f}]"
    )


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


def test_idle(streamer: PlannerStreamer, reader: PicoVRReader, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = None
    print("[1] Idle (no joystick, no nav) ... ", end="", flush=True)
    while True:
        cmd = streamer.command
        if cmd is not None:
            print(PASS)
            print(f"  {_fmt(cmd)}")
            return True
        time.sleep(1.0 / PRINT_HZ)


def test_controller(streamer: PlannerStreamer, reader: PicoVRReader, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = None
    print("[2] Controller — move left joystick")
    while True:
        cmd = streamer.command
        if cmd is not None:
            print(f"\r  {_fmt(cmd)}", end="", flush=True)
            if _joystick_active(reader):
                print(f"\n  {PASS}")
                return True
        time.sleep(1.0 / PRINT_HZ)


def test_nav_only(streamer: PlannerStreamer, reader: PicoVRReader, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = DUMMY_NAV
    print("[3] Nav only — release joystick")
    while True:
        cmd = streamer.command
        if cmd is not None:
            print(f"\r  {_fmt(cmd)}", end="", flush=True)
            if not _joystick_active(reader):
                print(f"\n  {PASS}")
                return True
        time.sleep(1.0 / PRINT_HZ)


def test_nav_ctrl(streamer: PlannerStreamer, reader: PicoVRReader, mock_nav: _MockNav) -> bool:
    mock_nav._vel_cmd = DUMMY_NAV
    print("[4] Nav + ctrl — move joystick to override nav")
    while True:
        cmd = streamer.command
        if cmd is not None:
            print(f"\r  {_fmt(cmd)}", end="", flush=True)
            if _joystick_active(reader):
                print(f"\n  {PASS}")
                return True
        time.sleep(1.0 / PRINT_HZ)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

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

        steps = [
            ("Connection",  lambda: test_connection(reader)),
            ("Idle",        lambda: test_idle(streamer, reader, mock_nav)),
            ("Controller",  lambda: test_controller(streamer, reader, mock_nav)),
            ("Nav only",    lambda: test_nav_only(streamer, reader, mock_nav)),
            ("Nav + ctrl",  lambda: test_nav_ctrl(streamer, reader, mock_nav)),
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

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        manager.stop()
        NavigationProvider.reset()


if __name__ == "__main__":
    main()
