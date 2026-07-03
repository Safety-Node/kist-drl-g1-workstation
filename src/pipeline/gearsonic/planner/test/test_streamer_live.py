"""
Live test for PlannerStreamer.

Requires PicoVR headset connected.

Run:
  uv run src/pipeline/gearsonic/planner/test/test_streamer_live.py
"""

import math
import os
import sys
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.boot_manager.service_manager import ServiceManager
from src.pipeline.pico_vr.reader import PicoVRController, PicoVRReader
from src.pipeline.gearsonic.planner.streamer import LocomotionMode, PlannerCommand, PlannerStreamer

PASS     = "\033[92mPASS\033[0m"
FAIL     = "\033[91mFAIL\033[0m"
DEADZONE = 0.15
PRINT_HZ = 10


def _fmt_cmd(cmd: PlannerCommand) -> str:
    return (
        f"mode={cmd.mode:2d}({LocomotionMode(cmd.mode).name:<16})  "
        f"vel={cmd.target_vel:6.3f}  "
        f"move=[{cmd.movement_direction[0]:6.3f}, {cmd.movement_direction[1]:6.3f}]  "
        f"face=[{cmd.facing_direction[0]:6.3f}, {cmd.facing_direction[1]:6.3f}]"
    )


def _fmt_left(ctrl: PicoVRController) -> str:
    lx, ly = ctrl.left_joystick
    return f"left=({lx:+.3f}, {ly:+.3f})"


def _fmt_right(ctrl: PicoVRController) -> str:
    rx, ry = ctrl.right_joystick
    return f"right=({rx:+.3f}, {ry:+.3f})"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _wait_connection(reader: PicoVRReader) -> bool:
    print("[0] Waiting for VR headset ... ", end="", flush=True)
    while True:
        if reader.connected:
            print(PASS)
            return True
        time.sleep(0.2)


def _wait_neutral(reader: PicoVRReader) -> None:
    print("      Return to neutral ... ", end="", flush=True)
    while True:
        ctrl = reader.controller
        if ctrl is not None:
            lx, ly = ctrl.left_joystick
            rx, _  = ctrl.right_joystick
            if math.hypot(lx, ly) < DEADZONE and abs(rx) < DEADZONE:
                print(PASS)
                return
        time.sleep(0.05)


def _check_direction(
    reader:    PicoVRReader,
    streamer:  PlannerStreamer,
    label:     str,
    condition: Callable[[PicoVRController], bool],
    fmt_val:   Callable[[PicoVRController], str],
) -> bool:
    print(f"      {label} ... ", end="", flush=True)
    while True:
        ctrl = reader.controller
        if ctrl is not None and condition(ctrl):
            cmd = streamer.command
            val = fmt_val(ctrl)
            cmd_str = f"  →  {_fmt_cmd(cmd)}" if cmd is not None else ""
            print(f"{PASS}  {val}{cmd_str}")
            _wait_neutral(reader)
            return True
        time.sleep(0.05)


# ------------------------------------------------------------------
# Steps
# ------------------------------------------------------------------

def test_mode(reader: PicoVRReader, streamer: PlannerStreamer) -> bool:
    print("[1] Mode test")

    while streamer.command is None:
        time.sleep(0.05)
    before = streamer.command.mode

    print(f"      Press A+B (mode up, current={LocomotionMode(before).name}) ... ", end="", flush=True)
    while True:
        cmd = streamer.command
        if cmd is not None and cmd.mode == before + 1:
            print(f"{PASS}  → {LocomotionMode(cmd.mode).name}")
            after_up = cmd.mode
            break
        time.sleep(0.05)

    print(f"      Press X+Y (mode down, current={LocomotionMode(after_up).name}) ... ", end="", flush=True)
    while True:
        cmd = streamer.command
        if cmd is not None and cmd.mode == after_up - 1:
            print(f"{PASS}  → {LocomotionMode(cmd.mode).name}")
            return True
        time.sleep(0.05)


def test_left_joystick(reader: PicoVRReader, streamer: PlannerStreamer) -> bool:
    print("[2] Left joystick")
    directions = [
        ("East  (right)",    lambda c: c.left_joystick[0] >  DEADZONE, _fmt_left),
        ("West  (left) ",    lambda c: c.left_joystick[0] < -DEADZONE, _fmt_left),
        ("North (forward)",  lambda c: c.left_joystick[1] >  DEADZONE, _fmt_left),
        ("South (backward)", lambda c: c.left_joystick[1] < -DEADZONE, _fmt_left),
    ]
    return all(_check_direction(reader, streamer, label, cond, fmt) for label, cond, fmt in directions)


def test_right_joystick(reader: PicoVRReader, streamer: PlannerStreamer) -> bool:
    print("[3] Right joystick")
    directions = [
        ("East  (right)", lambda c: c.right_joystick[0] >  DEADZONE, _fmt_right),
        ("West  (left) ", lambda c: c.right_joystick[0] < -DEADZONE, _fmt_right),
    ]
    return all(_check_direction(reader, streamer, label, cond, fmt) for label, cond, fmt in directions)


def test_neutral(reader: PicoVRReader) -> bool:
    print("[4] All to neutral ... ", end="", flush=True)
    while True:
        ctrl = reader.controller
        if ctrl is not None:
            lx, ly = ctrl.left_joystick
            rx, _  = ctrl.right_joystick
            if math.hypot(lx, ly) < DEADZONE and abs(rx) < DEADZONE:
                print(PASS)
                return True
        time.sleep(0.05)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    reader   = PicoVRReader()
    streamer = PlannerStreamer()

    manager = ServiceManager()
    manager.register(reader)

    try:
        manager.start()
        streamer.start()

        steps = [
            ("Connection",     lambda: _wait_connection(reader)),
            ("Mode",           lambda: test_mode(reader, streamer)),
            ("Left joystick",  lambda: test_left_joystick(reader, streamer)),
            ("Right joystick", lambda: test_right_joystick(reader, streamer)),
            ("Neutral",        lambda: test_neutral(reader)),
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
                cmd = streamer.command
                if cmd is not None:
                    print(f"\r  {_fmt_cmd(cmd)}", end="", flush=True)
                time.sleep(1.0 / PRINT_HZ)

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        manager.stop()


if __name__ == "__main__":
    main()
