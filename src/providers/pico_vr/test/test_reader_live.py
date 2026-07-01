"""
Live test for PicoVRReader.

Requires:
  - RoboticsService running or auto-started via /opt/apps/roboticsservice/runService.sh
  - PICO headset connected to the same network with body tracking enabled

Run:
  python test/test_reader_live.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from src.providers.pico_vr.reader import PicoVRController, PicoVRReader

CONNECT_TIMEOUT_S = 15.0
DISCONNECT_TIMEOUT_S = 10.0
RECONNECT_TIMEOUT_S = 15.0
BUTTON_TIMEOUT_S = 10.0

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _prompt_next(msg: str = "") -> None:
    input(f"\n  {msg}Press Enter to continue...")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_connection(reader: PicoVRReader) -> bool:
    print("[1] Connection status ... ", end="", flush=True)
    deadline = time.monotonic() + CONNECT_TIMEOUT_S
    while time.monotonic() < deadline:
        if reader.connected:
            print(PASS)
            return True
        time.sleep(0.2)
    print(FAIL + f" (no connection within {CONNECT_TIMEOUT_S:.0f}s)")
    return False


def test_joints(reader: PicoVRReader) -> bool:
    print("[2] Joint values ... ", end="", flush=True)
    body_pose = reader.body_pose
    pose = reader.pose
    if body_pose is None or pose is None:
        print(FAIL + " (no data)")
        return False

    arr = body_pose.body_poses_np
    if arr.shape != (24, 7):
        print(FAIL + f" (unexpected shape {arr.shape})")
        return False

    # collect 10 frames to average FPS
    fps_samples = []
    last_ns = body_pose.timestamp_ns
    deadline = time.monotonic() + 2.0
    while len(fps_samples) < 10 and time.monotonic() < deadline:
        p = reader.body_pose
        if p is not None and p.timestamp_ns != last_ns:
            fps_samples.append(p.fps)
            last_ns = p.timestamp_ns
        time.sleep(0.002)
    avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0

    print(PASS + f"  (avg fps over {len(fps_samples)} frames: {avg_fps:.1f} Hz)")

    print("  --- body joints ---")
    for i, joint in enumerate(arr):
        x, y, z, qx, qy, qz, qw = joint
        print(f"  joint[{i:02d}]  pos=({x:7.3f}, {y:7.3f}, {z:7.3f})  quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")

    print("  --- pose ---")
    for label, data in [("headset", pose.headset), ("left_ctrl", pose.left_controller), ("right_ctrl", pose.right_controller)]:
        x, y, z, qx, qy, qz, qw = data
        print(f"  {label:<10}  pos=({x:7.3f}, {y:7.3f}, {z:7.3f})  quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")

    return True


def test_disconnect(reader: PicoVRReader) -> bool:
    print(f"[3] Disconnect the VR headset (waiting up to {DISCONNECT_TIMEOUT_S:.0f}s) ... ", end="", flush=True)
    deadline = time.monotonic() + DISCONNECT_TIMEOUT_S
    while time.monotonic() < deadline:
        if not reader.connected:
            print(PASS + f"  (body_pose={reader.body_pose})")
            return True
        time.sleep(0.2)
    print(FAIL + " (still connected)")
    return False


def test_reconnect(reader: PicoVRReader) -> bool:
    print(f"[4] Reconnect the VR headset (waiting up to {RECONNECT_TIMEOUT_S:.0f}s) ... ", end="", flush=True)
    deadline = time.monotonic() + RECONNECT_TIMEOUT_S
    while time.monotonic() < deadline:
        if reader.connected:
            print(PASS)
            return True
        time.sleep(0.2)
    print(FAIL + " (no reconnection)")
    return False


def _check_button(reader: PicoVRReader, label: str, condition) -> bool:
    print(f"  Press {label} ... ", end="", flush=True)
    deadline = time.monotonic() + BUTTON_TIMEOUT_S
    while time.monotonic() < deadline:
        ctrl = reader.controller
        if ctrl is not None and condition(ctrl):
            print(PASS)
            return True
        time.sleep(0.05)
    print(FAIL + f" (timeout {BUTTON_TIMEOUT_S:.0f}s)")
    return False


def test_left_controller(reader: PicoVRReader) -> bool:
    print("[5] Left controller")
    checks = [
        ("left trigger",   lambda c: c.left_trigger > 0.1),
        ("left grip",      lambda c: c.left_grip > 0.1),
        ("left joystick",  lambda c: abs(c.left_joystick[0]) > 0.3 or abs(c.left_joystick[1]) > 0.3),
        ("X button",       lambda c: c.btn_x),
        ("Y button",       lambda c: c.btn_y),
    ]
    return all(_check_button(reader, label, cond) for label, cond in checks)


def test_right_controller(reader: PicoVRReader) -> bool:
    print("[6] Right controller")
    checks = [
        ("right trigger",  lambda c: c.right_trigger > 0.1),
        ("right grip",     lambda c: c.right_grip > 0.1),
        ("right joystick", lambda c: abs(c.right_joystick[0]) > 0.3 or abs(c.right_joystick[1]) > 0.3),
        ("A button",       lambda c: c.btn_a),
        ("B button",       lambda c: c.btn_b),
    ]
    return all(_check_button(reader, label, cond) for label, cond in checks)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    reader = PicoVRReader()
    try:
        reader.start()

        steps = [
            ("Connection",       lambda: test_connection(reader)),
            ("Joint values",     lambda: test_joints(reader)),
            ("Disconnect",       lambda: test_disconnect(reader)),
            ("Reconnect",        lambda: test_reconnect(reader)),
            ("Left controller",  lambda: test_left_controller(reader)),
            ("Right controller", lambda: test_right_controller(reader)),
        ]

        results = {}
        for name, fn in steps:
            ok = fn()
            results[name] = ok
            if not ok:
                print(f"\nStopped at: {name}")
                break
            if name not in ("Disconnect", "Reconnect"):
                _prompt_next()

        print("\n--- Results ---")
        for name, ok in results.items():
            status = PASS if ok else FAIL
            print(f"  {status}  {name}")
        passed = sum(results.values())
        print(f"\n{passed}/{len(results)} passed")

    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
