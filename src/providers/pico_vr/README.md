# pico_vr

Module for reading body tracking data from the PICO VR headset.

## Overview

Receives SMPL body tracking data from the PICO headset via the XRoboToolkit SDK.
Polls in a background thread and provides the latest sample in a thread-safe manner.

Output: `body_poses_np` — `(24, 7)` ndarray, Unity frame, scalar-last quaternion  
Joints used: Root(0), Neck(12), L-Wrist(22), R-Wrist(23)

Reference: `gear_sonic/scripts/pico_manager_thread_server.py` — `PicoReader` class

## Dependencies

| Item | Description |
|---|---|
| `xrobotoolkit_sdk` | XRoboToolkit Python bindings. Import error if not installed |
| `/opt/apps/roboticsservice/runService.sh` | Launches the `RoboticsServiceProcess` daemon. Called automatically on `start()` |
| PICO headset | Connects to `RoboticsServiceProcess` over WiFi |
