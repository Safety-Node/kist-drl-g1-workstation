# pico_vr

Module for reading body tracking data from the PICO VR headset.

## Architecture

<!-- architecture diagram -->

Reference: `gear_sonic/scripts/pico_manager_thread_server.py` — `PicoReader` class

## Dependencies

| Package | Purpose |
|---|---|
| `xrobotoolkit_sdk` | XRoboToolkit Python bindings. `roboticsservice` is installed as part of this package |

### Installing xrobotoolkit_sdk

1. Install system dependencies

```bash
sudo apt install libx264-dev libavcodec-dev libavutil-dev libswscale-dev libyaml-cpp-dev
```

2. Install `xrobotoolkit_sdk` (wheel provided by PICO)
