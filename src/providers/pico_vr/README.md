# pico_vr

Module for reading body tracking data from the PICO VR headset.

## Architecture

![architecture](docs/architecture.png)

Reference: `gear_sonic/scripts/pico_manager_thread_server.py` — `PicoReader` class

## Dependencies

| Package | Purpose |
|---|---|
| `xrobotoolkit_sdk` | XRoboToolkit Python bindings |
| RoboticsService | PICO PC service daemon (`/opt/apps/roboticsservice/`) |

### Installing RoboticsService

**Ubuntu 22.04 x86_64**

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```


### Installing xrobotoolkit_sdk

```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
cd XRoboToolkit-PC-Service-Pybind

# 1. Build and copy the native library
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
cd ../../..

mkdir -p lib include
cp XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/

# 2. Install pybind11
pip install pybind11

# 3. Build and install xrobotoolkit_sdk
python setup.py install --user
```

## Environment

Source `env.sh` before running to set `LD_LIBRARY_PATH`:

```bash
source src/providers/pico_vr/env.sh
```

## VR Setup

See [docs/pico_vr_setup.md](docs/pico_vr_setup.md).

## Usage

```python
from src.providers.pico_vr import PicoVRReader

reader = PicoVRReader()
reader.start()

body = reader.body_pose   # PicoVRBodyPose | None  — (24, 7) SMPL joints
vr   = reader.pose        # PicoVRPose | None      — headset + L/R controller poses
ctrl = reader.controller  # PicoVRController | None — buttons, triggers, joysticks

reader.stop()
```

All properties return `None` until the headset connects and body tracking is enabled.
Stale data is cleared to `None` after `stale_timeout_s` (default 5s) of no new frames.
