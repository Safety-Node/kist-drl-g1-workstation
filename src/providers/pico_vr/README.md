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

**arm64 (Jetson Orin, etc.)**

```bash
sudo dpkg -i gear_sonic_deploy/thirdparty/roboticsservice_1.0.0.0_arm64.deb
```

설치 확인 및 실행:

```bash
ls /opt/apps/roboticsservice
/opt/apps/roboticsservice/runService.sh
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
