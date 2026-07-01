"""
Live test for PicoVRReader.

Requires:
  - RoboticsService running or auto-started via /opt/apps/roboticsservice/runService.sh
  - PICO headset connected to the same network with body tracking enabled

Run:
  python test_reader_live.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from src.providers.pico_vr.reader import PicoVRReader


def main():
    provider = PicoVRReader()

    try:
        provider.start()
        print("Waiting for PICO body tracking data... (Ctrl+C to stop)")

        while True:
            sample = provider.body_pose

            if sample is None:
                print(f"connected={provider.connected} | no data yet")
                time.sleep(1.0)
                continue

            print(
                f"connected={provider.connected} | "
                f"fps={sample.fps:.1f} | "
                f"dt={sample.dt * 1000:.1f}ms | "
                f"root_pos={sample.body_poses_np[0, :3]}"
            )
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        provider.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
