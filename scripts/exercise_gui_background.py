"""Exercise the REAL GUIBackground WS publisher against the browser renderer.

Runs the actual gui_background.py (not the renderer's mock): wires the real
UnitreeG1Provider + TaskSrvProvider (rclpy stubbed so it imports without ROS;
STT/TTS providers stubbed since GUIBackground only reads their status), injects
a fake camera frame, runs TaskSrvBg, and auto-drives the move_test scenario so
the status overlay animates. Connect the kist-drl-g1-gui renderer to it.

Run from repo root:
    .venv/bin/python scripts/exercise_gui_background.py
    # or:  uv run python scripts/exercise_gui_background.py
Then serve + open the renderer (separate terminals):
    cd ../kist-drl-g1-gui && python3 -m http.server 8080
    # browser: http://localhost:8080/?ws=ws://localhost:8081
Ctrl+C to stop.
"""

import io
import logging
import math
import signal
import sys
import threading
import time
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# rclpy stub so the REAL UnitreeG1Provider imports without ROS.
if "rclpy" not in sys.modules:
    _stub("rclpy", ok=lambda: False, init=lambda **kw: None, shutdown=lambda: None)
    _stub("rclpy.node", Node=object)
    _stub("rclpy.executors", MultiThreadedExecutor=object)
    _stub("rclpy.qos", HistoryPolicy=object, QoSProfile=object, ReliabilityPolicy=object)
    _stub("sensor_msgs")
    _stub("sensor_msgs.msg", Imu=object)

from providers.unitree_g1_provider import TopicCache, UnitreeG1Provider  # noqa: E402
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider  # noqa: E402
from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig  # noqa: E402
from backgrounds.plugins.gui_background import GUIBackground, GUIBackgroundConfig  # noqa: E402


def _make_jpeg(i: int) -> bytes:
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 360), (18, 22, 28))
        d = ImageDraw.Draw(img)
        x = int(320 + 220 * math.sin(i / 12.0))
        d.ellipse([x - 34, 150, x + 34, 218], fill=(60, 150, 255))
        d.text((12, 12), "GUIBackground TEST  frame %d" % i, fill=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except Exception:
        return b"\xff\xd8\xff\xe0test\xff\xd9"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", force=True)

    for cls in (UnitreeG1Provider, TaskSrvProvider):
        cls.reset()  # type: ignore[attr-defined]

    g1 = UnitreeG1Provider()
    ts = TaskSrvProvider(TaskSrvConfig())
    ts.start()  # loads move_test

    stop = threading.Event()

    bg_task = TaskSrvBg(TaskSrvBgConfig())
    bg_task.set_stop_event(stop)
    threading.Thread(target=bg_task.run, name="TaskSrvBg", daemon=True).start()

    gui = GUIBackground(GUIBackgroundConfig(ws_port=8081))
    gui.set_stop_event(stop)
    threading.Thread(target=gui.run, name="GUIBackground", daemon=True).start()

    def frames():
        i = 0
        while not stop.is_set():
            i += 1
            g1._color = TopicCache(value=_make_jpeg(i), last_seen_ts=time.monotonic())
            stop.wait(1 / 10.0)

    threading.Thread(target=frames, name="frames", daemon=True).start()

    def drive():
        def pose(x, y, yaw):
            g1._uwb_pose = TopicCache(value={"x": x, "y": y, "yaw": yaw}, last_seen_ts=time.monotonic())

        while not stop.is_set():
            ts.on_audio("이동 테스트 시작", ts=time.monotonic()); stop.wait(2.5)
            ts.on_audio("냉장고로 가", ts=time.monotonic()); stop.wait(2.5)
            pose(2.10, -0.40, 1.5708); stop.wait(3.0)
            ts.on_audio("식탁으로 가", ts=time.monotonic()); stop.wait(2.5)
            pose(0.50, 0.30, -1.5708); stop.wait(3.0)
            pose(0.0, 0.0, 0.0); stop.wait(3.0)  # scenario ends → IDLE, then loop

    threading.Thread(target=drive, name="scenario", daemon=True).start()

    logging.info(
        ">>> REAL GUIBackground on ws://0.0.0.0:8081 — open renderer "
        "http://localhost:8080/?ws=ws://localhost:8081 ; Ctrl+C to stop"
    )
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        stop.set()
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
