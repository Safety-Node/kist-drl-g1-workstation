import contextlib
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xrobotoolkit_sdk as xrt
import yaml

from src.providers.singleton import singleton

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "reader_config.yaml"


@contextlib.contextmanager
def _silence():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_out, 1); os.close(saved_out)
        os.dup2(saved_err, 2); os.close(saved_err)


def _load_config() -> dict:
    raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "service_script":  raw["service"]["script"],
        "polling_sleep_s": raw["polling"]["sleep_s"],
        "stale_timeout_s": raw["connection"]["stale_timeout_s"],
    }


@dataclass
class PicoVRBodyPose:
    body_poses_np: np.ndarray    # (24, 7) [x,y,z,qx,qy,qz,qw] Unity frame, scalar-last
    timestamp_ns: int            # xrt.get_time_stamp_ns()
    timestamp_monotonic: float   # time.monotonic()
    dt: float                    # elapsed time since previous frame (s)
    fps: float                   # instantaneous FPS


@dataclass
class PicoVRPose:
    headset: np.ndarray           # (7,) [x,y,z,qx,qy,qz,qw] Unity frame, scalar-last
    left_controller: np.ndarray   # (7,)
    right_controller: np.ndarray  # (7,)
    timestamp_ns: int
    timestamp_monotonic: float


@dataclass
class PicoVRController:
    left_trigger: float
    right_trigger: float
    left_grip: float
    right_grip: float
    left_joystick: tuple
    right_joystick: tuple
    btn_a: bool
    btn_b: bool
    btn_x: bool
    btn_y: bool


@singleton
class PicoVRReader:
    """PICO body tracking and controller input reader."""

    depends_on = []

    def __init__(self):
        self._config = _load_config()

        self._lock = threading.Lock()
        self._latest_body_pose: Optional[PicoVRBodyPose] = None
        self._latest_pose: Optional[PicoVRPose] = None
        self._latest_controller: Optional[PicoVRController] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._fps: float = 0.0
        self._last_stamp_ns: Optional[int] = None
        self._last_new_data_mono: float = time.monotonic()
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """start background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        subprocess.Popen(
            ["bash", self._config["service_script"]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with _silence():
            xrt.init()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pico_vr_reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("PicoVRReader started")

    def stop(self) -> None:
        """stop background thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("PicoVRReader: thread did not stop within 2s")
        self._thread = None
        try:
            with _silence():
                xrt.close()
        except Exception:
            pass
        with self._lock:
            self._connected = False
        logger.info("PicoVRReader stopped")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def body_pose(self) -> Optional[PicoVRBodyPose]:
        with self._lock:
            return self._latest_body_pose

    @property
    def pose(self) -> Optional[PicoVRPose]:
        with self._lock:
            return self._latest_pose

    @property
    def controller(self) -> Optional[PicoVRController]:
        with self._lock:
            return self._latest_controller

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # waiting for body tracking
            if not xrt.is_body_data_available():
                time.sleep(self._config["polling_sleep_s"])
                self._check_stale()
                continue

            # check new frame with SDK timestamp
            stamp_ns = int(xrt.get_time_stamp_ns())
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(self._config["polling_sleep_s"])
                self._check_stale()
                continue

            # dt / fps
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                self._fps = 1.0 / device_dt
            self._last_stamp_ns = stamp_ns

            now_mono = time.monotonic()

            try:
                body_pose = PicoVRBodyPose(
                    body_poses_np=np.array(xrt.get_body_joints_pose(), dtype=np.float64),
                    timestamp_ns=stamp_ns,
                    timestamp_monotonic=now_mono,
                    dt=device_dt,
                    fps=self._fps,
                )
                pose = PicoVRPose(
                    headset=np.array(xrt.get_headset_pose(), dtype=np.float64),
                    left_controller=np.array(xrt.get_left_controller_pose(), dtype=np.float64),
                    right_controller=np.array(xrt.get_right_controller_pose(), dtype=np.float64),
                    timestamp_ns=stamp_ns,
                    timestamp_monotonic=now_mono,
                )
                controller = PicoVRController(
                    left_trigger=float(xrt.get_left_trigger()),
                    right_trigger=float(xrt.get_right_trigger()),
                    left_grip=float(xrt.get_left_grip()),
                    right_grip=float(xrt.get_right_grip()),
                    left_joystick=tuple(xrt.get_left_axis()),
                    right_joystick=tuple(xrt.get_right_axis()),
                    btn_a=bool(xrt.get_A_button()),
                    btn_b=bool(xrt.get_B_button()),
                    btn_x=bool(xrt.get_X_button()),
                    btn_y=bool(xrt.get_Y_button()),
                )
                with self._lock:
                    self._latest_body_pose = body_pose
                    self._latest_pose = pose
                    self._latest_controller = controller
                    self._connected = True
                self._last_new_data_mono = now_mono

            except Exception:
                logger.exception("PicoVRReader: read error")

    def _check_stale(self) -> None:
        elapsed = time.monotonic() - self._last_new_data_mono
        if elapsed > self._config["stale_timeout_s"]:
            with self._lock:
                if self._connected:
                    logger.warning(
                        "PicoVRReader: no new data for %.1fs — disconnected",
                        elapsed,
                    )
                self._connected = False
                self._latest_body_pose = None
                self._latest_pose = None
                self._latest_controller = None
