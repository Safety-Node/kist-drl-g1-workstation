"""
PicoVRReaderProvider

XRoboToolkit SDK(xrobotoolkit_sdk)를 통해 PICO VR 바디 트래킹 데이터를
백그라운드 스레드로 폴링하고 최신 샘플을 스레드 안전하게 제공한다.

참조: gear_sonic/scripts/pico_manager_thread_server.py — PicoReader 클래스

읽는 데이터:
    body_poses_np  ndarray (24, 7)   SMPL 24관절
                                      [x, y, z, qx, qy, qz, qw]
                                      Unity frame, scalar-last 쿼터니언

프레임 감지:
    xrt.get_time_stamp_ns() 변화 기준 (GearSonic 동일)
    동일 timestamp = 새 데이터 없음 → skip

연결 판정:
    xrt.is_body_data_available() == True 이고 데이터가 들어오면 connected=True
    _STALE_TIMEOUT_S 이상 새 데이터 없으면 connected=False
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import xrobotoolkit_sdk as xrt

_ROBOTICS_SERVICE_SCRIPT = "/opt/apps/roboticsservice/runService.sh"

from ..singleton import singleton

logger = logging.getLogger(__name__)

_STALE_TIMEOUT_S = 5.0


@dataclass
class PicoVRBodySample:
    """XRT SDK에서 한 틱에 읽은 SMPL 바디 포즈 스냅샷."""

    body_poses_np: np.ndarray    # (24, 7) [x,y,z,qx,qy,qz,qw] Unity frame, scalar-last
    timestamp_ns: int            # xrt.get_time_stamp_ns()
    timestamp_monotonic: float   # time.monotonic()
    dt: float                    # 직전 프레임 대비 경과 시간 (s)
    fps: float                   # instantaneous FPS


@singleton
class PicoVRReaderProvider:
    """
    PICO 바디 트래킹 싱글턴 Provider.

    사용법:
        provider = PicoVRReaderProvider()
        provider.start()

        sample: PicoVRBodySample | None = provider.get_latest()

        provider.stop()
    """

    def __init__(self, stale_timeout_s: float = _STALE_TIMEOUT_S):
        self._stale_timeout_s = stale_timeout_s

        self._lock = threading.Lock()
        self._latest: Optional[PicoVRBodySample] = None

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
        """백그라운드 폴링 스레드 시작. 중복 호출 시 no-op."""
        if self._thread is not None and self._thread.is_alive():
            return

        subprocess.Popen(["bash", _ROBOTICS_SERVICE_SCRIPT])
        xrt.init()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pico_vr_reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("PicoVRReaderProvider started")

    def stop(self) -> None:
        """백그라운드 스레드 정지."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("PicoVRReaderProvider: thread did not stop within 2s")
        self._thread = None
        try:
            xrt.close()
        except Exception:
            pass
        with self._lock:
            self._connected = False
        logger.info("PicoVRReaderProvider stopped")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def data(self) -> Optional[PicoVRBodySample]:
        with self._lock:
            return self._latest

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    # ------------------------------------------------------------------
    # Background polling  (GearSonic PicoReader._run 동일 로직)
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # 바디 트래킹 준비 대기
            if not xrt.is_body_data_available():
                time.sleep(0.01)
                self._check_stale()
                continue

            # SDK timestamp 로 새 프레임 판별
            stamp_ns = int(xrt.get_time_stamp_ns())
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.01)
                continue

            # dt / fps
            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                self._fps = 1.0 / device_dt
            self._last_stamp_ns = stamp_ns

            now_mono = time.monotonic()

            try:
                body_poses_np = np.array(xrt.get_body_joints_pose(), dtype=np.float64)

                sample = PicoVRBodySample(
                    body_poses_np=body_poses_np,
                    timestamp_ns=stamp_ns,
                    timestamp_monotonic=now_mono,
                    dt=device_dt,
                    fps=self._fps,
                )
                with self._lock:
                    self._latest = sample
                    self._connected = True
                self._last_new_data_mono = now_mono

            except Exception:
                logger.exception("PicoVRReaderProvider: get_body_joints_pose error")


    def _check_stale(self) -> None:
        elapsed = time.monotonic() - self._last_new_data_mono
        if elapsed > self._stale_timeout_s:
            with self._lock:
                if self._connected:
                    logger.warning(
                        "PicoVRReaderProvider: no new data for %.1fs — disconnected",
                        elapsed,
                    )
                self._connected = False
