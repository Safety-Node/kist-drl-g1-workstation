"""
PicoReaderProvider

XRoboToolkit SDK(xrobotoolkit_sdk)를 통해 PICO VR 헤드셋·컨트롤러 포즈와
바디 트래킹 데이터를 백그라운드 스레드로 폴링하고, 최신 샘플을 스레드 안전하게 제공한다.

의존 패키지:
    xrobotoolkit_sdk  — Pico/XRT pybind11 바인딩 (aarch64 / x86_64 빌드)
    numpy

SDK 미설치 환경에서도 import는 성공하며, connected=False / get_latest()=None 을
반환해 상위 코드가 graceful-degrade 할 수 있다.

연결 판정:
    헤드셋 포즈가 identity([0,0,0,0,0,0,1])에서 벗어나면 connected=True.
    바디 트래킹(motion tracker 필요)은 없어도 동작한다.

데이터 레이아웃 (XRT SDK 원본, Unity 좌표계):
    headset_pose          ndarray (7,)    [x, y, z, qx, qy, qz, qw]  scalar-last
    left_controller_pose  ndarray (7,)
    right_controller_pose ndarray (7,)
    body_poses            ndarray (24, 7) SMPL joints — motion tracker 있을 때만 non-None

    좌표 변환은 이 Provider 책임 밖 — 호출부에서 수행한다.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .singleton import singleton

logger = logging.getLogger(__name__)

try:
    import xrobotoolkit_sdk as xrt

    _XRT_AVAILABLE = True
except ImportError:
    xrt = None  # type: ignore[assignment]
    _XRT_AVAILABLE = False
    logger.warning("PicoReaderProvider: xrobotoolkit_sdk not found — running in stub mode")

_MIN_QUAT_NORM = 0.5  # 유효한 포즈의 최소 쿼터니언 노름 (SDK 미연결 시 all-zeros 방어)

# SMPL 인덱스 상수 (body_poses 필드 사용 시)
SMPL_ROOT = 0
SMPL_NECK = 12
SMPL_LEFT_WRIST = 22
SMPL_RIGHT_WRIST = 23

_STALE_TIMEOUT_S = 5.0
_LOG_INTERVAL_S  = 5.0


@dataclass
class VRPoseData:
    """XRT SDK 에서 한 틱에 읽은 VR 기기 포즈 스냅샷."""

    headset_pose: np.ndarray            # (7,)  Unity frame [x,y,z,qx,qy,qz,qw]
    left_controller_pose: np.ndarray    # (7,)
    right_controller_pose: np.ndarray   # (7,)
    timestamp_ns: int                   # time.monotonic_ns() 기반
    timestamp_monotonic: float          # time.monotonic()
    dt: float                           # 직전 프레임 대비 경과 시간 (seconds)
    fps: float                          # EMA 추정 FPS
    body_poses: Optional[np.ndarray] = None  # (24, 7) — motion tracker 있을 때만


# 하위 호환: BodyPoseData 별칭
BodyPoseData = VRPoseData


@dataclass
class ControllerData:
    """컨트롤러 버튼·아날로그 입력 스냅샷."""

    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0
    left_joystick: tuple = field(default_factory=lambda: (0.0, 0.0))
    right_joystick: tuple = field(default_factory=lambda: (0.0, 0.0))
    btn_a: bool = False
    btn_b: bool = False
    btn_x: bool = False
    btn_y: bool = False
    left_menu: bool = False
    right_menu: bool = False


@singleton
class PicoReaderProvider:
    """
    PICO VR 헤드셋 데이터 싱글턴 Provider.

    사용법:
        provider = PicoReaderProvider()
        provider.start()

        data: VRPoseData | None = provider.get_latest()
        ctrl: ControllerData | None = provider.get_controller()

        provider.stop()

    헤드셋 포즈가 identity([0,0,0,0,0,0,1])에서 벗어나면 connected=True.
    바디 트래킹(motion tracker)이 없어도 헤드셋·컨트롤러 포즈는 정상 수신된다.
    """

    def __init__(self, stale_timeout_s: float = _STALE_TIMEOUT_S):
        self._stale_timeout_s = stale_timeout_s

        self._lock = threading.Lock()
        self._latest: Optional[VRPoseData] = None
        self._controller: Optional[ControllerData] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._fps_ema: float = 0.0
        self._last_stamp_ns: Optional[int] = None
        self._last_headset_pose: Optional[np.ndarray] = None
        self._last_new_data_mono: float = time.monotonic()
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """백그라운드 폴링 스레드를 시작한다. 중복 호출 시 no-op."""
        if self._thread is not None and self._thread.is_alive():
            return

        if not _XRT_AVAILABLE:
            logger.warning("PicoReaderProvider.start(): SDK 없음, stub mode 유지")
            return

        xrt.init()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pico_reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("PicoReaderProvider started")

    def stop(self) -> None:
        """백그라운드 스레드를 정지하고 합류를 기다린다."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("PicoReaderProvider: 스레드가 2s 내에 종료되지 않음")
        self._thread = None
        if _XRT_AVAILABLE and xrt is not None:
            try:
                xrt.close()
            except Exception:
                pass
        with self._lock:
            self._connected = False
        logger.info("PicoReaderProvider stopped")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[VRPoseData]:
        """최신 VRPoseData 를 반환한다. 아직 데이터가 없거나 연결이 끊기면 None."""
        with self._lock:
            return self._latest

    def get_controller(self) -> Optional[ControllerData]:
        """최신 ControllerData 를 반환한다."""
        with self._lock:
            return self._controller

    @property
    def connected(self) -> bool:
        """헤드셋에서 새 포즈 데이터가 들어오고 있는지 여부."""
        with self._lock:
            return self._connected

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _run(self) -> None:
        last_log = time.time()

        while not self._stop_event.is_set():
            try:
                headset_pose = np.array(xrt.get_headset_pose(), dtype=np.float64)
            except Exception:
                logger.exception("PicoReaderProvider: get_headset_pose 오류")
                self._check_stale()
                time.sleep(0.001)
                continue

            # 쿼터니언 노름이 0에 가까우면 미연결 (SDK 초기화 중 all-zeros 반환)
            if np.linalg.norm(headset_pose[3:]) < _MIN_QUAT_NORM:
                self._check_stale()
                time.sleep(0.001)
                continue

            # 새 프레임인지 확인 (헤드셋 포즈 변화 기준)
            prev_pose = self._last_headset_pose
            if prev_pose is not None and np.array_equal(headset_pose, prev_pose):
                time.sleep(0.0001)
                continue

            now_mono = time.monotonic()
            stamp_ns = time.monotonic_ns()

            self._last_new_data_mono = now_mono
            self._last_headset_pose = headset_pose.copy()

            prev_stamp_ns = self._last_stamp_ns
            self._last_stamp_ns = stamp_ns

            dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if dt > 0.0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst)

            try:
                left_pose  = np.array(xrt.get_left_controller_pose(),  dtype=np.float64)
                right_pose = np.array(xrt.get_right_controller_pose(), dtype=np.float64)

                body_poses = None
                if xrt.is_body_data_available():
                    body_poses = np.array(xrt.get_body_joints_pose(), dtype=np.float64)

                vr_data = VRPoseData(
                    headset_pose=headset_pose,
                    left_controller_pose=left_pose,
                    right_controller_pose=right_pose,
                    timestamp_ns=stamp_ns,
                    timestamp_monotonic=now_mono,
                    dt=dt,
                    fps=self._fps_ema,
                    body_poses=body_poses,
                )
                controller_data = self._read_controller()

                with self._lock:
                    self._latest = vr_data
                    self._controller = controller_data
                    self._connected = True

            except Exception:
                logger.exception("PicoReaderProvider: XRT 읽기 오류")

            now = time.time()
            if now - last_log >= _LOG_INTERVAL_S:
                logger.info(
                    "PicoReaderProvider: dt=%.2f ms | fps=%.1f",
                    dt * 1000.0,
                    self._fps_ema,
                )
                last_log = now

    def _check_stale(self) -> None:
        elapsed = time.monotonic() - self._last_new_data_mono
        if elapsed > self._stale_timeout_s:
            with self._lock:
                if self._connected:
                    logger.warning(
                        "PicoReaderProvider: %.1fs 동안 새 데이터 없음 — disconnected",
                        elapsed,
                    )
                self._connected = False

    def _read_controller(self) -> ControllerData:
        return ControllerData(
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
            left_menu=bool(xrt.get_left_menu_button()),
            right_menu=bool(xrt.get_right_menu_button()),
        )
