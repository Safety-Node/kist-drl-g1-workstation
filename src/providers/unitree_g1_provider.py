"""
UnitreeG1 Provider [TASK-41, REQ-32/33]

PC-side facade for ``/bridge/*`` DDS topics. Single owner of subscribers +
publishers; downstream providers poll the data properties.
rclpy + CycloneDDS direct (no Zenoh bridge daemon).

Subscribes (BestEffort unless noted):
  /bridge/sensors/color/compressed   sensor_msgs/CompressedImage
  /bridge/sensors/depth/image_raw    sensor_msgs/Image
  /bridge/sensors/audio_pcm          g1_onboard_msgs/AudioPCM
  /bridge/sensors/joint_states       sensor_msgs/JointState
  /bridge/sensors/imu                sensor_msgs/Imu        (base, IF-41)
  /bridge/sensors/imu/ankle_left     sensor_msgs/Imu        (NEW 2026-05-22)
  /bridge/sensors/imu/ankle_right    sensor_msgs/Imu        (NEW 2026-05-22)
  /bridge/sensors/uwb_pose           geometry_msgs/PoseStamped
  /bridge/motor/buf_state            g1_onboard_msgs/BufState        [Reliable]
  /bridge/audio/speaker_state        g1_onboard_msgs/SpeakerState    [Reliable]
  /bridge/safety/estop               g1_onboard_msgs/EstopFlag       [Reliable]

Publishes (Reliable):
  /bridge/cmd/arm        g1_onboard_msgs/JointCmdChunk  rt/arm_sdk, IF-6
  /bridge/cmd/low        g1_onboard_msgs/JointCmdChunk  rt/lowcmd, NEW 2026-05-22
  /bridge/cmd/loco       g1_onboard_msgs/LocoCommand    StandUp/Damp/SitDown
  /bridge/cmd/vel        geometry_msgs/Twist            NavigationProvider walking velocity
  /bridge/cmd/audio_out  g1_onboard_msgs/AudioPCM       TTS playback

Deprecated, NOT handled: /bridge/cmd/nav_goal, /bridge/nav/state
(navigation pkg removed 2026-05-22).
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, TypedDict

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, Imu, JointState

from .singleton import singleton

# g1_onboard_msgs imports — deferred to avoid circular import at module load
# (rclpy type support is initialised during rclpy.init(), before these are used).
from g1_onboard_msgs.msg import (
    AudioPCM,
    BufState,
    EstopFlag,
    JointCmd as JointCmdMsg,
    JointCmdChunk as JointCmdChunkMsg,
    LocoCommand as LocoCommandMsg,
    SpeakerState,
)

# LocoCommand name → LocoCommand.action constant (matches move_connector._LOCO_MAP)
_LOCO_NAME_TO_ACTION = {
    "StandUp": LocoCommandMsg.ACTION_STAND_UP,
    "SitDown": LocoCommandMsg.ACTION_SIT_DOWN,
    "Damp": LocoCommandMsg.ACTION_DAMP,
    "BalanceStand": LocoCommandMsg.ACTION_BALANCE_STAND,
}


@dataclass(frozen=True)
class TopicCache:
    """Last-known value + monotonic timestamp for a single subscribed topic.

    **Immutable**: DDS callbacks must replace the parent attribute with a
    fresh ``TopicCache(...)`` instance rather than mutating fields in place.
    Single attribute assignment is atomic under the GIL; readers in other
    threads see either the old or the new full instance, never a half-update
    where ``last_seen_ts`` is new but ``value`` is still old.

    A fresh TopicCache (``last_seen_ts == 0.0``) is treated as stale by
    :meth:`stale` so downstream code can guard against "never received yet".
    """

    value: Optional[Any] = None
    last_seen_ts: float = 0.0  # monotonic seconds; 0.0 == never received

    def stale(self, now: float, ttl_s: float) -> bool:
        """True if no message received within ``ttl_s`` of ``now``."""
        return (self.last_seen_ts == 0.0) or ((now - self.last_seen_ts) > ttl_s)


# Outbound command payload shapes. TypedDict gives pyright / IDE help at
# call sites (VLA Provider's chunk split, MoveConnector's loco dispatch)
# without the runtime cost of a Pydantic model — DDS serialisation does
# the final validation.
class JointCmd(TypedDict):
    """
    Per-step joint command (IF-6 upper body / NEW 2026-05-22 lower body).

    In-process shape for one step inside a ``JointCmdChunk``. The wire
    no longer carries single-step ``JointCmd`` for VLA — chunks are the
    unit (2026-05-26). ``step_index`` is retained for
    trace/log, ``chunk_id`` for self-contained per-step logging.
    """

    joint_names: List[str]
    q: List[float]
    dq: List[float]
    kp: List[float]
    kd: List[float]
    tau_ff: List[float]
    # TODO(REQ-33) [TASK-41]: clarify mode enum semantics per ICD IF-6 +
    # Unitree G1 SDK (rt/arm_sdk vs rt/lowcmd). Typical Unitree motor
    # mode values: 0=disabled/damp, 1=position control (PD), other values
    # SDK-version-specific. Promote to ``Literal[...]`` once locked.
    mode: int
    weight: float          # respected on arm path; ignored on low path
    chunk_id: int          # duplicated on each step for self-contained logging
    step_index: int        # 0..action_horizon-1 within the chunk (trace/log)


class JointCmdChunk(TypedDict):
    """
    Action chunk wire payload (2026-05-26).

    Mirrors ``g1_onboard_msgs/JointCmdChunk.msg``. The PC VLAProvider
    builds one of these per arm/low half per inference; NX
    ``motor_controller`` unpacks ``steps`` into its 100 Hz ring buffer
    and applies ``queue_aggregate.crossfade()`` on ``chunk_id``
    transitions.
    """

    chunk_id: int          # wrap rule: skip 0 on overflow
    steps: List[JointCmd]  # length = action_horizon, currently 16


class LocoCommand(TypedDict):
    """High-level posture transition for Unitree LocoClient."""

    name: Literal["StandUp", "SitDown", "Damp", "BalanceStand"]


@singleton
class UnitreeG1Provider:
    """
    Workstation-side LAN bridge to the G1 onboard ROS 2 (CycloneDDS / Zenoh) stack.

    Owns all ``/bridge/*`` subscribers + publishers. Downstream OM1 providers
    poll the data properties; raw DDS handles never leak out of this class.
    """

    def __init__(
        self,
        ros_domain_id: Optional[int] = None,
        cyclonedds_uri: Optional[str] = None,
        comm_bridge_host: Optional[str] = None,
        heartbeat_timeout_ms: int = 500,
        sensor_ttl_ms: int = 200,
        state_ttl_ms: int = 1000,
        executor_threads: int = 4,
    ):
        """
        Parameters
        ----------
        ros_domain_id : int, optional
            ROS_DOMAIN_ID shared with the G1 onboard via comm_bridge.
            Defaults to ``ROS_DOMAIN_ID`` env var, or 0 if unset.
        cyclonedds_uri : str, optional
            File URI to ``cyclonedds.xml`` (partition filter). When None,
            start() should read ``CYCLONEDDS_URI`` env var, then fall back
            to the repo's ``cyclonedds/cyclonedds.xml``. TODO(REQ-32).
        comm_bridge_host : str, optional
            IP/hostname of the Orin NX running comm_bridge (logging /
            diagnostics only — discovery is multicast). When None, host-
            targeted log lines are skipped. TODO(REQ-32).
        heartbeat_timeout_ms : int
            Max age of last comm_bridge heartbeat before outbound cmds are blocked.
        sensor_ttl_ms : int
            Sensor stream stale threshold (BestEffort topics).
        state_ttl_ms : int
            State stream stale threshold (Reliable topics like estop/buf_state).
        executor_threads : int
            Number of threads in the MultiThreadedExecutor. Each subscription
            gets its own MutuallyExclusiveCallbackGroup by default, so threads
            allow concurrent callback execution across topics — important for
            estop latency not being blocked by slow sensor callbacks (e.g. audio_pcm).
        """
        self._ros_domain_id = ros_domain_id if ros_domain_id is not None else int(os.environ.get("ROS_DOMAIN_ID", "0"))
        self._cyclonedds_uri = cyclonedds_uri
        self._comm_bridge_host = comm_bridge_host
        self._heartbeat_timeout_ms = heartbeat_timeout_ms
        self._sensor_ttl_ms = sensor_ttl_ms
        self._state_ttl_ms = state_ttl_ms
        self._executor_threads = executor_threads

        self._connected = False

        # rclpy lifecycle handles
        self._lock = threading.Lock()
        self._node: Optional[rclpy.node.Node] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None

        # Sensor topic caches (BestEffort)
        self._color: TopicCache = TopicCache()
        self._depth: TopicCache = TopicCache()
        self._audio_pcm: TopicCache = TopicCache()
        self._joint_state: TopicCache = TopicCache()
        self._imu_base: TopicCache = TopicCache()
        self._imu_ankle_left: TopicCache = TopicCache()
        self._imu_ankle_right: TopicCache = TopicCache()
        self._uwb_pose: TopicCache = TopicCache()
        self._location: TopicCache = TopicCache()
        self._occupancy: TopicCache = TopicCache()

        # State topic caches (Reliable)
        self._buf_state: TopicCache = TopicCache()
        self._speaker_state: TopicCache = TopicCache()
        self._estop: TopicCache = TopicCache()

        # Subscriber/publisher handles — kept alive to prevent GC
        self._sub_color = None
        self._sub_depth = None
        self._sub_audio_pcm = None
        self._sub_joint_state = None
        self._sub_imu_base = None
        self._sub_imu_ankle_left = None
        self._sub_imu_ankle_right = None
        self._sub_uwb_pose = None
        self._sub_location = None
        self._sub_occupancy = None
        self._sub_buf_state = None
        self._sub_speaker_state = None
        self._sub_estop = None

        self._pub_arm = None
        self._pub_low = None
        self._pub_loco = None
        self._pub_vel = None
        self._pub_audio_out = None

        # Push-style callback lists — initialized here so register_*_callback
        # is safe to call before start().
        self._cb_lock = threading.Lock()
        self._audio_callbacks: List[Callable[[bytes, float], None]] = []
        self._estop_callbacks: List[Callable[[bool, float], None]] = []

        logging.info(
            "UnitreeG1Provider: initialized "
            "(domain=%d, host=%s, sensor_ttl=%dms, state_ttl=%dms)",
            self._ros_domain_id, comm_bridge_host, sensor_ttl_ms, state_ttl_ms,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Initialize rclpy/CycloneDDS participant + spin subscribers/publishers."""
        with self._lock:
            if self._connected:
                return

            # CYCLONEDDS_URI: constructor arg > env var already set.
            uri = self._cyclonedds_uri or os.environ.get("CYCLONEDDS_URI")
            if uri:
                os.environ["CYCLONEDDS_URI"] = uri
                logging.info("UnitreeG1Provider: CYCLONEDDS_URI=%s", uri)
            else:
                logging.warning(
                    "UnitreeG1Provider: CYCLONEDDS_URI not set — "
                    "source env.sh or pass cyclonedds_uri= to constructor"
                )

            os.environ["ROS_DOMAIN_ID"] = str(self._ros_domain_id)

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node("unitree_g1_provider")
            self._executor = MultiThreadedExecutor(num_threads=self._executor_threads)
            self._executor.add_node(self._node)

            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                name="unitree_g1_spin",
                daemon=True,
            )
            self._spin_thread.start()

            _qos_be = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            _qos_rel = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            _qos_rel_pub = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )

            # Per-subscription MutuallyExclusiveCallbackGroup — rclpy 의 기본
            # callback group 은 노드 단일 MEC 라서, 인자 미지정 시 모든 구독이
            # **순차 실행**돼 MultiThreadedExecutor 의 병렬성을 무효화한다.
            # audio_pcm(50Hz) 이 Image deserialize 뒤로 밀리면 BestEffort 라
            # depth 가 아무리 커도 stt 가 누락된 PCM 을 받게 됨. 각 구독에
            # 전용 MEC 를 주면 executor_threads 만큼 진짜 병렬로 실행된다.
            cb_audio = MutuallyExclusiveCallbackGroup()
            cb_color = MutuallyExclusiveCallbackGroup()
            cb_depth = MutuallyExclusiveCallbackGroup()
            cb_joint = MutuallyExclusiveCallbackGroup()
            cb_imu_base = MutuallyExclusiveCallbackGroup()
            cb_imu_ankle_left = MutuallyExclusiveCallbackGroup()
            cb_imu_ankle_right = MutuallyExclusiveCallbackGroup()
            cb_uwb = MutuallyExclusiveCallbackGroup()
            cb_location = MutuallyExclusiveCallbackGroup()
            cb_occupancy = MutuallyExclusiveCallbackGroup()
            cb_buf = MutuallyExclusiveCallbackGroup()
            cb_speaker = MutuallyExclusiveCallbackGroup()
            cb_estop = MutuallyExclusiveCallbackGroup()

            # BestEffort subscribers
            self._sub_color = self._node.create_subscription(
                CompressedImage, "/bridge/sensors/color/compressed",
                self._on_color, _qos_be, callback_group=cb_color,
            )
            self._sub_depth = self._node.create_subscription(
                Image, "/bridge/sensors/depth/image_raw",
                self._on_depth, _qos_be, callback_group=cb_depth,
            )
            self._sub_audio_pcm = self._node.create_subscription(
                AudioPCM, "/bridge/sensors/audio_pcm",
                self._on_audio_pcm, _qos_be, callback_group=cb_audio,
            )
            self._sub_joint_state = self._node.create_subscription(
                JointState, "/bridge/sensors/joint_states",
                self._on_joint_state, _qos_be, callback_group=cb_joint,
            )
            self._sub_imu_base = self._node.create_subscription(
                Imu, "/bridge/sensors/imu",
                self._on_imu_base, _qos_be, callback_group=cb_imu_base,
            )
            self._sub_imu_ankle_left = self._node.create_subscription(
                Imu, "/bridge/sensors/imu/ankle_left",
                self._on_imu_ankle_left, _qos_be, callback_group=cb_imu_ankle_left,
            )
            self._sub_imu_ankle_right = self._node.create_subscription(
                Imu, "/bridge/sensors/imu/ankle_right",
                self._on_imu_ankle_right, _qos_be, callback_group=cb_imu_ankle_right,
            )
            self._sub_uwb_pose = self._node.create_subscription(
                PoseStamped, "/bridge/sensors/uwb_pose",
                self._on_uwb_pose, _qos_be, callback_group=cb_uwb,
            )
            self._sub_location = self._node.create_subscription(
                PoseStamped, "/bridge/sensors/location",
                self._on_location, _qos_be, callback_group=cb_location,
            )
            self._sub_occupancy = self._node.create_subscription(
                OccupancyGrid, "/bridge/sensors/lidar/occupancy",
                self._on_occupancy, _qos_be, callback_group=cb_occupancy,
            )
            self._sub_location = self._node.create_subscription(
                PoseStamped, "/bridge/sensors/location",
                self._on_location, _qos_be,
            )
            self._sub_occupancy = self._node.create_subscription(
                OccupancyGrid, "/bridge/sensors/lidar/occupancy",
                self._on_occupancy, _qos_be,
            )

            # Reliable subscribers
            self._sub_buf_state = self._node.create_subscription(
                BufState, "/bridge/motor/buf_state",
                self._on_buf_state, _qos_rel, callback_group=cb_buf,
            )
            self._sub_speaker_state = self._node.create_subscription(
                SpeakerState, "/bridge/audio/speaker_state",
                self._on_speaker_state, _qos_rel, callback_group=cb_speaker,
            )
            self._sub_estop = self._node.create_subscription(
                EstopFlag, "/bridge/safety/estop",
                self._on_estop, _qos_rel, callback_group=cb_estop,
            )

            # Reliable publishers
            self._pub_arm = self._node.create_publisher(
                JointCmdChunkMsg, "/bridge/cmd/arm", _qos_rel_pub,
            )
            self._pub_low = self._node.create_publisher(
                JointCmdChunkMsg, "/bridge/cmd/low", _qos_rel_pub,
            )
            self._pub_loco = self._node.create_publisher(
                LocoCommandMsg, "/bridge/cmd/loco", _qos_rel_pub,
            )
            self._pub_vel = self._node.create_publisher(
                Twist, "/bridge/cmd/vel", _qos_rel_pub,
            )
            self._pub_audio_out = self._node.create_publisher(
                AudioPCM, "/bridge/cmd/audio_out", _qos_rel_pub,
            )

            self._connected = True
            logging.info(
                "UnitreeG1Provider started (domain=%d, node=%s)",
                self._ros_domain_id,
                self._node.get_name(),
            )

    def stop(self) -> None:
        """Tear down DDS participant cleanly."""
        with self._lock:
            if not self._connected:
                return
            self._connected = False

        # 1. executor에 종료 신호 → spin thread가 루프 탈출
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=0.0)

        # 2. spin thread 완전 종료 대기 (executor 멈춘 뒤)
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=3.0)
            if self._spin_thread.is_alive():
                logging.warning("UnitreeG1Provider: spin thread did not stop within 3s")
            self._spin_thread = None

        # 3. executor 리소스 해제
        self._executor = None

        # 4. node destroy (spin thread가 완전히 끝난 뒤)
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

        if rclpy.ok():
            rclpy.shutdown()

        logging.info("UnitreeG1Provider stopped")

    # ------------------------------------------------------------------
    # DDS subscription callbacks (called from MultiThreadedExecutor threads)
    # ------------------------------------------------------------------
    def _on_imu_base(self, msg: Imu) -> None:
        self._imu_base = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_color(self, msg: CompressedImage) -> None:
        self._color = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_depth(self, msg: Image) -> None:
        self._depth = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_audio_pcm(self, msg: AudioPCM) -> None:
        ts = time.monotonic()
        self._audio_pcm = TopicCache(value=msg, last_seen_ts=ts)
        pcm_bytes = bytes(msg.data)
        with self._cb_lock:
            cbs = list(self._audio_callbacks)
        for cb in cbs:
            try:
                cb(pcm_bytes, ts)
            except Exception:
                logging.exception("UnitreeG1Provider: audio_pcm callback error")

    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_imu_ankle_left(self, msg: Imu) -> None:
        self._imu_ankle_left = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_imu_ankle_right(self, msg: Imu) -> None:
        self._imu_ankle_right = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_uwb_pose(self, msg: PoseStamped) -> None:
        self._uwb_pose = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_buf_state(self, msg: BufState) -> None:
        self._buf_state = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_speaker_state(self, msg: SpeakerState) -> None:
        self._speaker_state = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_estop(self, msg: EstopFlag) -> None:
        ts = time.monotonic()
        self._estop = TopicCache(value=msg, last_seen_ts=ts)
        active = bool(msg.active)
        with self._cb_lock:
            cbs = list(self._estop_callbacks)
        for cb in cbs:
            try:
                cb(active, ts)
            except Exception:
                logging.exception("UnitreeG1Provider: estop callback error")

    def _on_location(self, msg: PoseStamped) -> None:
        self._location = TopicCache(value=msg, last_seen_ts=time.monotonic())

    def _on_occupancy(self, msg: OccupancyGrid) -> None:
        self._occupancy = TopicCache(value=msg, last_seen_ts=time.monotonic())

    # ------------------------------------------------------------------
    # Sensor data properties (BestEffort, sensor_ttl_ms)
    # ------------------------------------------------------------------
    @property
    def color(self) -> TopicCache:
        """Latest ``/bridge/sensors/color/compressed`` (CompressedImage)."""
        return self._color

    @property
    def depth(self) -> TopicCache:
        """Latest ``/bridge/sensors/depth/image_raw`` (Image)."""
        return self._depth

    @property
    def audio_pcm(self) -> TopicCache:
        """Latest ``/bridge/sensors/audio_pcm`` (AudioPCM chunk)."""
        return self._audio_pcm

    @property
    def joint_state(self) -> TopicCache:
        """Latest ``/bridge/sensors/joint_states`` (JointState)."""
        return self._joint_state

    @property
    def imu_base(self) -> TopicCache:
        """Latest ``/bridge/sensors/imu`` (base IMU, IF-41)."""
        return self._imu_base

    @property
    def imu_ankle_left(self) -> TopicCache:
        """Latest ``/bridge/sensors/imu/ankle_left`` (NEW 2026-05-22, GearSonic input)."""
        return self._imu_ankle_left

    @property
    def imu_ankle_right(self) -> TopicCache:
        """Latest ``/bridge/sensors/imu/ankle_right`` (NEW 2026-05-22, GearSonic input)."""
        return self._imu_ankle_right

    @property
    def uwb_pose(self) -> TopicCache:
        """Latest ``/bridge/sensors/uwb_pose`` — TaskSrvProvider locomotion sub-task success."""
        return self._uwb_pose

    @property
    def location(self) -> TopicCache:
        """Latest ``/bridge/sensors/location`` — EKF-fused pose (PoseStamped, map frame)."""
        return self._location

    @property
    def occupancy(self) -> TopicCache:
        """Latest ``/bridge/sensors/lidar/occupancy`` — 2D obstacle grid (OccupancyGrid)."""
        return self._occupancy

    # ------------------------------------------------------------------
    # State properties (Reliable, state_ttl_ms)
    # ------------------------------------------------------------------
    @property
    def buf_state(self) -> TopicCache:
        """
        Latest ``/bridge/motor/buf_state`` (motor ring buffer telemetry).

        Consumer (TBD): VLA Provider may poll this to detect NX ring-buffer
        near-empty and proactively re-trigger ``infer()`` so the next chunk
        lands before the previous one fully drains. Underflow itself is
        handled by NX motor_controller (republish last step at 100 Hz).
        """
        return self._buf_state

    @property
    def speaker_state(self) -> TopicCache:
        """Latest ``/bridge/audio/speaker_state`` (TTS playback, STT echo-cancel hint)."""
        return self._speaker_state

    @property
    def estop(self) -> TopicCache:
        """Latest ``/bridge/safety/estop`` (EstopFlag from NX safety_monitor)."""
        return self._estop

    # ------------------------------------------------------------------
    # Push-style sensor subscriptions (for callers that need low-latency
    # delivery rather than polling the TopicCache property)
    # ------------------------------------------------------------------
    def register_audio_callback(
        self, callback: Callable[[bytes, float], None]
    ) -> None:
        """
        Subscribe to ``/bridge/sensors/audio_pcm`` with a push callback
        ``cb(pcm, ts)``. Used by STTProvider to avoid the polling latency
        + duplicate-detection burden of reading the audio_pcm TopicCache.

        Multi-subscriber; thread-safe (callback list under lock).
        """
        with self._cb_lock:
            if callback not in self._audio_callbacks:
                self._audio_callbacks.append(callback)

    def unregister_audio_callback(
        self, callback: Callable[[bytes, float], None]
    ) -> None:
        """Remove ``callback``; no-op if not registered."""
        with self._cb_lock:
            try:
                self._audio_callbacks.remove(callback)
            except ValueError:
                pass

    def register_estop_callback(
        self, callback: Callable[[bool, float], None]
    ) -> None:
        """
        Subscribe to ``/bridge/safety/estop`` with a push callback
        ``cb(active, ts)``. Push (not poll) so consumers (TaskSrvProvider /
        VLA Provider) can hit the ≤200 ms E-STOP propagation budget — a
        TaskSrvBg poll at 10 Hz would burn up to 100 ms of that budget on
        its own.
        """
        with self._cb_lock:
            if callback not in self._estop_callbacks:
                self._estop_callbacks.append(callback)

    def unregister_estop_callback(
        self, callback: Callable[[bool, float], None]
    ) -> None:
        """Remove ``callback``; no-op if not registered."""
        with self._cb_lock:
            try:
                self._estop_callbacks.remove(callback)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Health / stale helpers
    # ------------------------------------------------------------------
    def comm_bridge_alive(self) -> bool:
        """
        True if any **Reliable** ``/bridge/*`` topic was seen within
        ``heartbeat_timeout_ms``. BestEffort sensors (audio_pcm, IMU,
        joint_states) are excluded — they keep flowing from publisher cache
        even if comm_bridge has hung, so they are not a true liveness
        signal. Reliable topics (estop, buf_state, speaker_state) require
        the bridge to be actively forwarding to advance ``last_seen_ts``.
        """
        # TODO: re-enable once NX bridge publishes estop/buf_state/speaker_state
        return True
        ttl_s = self._heartbeat_timeout_ms / 1000.0
        now = time.monotonic()
        return (
            not self._estop.stale(now=now, ttl_s=ttl_s)
            or not self._buf_state.stale(now=now, ttl_s=ttl_s)
            or not self._speaker_state.stale(now=now, ttl_s=ttl_s)
        )

    # ------------------------------------------------------------------
    # Publishers (cmd outbound, Reliable)
    #
    # All publish methods MUST be log-and-drop on comm_bridge_alive() == False
    # — the upstream callers (VLA / TTS / MoveConnector via TaskSrvProvider
    # _schedule_coro) are fire-and-forget asyncio.Tasks and cannot retrieve
    # an exception raised here.
    # ------------------------------------------------------------------
    def publish_joint_chunk_arm(self, chunk: JointCmdChunk) -> None:
        """
        Publish a ``JointCmdChunk`` (rt/arm_sdk path) to ``/bridge/cmd/arm``.

        2026-05-26 — wire unit is the chunk, not the step.
        NX motor_controller unpacks ``chunk.steps`` into ``joint_buf`` and
        paces at 100 Hz. Weight respected on NX side
        (motor_cmd[29].q ramp). ICD IF-6 (arm joint set) applies to
        ``steps[i].joint_names``.
        """
        self._publish_joint_chunk(self._pub_arm, "/bridge/cmd/arm", chunk)

    def publish_joint_chunk_low(self, chunk: JointCmdChunk) -> None:
        """
        Publish a ``JointCmdChunk`` (rt/lowcmd path) to ``/bridge/cmd/low``.

        2026-05-26 — wire unit is the chunk. NEW
        2026-05-22 (whole-body VLA walking) topic; weight ignored on NX
        side. NX motor_controller unpacks ``chunk.steps`` into
        ``joint_buf`` and paces at 100 Hz.
        """
        self._publish_joint_chunk(self._pub_low, "/bridge/cmd/low", chunk)

    def _publish_joint_chunk(self, pub, topic: str, chunk: JointCmdChunk) -> None:
        """Shared serialisation + publish for arm/low chunk paths."""
        if not self._connected or pub is None:
            logging.warning("UnitreeG1Provider: not started, dropping %s", topic)
            return
        if not self.comm_bridge_alive():
            logging.warning("UnitreeG1Provider: comm_bridge not alive, dropping %s", topic)
            return
        try:
            msg = JointCmdChunkMsg()
            msg.chunk_id = int(chunk["chunk_id"])
            for step in chunk["steps"]:
                s = JointCmdMsg()
                s.joint_names = list(step["joint_names"])
                s.q = list(step["q"])
                s.dq = list(step["dq"])
                s.kp = list(step["kp"])
                s.kd = list(step["kd"])
                s.tau_ff = list(step["tau_ff"])
                s.mode = int(step["mode"])
                s.weight = float(step["weight"])
                s.chunk_id = int(step["chunk_id"])
                s.step_index = int(step["step_index"])
                msg.steps.append(s)
            pub.publish(msg)
        except Exception:
            logging.exception("UnitreeG1Provider: failed to publish %s", topic)

    def publish_loco_cmd(self, loco_command: LocoCommand) -> None:
        """
        Publish a LocoCommand to ``/bridge/cmd/loco``.

        Used by MoveConnector for posture transitions (StandUp / SitDown /
        Damp / BalanceStand) — see ``_LOCO_MAP`` in move_connector.py.
        """
        if not self._connected or self._pub_loco is None:
            logging.warning("UnitreeG1Provider: not started, dropping loco_cmd")
            return
        if not self.comm_bridge_alive():
            logging.warning("UnitreeG1Provider: comm_bridge not alive, dropping loco_cmd")
            return
        try:
            action = _LOCO_NAME_TO_ACTION.get(loco_command["name"])
            if action is None:
                logging.error(
                    "UnitreeG1Provider: unknown loco command '%s', dropping",
                    loco_command["name"],
                )
                return
            msg = LocoCommandMsg()
            msg.action = action
            self._pub_loco.publish(msg)
        except Exception:
            logging.exception("UnitreeG1Provider: failed to publish loco_cmd")

    def publish_twist(self, vx: float, vy: float, vyaw: float) -> None:
        """Publish geometry_msgs/Twist on /bridge/cmd/vel for NX motor_controller's
        LocoClient.Move(vx, vy, vyaw) dispatch.

        Called by NavigationProvider at NavigationProviderConfig.control_rate_hz.
        Continuous walking velocity. Discrete LocoClient preset transitions
        (StandUp / SitDown / BalanceStand / ZeroTorque) go through a separate
        path (send_loco_command / LocoCommand TypedDict).
        """
        if not self._connected or self._pub_vel is None:
            logging.warning("UnitreeG1Provider: not started, dropping twist")
            return
        if not self.comm_bridge_alive():
            logging.warning("UnitreeG1Provider: comm_bridge not alive, dropping twist")
            return
        try:
            msg = Twist()
            msg.linear.x = float(vx)
            msg.linear.y = float(vy)
            msg.angular.z = float(vyaw)
            self._pub_vel.publish(msg)
        except Exception:
            logging.exception("UnitreeG1Provider: failed to publish twist")

    def publish_audio_out(self, pcm: bytes) -> None:
        """
        Publish synthesized PCM audio to ``/bridge/cmd/audio_out`` (onboard speaker).
        TTS Provider → Speak Connector → here.

        Assumes mono 16-bit 16 kHz PCM (standard TTS output).
        Payloads exceeding 65500 B are dropped with a warning — caller
        should chunk before calling if utterances are long.
        """
        if not self._connected or self._pub_audio_out is None:
            logging.warning("UnitreeG1Provider: not started, dropping audio_out")
            return
        if not self.comm_bridge_alive():
            logging.warning("UnitreeG1Provider: comm_bridge not alive, dropping audio_out")
            return
        _MAX_PAYLOAD = 65500
        if len(pcm) > _MAX_PAYLOAD:
            logging.warning(
                "UnitreeG1Provider: audio_out payload %d B exceeds %d B limit, dropping",
                len(pcm), _MAX_PAYLOAD,
            )
            return
        try:
            msg = AudioPCM()
            msg.sample_rate = 16000
            msg.channels = 1
            msg.bit_depth = 16
            msg.data = list(pcm)
            self._pub_audio_out.publish(msg)
        except Exception:
            logging.exception("UnitreeG1Provider: failed to publish audio_out")
