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

TODO(REQ-33) [TASK-41]: publish methods.
                        Watchdog fail must LOG + DROP, never raise — callers
                        (VLA, TTS) are fire-and-forget via _schedule_coro and
                        cannot retrieve exceptions.
TODO(REQ-32) [TASK-41]: reconnect strategy on LAN drop.
TODO(REQ-32) [TASK-41]: register_estop_callback push API for ≤200 ms E-STOP
                        propagation budget — polling via TaskSrvBg.tick (10 Hz,
                        100 ms period) is borderline.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, TypedDict

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

from .singleton import singleton


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
        ros_domain_id: int = 0,
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
        ros_domain_id : int
            ROS_DOMAIN_ID shared with the G1 onboard via comm_bridge.
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
        self._ros_domain_id = ros_domain_id
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

        # State topic caches (Reliable)
        self._buf_state: TopicCache = TopicCache()
        self._speaker_state: TopicCache = TopicCache()
        self._estop: TopicCache = TopicCache()

        logging.info(
            "UnitreeG1Provider: skeleton initialized "
            "(domain=%d, host=%s, sensor_ttl=%dms, state_ttl=%dms)",
            ros_domain_id, comm_bridge_host, sensor_ttl_ms, state_ttl_ms,
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
            self._connected = True

            _qos_be = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._node.create_subscription(
                Imu,
                "/bridge/sensors/imu",
                self._on_imu_base,
                _qos_be,
            )
            logging.info("UnitreeG1Provider: subscribed /bridge/sensors/imu")

            # TODO(REQ-32) [TASK-41]: bind remaining subscribers with correct QoS
            # TODO(REQ-33) [TASK-41]: bind publishers
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

        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2.0)
            self._executor = None

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=3.0)
            if self._spin_thread.is_alive():
                logging.warning("UnitreeG1Provider: spin thread did not stop within 3s")
            self._spin_thread = None

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
        """Callback for /bridge/sensors/imu (base IMU, BestEffort)."""
        self._imu_base = TopicCache(value=msg, last_seen_ts=time.monotonic())

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
        # TODO(REQ-32) [TASK-41]: append to subscriber list under lock
        raise NotImplementedError("UnitreeG1Provider.register_audio_callback: TBD [TASK-41]")

    def unregister_audio_callback(
        self, callback: Callable[[bytes, float], None]
    ) -> None:
        """Remove ``callback``; no-op if not registered."""
        # TODO(REQ-32) [TASK-41]: remove from subscriber list under lock
        raise NotImplementedError("UnitreeG1Provider.unregister_audio_callback: TBD [TASK-41]")

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
        # TODO(REQ-32) [TASK-41]: append to subscriber list under lock
        raise NotImplementedError("UnitreeG1Provider.register_estop_callback: TBD [TASK-41]")

    def unregister_estop_callback(
        self, callback: Callable[[bool, float], None]
    ) -> None:
        """Remove ``callback``; no-op if not registered."""
        raise NotImplementedError("UnitreeG1Provider.unregister_estop_callback: TBD [TASK-41]")

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
        # TODO(REQ-33) [TASK-41]: replace with Reliable caches (estop,
        #   buf_state, speaker_state) once those subscriptions are added.
        #   For now, IMU (BestEffort) is used as a proxy — acceptable while
        #   Reliable subscriptions are not yet wired.
        ttl_s = self._heartbeat_timeout_ms / 1000.0
        now = time.monotonic()
        return not self._imu_base.stale(now=now, ttl_s=ttl_s)

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
        # TODO(REQ-33) [TASK-41]: if not self.comm_bridge_alive(): log + drop
        # TODO(REQ-33) [TASK-41]: serialize JointCmdChunk → g1_onboard_msgs/JointCmdChunk, publish
        raise NotImplementedError("UnitreeG1Provider.publish_joint_chunk_arm: TBD [TASK-41]")

    def publish_joint_chunk_low(self, chunk: JointCmdChunk) -> None:
        """
        Publish a ``JointCmdChunk`` (rt/lowcmd path) to ``/bridge/cmd/low``.

        2026-05-26 — wire unit is the chunk. NEW
        2026-05-22 (whole-body VLA walking) topic; weight ignored on NX
        side. NX motor_controller unpacks ``chunk.steps`` into
        ``joint_buf`` and paces at 100 Hz.
        """
        # TODO(REQ-33) [TASK-41]: if not self.comm_bridge_alive(): log + drop
        # TODO(REQ-33) [TASK-41]: serialize + publish
        raise NotImplementedError("UnitreeG1Provider.publish_joint_chunk_low: TBD [TASK-41]")

    def publish_loco_cmd(self, loco_command: LocoCommand) -> None:
        """
        Publish a LocoCommand to ``/bridge/cmd/loco``.

        Used by MoveConnector for posture transitions (StandUp / SitDown /
        Damp / BalanceStand) — see ``_LOCO_MAP`` in move_connector.py.
        """
        # TODO(REQ-33) [TASK-41]: if not self.comm_bridge_alive(): log + drop
        # TODO(REQ-33) [TASK-41]: serialize + publish
        raise NotImplementedError("UnitreeG1Provider.publish_loco_cmd: TBD [TASK-41]")

    def publish_twist(self, vx: float, vy: float, vyaw: float) -> None:
        """Publish geometry_msgs/Twist on /bridge/cmd/vel for NX motor_controller's
        LocoClient.Move(vx, vy, vyaw) dispatch.

        Called by NavigationProvider at NavigationProviderConfig.control_rate_hz.
        Continuous walking velocity. Discrete LocoClient preset transitions
        (StandUp / SitDown / BalanceStand / ZeroTorque) go through a separate
        path (send_loco_command / LocoCommand TypedDict).

        TODO(REQ "Twist Cmd Wire") [TASK-41]: implement DDS publisher.
        """
        raise NotImplementedError("UnitreeG1Provider.publish_twist — scaffold")

    def publish_audio_out(self, pcm: bytes) -> None:
        """
        Publish synthesized PCM audio to ``/bridge/cmd/audio_out`` (onboard speaker).
        TTS Provider → Speak Connector → here.
        """
        # TODO(REQ-29) [TASK-41]: if not self.comm_bridge_alive(): log + drop
        # TODO(REQ-29) [TASK-41]: chunking + flow control + publish
        raise NotImplementedError("UnitreeG1Provider.publish_audio_out: TBD [TASK-41]")
