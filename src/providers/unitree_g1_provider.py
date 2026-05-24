"""
UnitreeG1 Provider -- KIST DRL G1 Workstation [TASK-41]
=======================================================

Consolidated PC-side facade for all G1 onboard sensor/state subscriptions
and motion command publishing. Single owner of the Zenoh/CycloneDDS-bridged
``/bridge/*`` topics so downstream OM1 providers (VLA / TaskSrvProvider /
GUI / etc.) just poll data properties without touching DDS themselves.

drawio C4 Container:
    Name        : UnitreeG1 Provider
    Technology  : LAN / Python (Zenoh / CycloneDDS bridge)
    Description : DDS subscriber + topic exposure consolidator.

---

## ICD bridge topics

Subscribes (sensor + state):

| Topic                                     | Type                              | QoS         |
|-------------------------------------------|-----------------------------------|-------------|
| /bridge/sensors/color/compressed          | sensor_msgs/CompressedImage       | BestEffort  |
| /bridge/sensors/depth/image_raw           | sensor_msgs/Image                 | BestEffort  |
| /bridge/sensors/audio_pcm                 | g1_onboard_msgs/AudioPCM          | BestEffort  |
| /bridge/sensors/joint_states              | sensor_msgs/JointState            | BestEffort  |
| /bridge/sensors/imu                       | sensor_msgs/Imu (base, IF-41)     | BestEffort  |
| /bridge/sensors/imu/ankle_left            | sensor_msgs/Imu (NEW 2026-05-22)  | BestEffort  |
| /bridge/sensors/imu/ankle_right           | sensor_msgs/Imu (NEW 2026-05-22)  | BestEffort  |
| /bridge/sensors/uwb_pose                  | geometry_msgs/PoseStamped         | BestEffort  |
| /bridge/motor/buf_state                   | g1_onboard_msgs/BufState          | Reliable    |
| /bridge/audio/speaker_state               | g1_onboard_msgs/SpeakerState      | Reliable    |
| /bridge/safety/estop                      | g1_onboard_msgs/EstopFlag         | Reliable    |

Publishes (cmd outbound, Reliable):

| Topic                  | Type                          | Note                                              |
|------------------------|-------------------------------|---------------------------------------------------|
| /bridge/cmd/arm        | g1_onboard_msgs/JointCmd      | rt/arm_sdk path, IF-6 (weight respected)          |
| /bridge/cmd/low        | g1_onboard_msgs/JointCmd      | rt/lowcmd path, NEW 2026-05-22 (weight ignored)   |
| /bridge/cmd/loco       | g1_onboard_msgs/LocoCommand   | StandUp/Damp/SitDown — usage scope TBD            |
| /bridge/cmd/audio_out  | g1_onboard_msgs/AudioPCM      | TTS playback to onboard speaker                   |

Deprecated topics intentionally NOT handled (audit only):

| Topic                  | Reason                                                                |
|------------------------|-----------------------------------------------------------------------|
| /bridge/cmd/nav_goal   | Nav Cmd Goal ICD [DEPRECATED 2026-05-22] — navigation pkg removed     |
| /bridge/nav/state      | navigation pkg removed; no producer on NX                             |

---

## Downstream consumers (PC providers polling this facade)

- VLA Provider           : color/depth + joint_state + IMU (base + ankle L/R) + buf_state
- TaskSrvProvider        : uwb_pose + joint_state (sub-task success polling)
- STT Provider           : audio_pcm + speaker_state (echo cancel hint)
- Speak Connector + TTS  : publish_audio_out (sink)
- GUI Background         : everything (via IOProvider)

---

TODO(REQ-32) [TASK-41]: rclpy / CycloneDDS participant init; load cyclonedds.xml.
TODO(REQ-32) [TASK-41]: subscribe all topics with QoS table above (BestEffort vs Reliable).
TODO(REQ-32) [TASK-41]: TopicCache update on every callback (value + monotonic ts).
TODO(REQ-32) [TASK-41]: stale detection helpers (`stale(now, ttl_s)`).
TODO(REQ-33) [TASK-41]: publish methods for /bridge/cmd/{arm, low, loco, audio_out}.
TODO(REQ-33) [TASK-41]: comm_bridge_alive() heartbeat — block publishes if stale > N ms.
TODO(REQ-32) [TASK-41]: reconnect strategy on LAN drop / Zenoh bridge restart.
TODO(REQ-32) [TASK-41]: unit/integration tests under tests/providers/.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from .singleton import singleton


@dataclass
class TopicCache:
    """Last-known value + monotonic timestamp for a single subscribed topic.

    A fresh TopicCache (``last_seen_ts == 0.0``) is treated as stale by
    :meth:`stale` so downstream code can guard against "never received yet".
    """

    value: Optional[Any] = None
    last_seen_ts: float = 0.0  # monotonic seconds; 0.0 == never received

    def stale(self, now: float, ttl_s: float) -> bool:
        """True if no message received within ``ttl_s`` of ``now``."""
        return (self.last_seen_ts == 0.0) or ((now - self.last_seen_ts) > ttl_s)


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
    ):
        """
        Parameters
        ----------
        ros_domain_id : int
            ROS_DOMAIN_ID shared with the G1 onboard via comm_bridge.
        cyclonedds_uri : str, optional
            File URI to ``cyclonedds.xml`` (partition filter).
        comm_bridge_host : str, optional
            IP/hostname of the Orin NX running comm_bridge (logging/diagnostics).
        heartbeat_timeout_ms : int
            Max age of last comm_bridge heartbeat before outbound cmds are blocked.
        sensor_ttl_ms : int
            Sensor stream stale threshold (BestEffort topics).
        state_ttl_ms : int
            State stream stale threshold (Reliable topics like estop/buf_state).
        """
        self._ros_domain_id = ros_domain_id
        self._cyclonedds_uri = cyclonedds_uri
        self._comm_bridge_host = comm_bridge_host
        self._heartbeat_timeout_ms = heartbeat_timeout_ms
        self._sensor_ttl_ms = sensor_ttl_ms
        self._state_ttl_ms = state_ttl_ms

        self._connected = False

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
        # TODO(REQ-32) [TASK-41]: rclpy.init + CYCLONEDDS_URI export
        # TODO(REQ-32) [TASK-41]: bind subscribers with correct QoS
        # TODO(REQ-33) [TASK-41]: bind publishers
        # TODO(REQ-32) [TASK-41]: spin executor in a worker thread
        raise NotImplementedError("UnitreeG1Provider.start: TBD [TASK-41]")

    def stop(self) -> None:
        """Tear down DDS participant cleanly."""
        # TODO(REQ-32) [TASK-41]: cancel pubs/subs, shutdown rclpy
        raise NotImplementedError("UnitreeG1Provider.stop: TBD [TASK-41]")

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
        """Latest ``/bridge/motor/buf_state`` (motor ring buffer telemetry)."""
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
    # Health / stale helpers
    # ------------------------------------------------------------------
    def comm_bridge_alive(self) -> bool:
        """True if any ``/bridge/*`` topic was seen within ``heartbeat_timeout_ms``."""
        # TODO(REQ-33) [TASK-41]: aggregate latest timestamps across caches
        raise NotImplementedError("UnitreeG1Provider.comm_bridge_alive: TBD [TASK-41]")

    # ------------------------------------------------------------------
    # Publishers (cmd outbound, Reliable)
    # ------------------------------------------------------------------
    def publish_joint_cmd_arm(self, joint_cmd: dict) -> None:
        """
        Publish a Joint Cmd Upper Body (rt/arm_sdk path) to ``/bridge/cmd/arm``.

        ICD IF-6. Fields: ``joint_names[]``, ``q[]``, ``dq[]``, ``kp[]``,
        ``kd[]``, ``tau_ff[]``, ``mode``, ``weight``, ``chunk_id``, ``step_index``.
        Weight respected on NX side (motor_cmd[29].q ramp).
        """
        # TODO(REQ-33) [TASK-41]: schema validate, comm_bridge_alive() watchdog, publish
        raise NotImplementedError("UnitreeG1Provider.publish_joint_cmd_arm: TBD [TASK-41]")

    def publish_joint_cmd_low(self, joint_cmd: dict) -> None:
        """
        Publish a Joint Cmd Lower Body (rt/lowcmd path) to ``/bridge/cmd/low``.

        NEW 2026-05-22 (whole-body VLA walking). Weight ignored on NX side.
        """
        # TODO(REQ-33) [TASK-41]: schema validate, watchdog, publish
        raise NotImplementedError("UnitreeG1Provider.publish_joint_cmd_low: TBD [TASK-41]")

    def publish_loco_cmd(self, loco_command: dict) -> None:
        """
        Publish a LocoCommand to ``/bridge/cmd/loco``.

        Usage scope TBD (likely demo entry/exit — StandUp / BalanceStand /
        SitDown / Damp — plus posture transitions; not finalized per 2026-05-22).
        """
        # TODO(REQ-33) [TASK-41]: schema validate, publish
        raise NotImplementedError("UnitreeG1Provider.publish_loco_cmd: TBD [TASK-41]")

    def publish_audio_out(self, pcm: bytes) -> None:
        """
        Publish synthesized PCM audio to ``/bridge/cmd/audio_out`` (onboard speaker).
        TTS Provider → Speak Connector → here.
        """
        # TODO(REQ-29) [TASK-41]: chunking + flow control + watchdog
        raise NotImplementedError("UnitreeG1Provider.publish_audio_out: TBD [TASK-41]")
