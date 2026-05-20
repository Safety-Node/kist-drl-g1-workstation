"""
UnitreeG1 Provider -- KIST DRL G1 Workstation
=============================================

drawio C4 Container:
    Name        : UnitreeG1 Provider
    Technology  : LAN / Python
    Description : Provider Interface for Unitree G1.

This is the workstation-side facade across all G1 onboard interactions
(sensors, comm_bridge, motor_controller, safety_monitor, navigation).
Underneath it composes the more specific providers already present:
    - unitree_g1_odom_provider.UnitreeG1OdomProvider
    - unitree_g1_locations_provider.UnitreeG1LocationsProvider
    - unitree_g1_navigation_provider.UnitreeG1NavigationProvider
    - unitree_camera_vlm_provider.UnitreeCameraVLMProvider

Edges (from drawio):
    <-> G1 Onboard.comm_bridge :
        Image, Audio Data + Speaker/Joint/Buf/Nav State    (inbound)
        Joint Cmd + Nav Cmd + Audio Out                    (outbound)
        [ROS 2 Topic / CycloneDDS / Ethernet]

TBD:
    - Topology assertion: ROS2/CycloneDDS domain id, ENABLE_LOOPBACK
    - Heartbeat watchdog with comm_bridge
    - Reconnect strategy on LAN drop
    - Schema check for outbound Joint/Nav commands
    - Safety: drop commands if comm_bridge stale > N ms
"""

import logging
from typing import Optional

from .singleton import singleton


@singleton
class UnitreeG1Provider:
    """
    Workstation-side LAN bridge to the G1 onboard stack.

    Coordinates pub/sub on ROS 2 (CycloneDDS) topics that comm_bridge exposes.
    """

    def __init__(
        self,
        ros_domain_id: int = 0,
        comm_bridge_host: Optional[str] = None,
        heartbeat_timeout_ms: int = 500,
    ):
        """
        Parameters
        ----------
        ros_domain_id : int
            ROS_DOMAIN_ID shared with the G1 onboard.
        comm_bridge_host : str, optional
            IP / hostname of the Orin NX running comm_bridge.
        heartbeat_timeout_ms : int
            Maximum age of last comm_bridge heartbeat before
            outbound commands are blocked (safety).
        """
        # TODO: initialize rclpy / cyclonedds participant
        # TODO: subscribe to onboard topics (joint_state, nav_state, audio, image)
        # TODO: publish topics (joint_cmd, nav_cmd, audio_out)
        self._ros_domain_id = ros_domain_id
        self._comm_bridge_host = comm_bridge_host
        self._heartbeat_timeout_ms = heartbeat_timeout_ms
        self._connected = False
        logging.info(
            "UnitreeG1Provider: skeleton initialized (domain=%d, host=%s)",
            ros_domain_id, comm_bridge_host,
        )

    def start(self) -> None:
        """Bring up DDS participant, subscribers, publishers."""
        # TODO: spin executor in a worker thread
        raise NotImplementedError("UnitreeG1Provider.start: TBD")

    def stop(self) -> None:
        """Tear down DDS participant cleanly."""
        # TODO: cancel pubs/subs, shutdown rclpy
        raise NotImplementedError("UnitreeG1Provider.stop: TBD")

    def publish_joint_cmd(self, joint_cmd_json: dict) -> None:
        """Publish a Joint Cmd payload to G1 onboard."""
        # TODO: schema validate, watchdog check, then publish
        raise NotImplementedError("UnitreeG1Provider.publish_joint_cmd: TBD")

    def publish_nav_cmd(self, nav_cmd: dict) -> None:
        """Publish a Nav Cmd (cmd_vel / goal) to G1 onboard navigation."""
        # TODO: same watchdog + schema constraints
        raise NotImplementedError("UnitreeG1Provider.publish_nav_cmd: TBD")

    def publish_audio_out(self, pcm: bytes) -> None:
        """Publish synthesized PCM audio to the onboard speaker."""
        # TODO: chunking + flow control
        raise NotImplementedError("UnitreeG1Provider.publish_audio_out: TBD")
