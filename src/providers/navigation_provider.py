"""
NavigationProvider [TASK-48, REQ "PC NavigationProvider"]

PC-side container. Translates Nav sub-tasks to walking velocity (vx/vy/vyaw)
via Unitree SDK loco_client. Internal Kalman Filter for UWB↔IMU calibration.

Pipeline:
    Move Connector → submit_nav_subtask(prompt)
        ↓
    NavigationProvider (internal Kalman + planner)
        ↓
    UnitreeG1Provider.publish_twist(vx, vy, vyaw)
        ↓ DDS /bridge/cmd/vel
    NX comm_bridge → motor_controller → LocoClient.Move

Status: scaffold. Full implementation 5/27~5/29 sensor HW test 이후 6/1~6/5
nav HW test.

Background:
- 2026-05-22 KIST mail had walking subsumed into low-level VLA whole-body
  (no separate path planner anywhere). 2026-05-26 KIST 회의 reversed that —
  VLA-locomotion 아키텍처 / GearSonic 배치 모두 미확정으로 시연 일정 충족
  불가. Locomotion was peeled back out into this NavigationProvider so VLA
  scope can shrink to arm/hand manipulation (CONV-005 범위 축소, CONV-012).
- Velocity is dispatched via the Unitree SDK `loco_client.Move(vx, vy, vyaw)`
  on NX, not via VLA joint chunks. Posture transitions
  (StandUp / SitDown / Damp / BalanceStand) stay on the discrete
  ``LocoCommand`` channel — see ``UnitreeG1Provider.publish_loco_cmd``.

TODO(REQ "PC NavigationProvider") [TASK-48]: implement Kalman filter
    (UWB pose ↔ IMU base/ankle integration) for robust pose estimate
    even when UWB drops.
TODO(REQ "PC NavigationProvider") [TASK-48]: implement planner — turn a
    sub-task prompt ("go to fridge") + current pose into a target waypoint
    (or a sequence of waypoints) and a velocity profile.
TODO(REQ "PC NavigationProvider") [TASK-48]: implement control loop —
    fixed ``control_rate_hz`` thread that consumes current pose, computes
    (vx, vy, vyaw), and calls ``UnitreeG1Provider.publish_twist``.
TODO(REQ "PC NavigationProvider") [TASK-48]: success / failure reporting
    surface for ``TaskSrvProvider`` — currently sub-task success is still
    polled on ``UnitreeG1Provider.uwb_pose`` via ``UwbPoseSuccess``; revisit
    whether NavigationProvider should also expose its own state so the
    poller can short-circuit if the planner has already failed.
TODO(REQ "PC NavigationProvider") [TASK-48]: E-STOP — register an estop
    callback via ``UnitreeG1Provider.register_estop_callback``. On
    active=True: stop the control loop and publish a zero Twist
    (vx=vy=vyaw=0) so NX sees an immediate decel intent in addition to
    the safety_monitor E-STOP path.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .singleton import singleton
from .unitree_g1_provider import UnitreeG1Provider


@dataclass
class NavigationProviderConfig:
    """NavigationProvider runtime configuration."""

    # Control-loop rate at which (vx, vy, vyaw) is recomputed and published
    # via ``UnitreeG1Provider.publish_twist``. TBD — 5/26 KIST 회의에서는
    # 10-50 Hz 추정. NX motor_controller's ``LocoClient.Move`` dispatch loop
    # operates at 100 Hz (REQ-34 v2026-05-22) so anything above ~50 Hz on
    # PC is wasted (NX zero-order-holds between updates).
    control_rate_hz: float = 10.0
    # Inbound UWB pose topic on the PC side (subscribed via
    # ``UnitreeG1Provider.uwb_pose`` — this field is informational, the
    # actual subscription lives in UnitreeG1Provider). TBD — final UWB
    # driver topic name is locked after 5/27~5/29 sensor HW test.
    uwb_topic_in: str = "/uwb/raw"


@singleton
class NavigationProvider:
    """
    PC-side navigation container (REQ "PC NavigationProvider", CONV-012).

    Consumes Nav sub-task prompts from ``MoveConnector``, runs an internal
    Kalman filter + planner, and emits continuous walking velocity via
    ``UnitreeG1Provider.publish_twist(vx, vy, vyaw)``. The NX
    ``motor_controller`` translates the published Twist into
    ``LocoClient.Move(vx, vy, vyaw)`` calls.

    Lifecycle (CONV-001): instantiated and ``.start()``-ed by ``run.py``;
    dependencies are fetched via ``@singleton`` (CONV-010). No ``bind()``
    needed — UnitreeG1Provider is the only Provider dep.

    Scaffold status: ``start()`` / ``stop()`` / ``submit_nav_subtask()`` all
    raise NotImplementedError until 6/1~6/5 nav HW test. Use
    ``run.py --scaffold-loop`` to skip past the NotImplementedError in
    boot so the rest of the runtime can still come up for sensor work.
    """

    def __init__(self, config: Optional[NavigationProviderConfig] = None):
        self._config = config or NavigationProviderConfig()
        # CONV-010: UnitreeG1Provider is @singleton — fetched here.
        self._unitree_g1 = UnitreeG1Provider()
        self._running = False
        logging.info(
            "NavigationProvider: skeleton initialized "
            "(control_rate=%.1fHz, uwb_topic_in=%s)",
            self._config.control_rate_hz, self._config.uwb_topic_in,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the Kalman filter + planner + control loop."""
        # TODO(REQ "PC NavigationProvider") [TASK-48]: spin up Kalman +
        # planner state, register estop callback, start the
        # control_rate_hz thread that emits Twist via
        # UnitreeG1Provider.publish_twist.
        raise NotImplementedError(
            "NavigationProvider.start: TBD [TASK-48]"
        )

    def stop(self) -> None:
        """Stop the control loop and zero the published Twist."""
        # TODO(REQ "PC NavigationProvider") [TASK-48]: join the control
        # thread, unregister estop callback, publish a zero Twist on
        # shutdown so NX sees an explicit "no velocity" command.
        raise NotImplementedError(
            "NavigationProvider.stop: TBD [TASK-48]"
        )

    # ------------------------------------------------------------------
    # Sub-task dispatch (from MoveConnector)
    # ------------------------------------------------------------------
    def submit_nav_subtask(self, prompt: str) -> None:
        """
        Accept a navigation sub-task prompt (free-form text) from MoveConnector.

        The planner converts ``prompt`` into a target waypoint (or sequence)
        and arms the control loop to drive there. This is a fire-and-forget
        call — completion is reported back to ``TaskSrvProvider`` via the
        UWB-pose success criterion (``UwbPoseSuccess`` on
        ``UnitreeG1Provider.uwb_pose``); we do not block on success here.

        Threading: called from the MoveConnector's async ``connect()`` which
        runs on the asyncio loop owned by ``TaskSrvProvider._schedule_coro``.
        Implementation MUST be non-blocking (enqueue + return) so the
        asyncio task does not stall.
        """
        # TODO(REQ "PC NavigationProvider") [TASK-48]: parse prompt → waypoint
        # (initial demo: small keyword table; planner stays out-of-band).
        # TODO(REQ "PC NavigationProvider") [TASK-48]: arm the control loop;
        # do NOT block.
        raise NotImplementedError(
            "NavigationProvider.submit_nav_subtask: TBD [TASK-48]"
        )
