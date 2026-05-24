"""
TaskSrv Provider — KIST DRL G1 Workstation [TASK-39]
=====================================================

Scenario-driven sub-task orchestrator that **replaces the LLM Cortex** for
the KIST demo (per CONV-004, KIST mail 2026-05-22, SYS-REQ
``[DEPRECATED 2026-05-24]`` on REQ-28 / REQ-36 / REQ-40).

The provider owns:

1. A loaded **scenario script** (YAML or Python) — ordered list of
   ``SubTask`` records, each with a natural-language prompt for the VLA
   Provider and a ``SuccessCriterion`` to decide when the sub-task is done.
2. A **keyword router** — fed by Sound Sensor's ``on_audio(text, ts)``
   callback. A matched keyword triggers the corresponding scenario.
3. The **dispatch path** to Move Connector (which formats the prompt into
   a VLA Provider call).
4. A **success-detection** poll — reads ``UnitreeG1Provider.uwb_pose`` and
   ``.joint_state`` (CONV-005 whole-body VLA: pose for locomotion sub-tasks,
   joint pos for manipulation sub-tasks; mix is per-sub-task choice).

The polling itself runs on ``TaskSrvBg`` (separate plugin) — this provider
only exposes ``tick()``; the BG calls it at a fixed rate so the provider
stays single-threaded and easy to test.

drawio C4 Container:
    Name        : TaskSrv Provider
    Technology  : Python
    Description : Scenario-driven sub-task orchestrator (replaces LLM Cortex).

---

## Data flow

    Sound Sensor (STT transcript)                       (TASK-46)
        ↓ on_audio(text, ts)
    TaskSrv Provider                                    (this class)
        ├─ keyword match → load scenario + queue sub-tasks
        ├─ dispatch current sub-task prompt
        │       ↓ Move Connector.send(prompt)
        │   Move Connector → VLA Provider                (TASK-???)
        │
        └─ poll UnitreeG1Provider for success
                ↑ uwb_pose (locomotion sub-tasks)
                ↑ joint_state (manipulation sub-tasks)
    TaskSrv BG (TASK-39 companion plugin)
        ↑ calls TaskSrvProvider.tick() at fixed rate

---

## Lifecycle (per CONV-001 Option D — explicit init in run.py)

    run.py:
        unitree_g1 = UnitreeG1Provider(...).start()
        vla = VLAProvider(...).start()                  # TBD
        move_conn = MoveConnector(...).bind(vla=vla)    # TBD
        task_srv = TaskSrvProvider(config=...)
        task_srv.bind(unitree_g1=unitree_g1, move_connector=move_conn)
        task_srv.start()                                # loads scenario script
        # TaskSrvBg picks up the singleton via TaskSrvProvider() and calls .tick()

Sound Sensor (also created in run.py) wires its callback to
``task_srv.on_audio``.

---

## Scenario script format (TBD — YAML candidate)

    # scenarios/refrigerator_pickup.yaml
    name: refrigerator_pickup
    triggers:
      - 양상추 꺼내줘
      - 양상추 가져와
    sub_tasks:
      - prompt: "Walk to the refrigerator."
        success:
          kind: uwb_pose
          target: { x: 1.5, y: 0.3, yaw: 0.0 }
          tolerance: { xy: 0.15, yaw: 0.20 }
          timeout_s: 30.0
      - prompt: "Open the refrigerator door."
        success:
          kind: joint_state
          target_joint: "left_gripper_finger_1"
          target_pos: 0.85
          tolerance: 0.05
          timeout_s: 10.0
      - prompt: "Grasp the lettuce."
        success:
          kind: composite
          all:
            - { kind: joint_state, target_joint: "right_gripper_finger_1", target_pos: 0.20, tolerance: 0.03 }
            - { kind: uwb_pose, target: { x: 1.5, y: 0.3, yaw: 0.0 }, tolerance: { xy: 0.10, yaw: 0.15 } }
          timeout_s: 8.0

Final shape is TBD — codifying here so the scaffold's dataclasses match
the YAML schema. See ``SuccessKind`` below.

---

## ICD references

| Direction | Topic / call                              | Note                          |
|-----------|-------------------------------------------|-------------------------------|
| inbound   | Sound Sensor → on_audio(text, ts)         | STT transcript router         |
| inbound   | UnitreeG1Provider.uwb_pose / .joint_state | success polling (TopicCache)  |
| outbound  | MoveConnector.send(prompt)                | sub-task dispatch to VLA      |
| outbound  | optional: IOProvider state for GUI BG     | active scenario / sub-task    |

---

TODO(REQ-NEW-TASKSRV) [TASK-39]: scenario loader — YAML → ``Scenario`` list.
                                  Validate against ``triggers`` uniqueness.
TODO(REQ-NEW-TASKSRV) [TASK-39]: on_audio keyword router — first-match wins;
                                  ignore if a scenario is already active.
TODO(REQ-NEW-TASKSRV) [TASK-39]: dispatch — push current sub-task prompt to
                                  Move Connector; record dispatch ts for timeout.
TODO(REQ-NEW-TASKSRV) [TASK-39]: tick() success poll — read UnitreeG1Provider
                                  TopicCache, evaluate active sub-task criterion.
TODO(REQ-NEW-TASKSRV) [TASK-39]: timeout / abort — surface a TaskFailure event;
                                  emit a Speak prompt ("실패했어요") via Speak Connector.
TODO(REQ-NEW-TASKSRV) [TASK-39]: state machine — IDLE → ACTIVE(scenario, idx)
                                  → SUCCESS / FAILED → IDLE.
TODO(REQ-NEW-TASKSRV) [TASK-39]: tests under tests/providers/test_task_srv_provider.py
                                  (stub UnitreeG1Provider TopicCache + stub MoveConnector).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .singleton import singleton


class SuccessKind(str, Enum):
    """Type of success criterion evaluated against UnitreeG1Provider state."""

    UWB_POSE = "uwb_pose"        # locomotion sub-task (pose tolerance)
    JOINT_STATE = "joint_state"  # manipulation sub-task (joint pos tolerance)
    COMPOSITE = "composite"      # all-of nested criteria
    # MANUAL  = "manual"         # human ack via GUI — future


class TaskState(str, Enum):
    """High-level state of the orchestrator."""

    IDLE = "idle"
    ACTIVE = "active"
    SUCCESS = "success"   # transient — flips back to IDLE on next tick
    FAILED = "failed"     # transient — flips back to IDLE on next tick


@dataclass
class SuccessCriterion:
    """
    One success rule. Exact field set depends on ``kind``; unused fields
    stay at default. Concrete validation lives in the scenario loader
    (TBD), this dataclass is a transport shape.
    """

    kind: SuccessKind
    target: Optional[Dict[str, float]] = None        # UWB_POSE: {x, y, yaw}
    tolerance: Optional[Dict[str, float]] = None     # UWB_POSE: {xy, yaw}
    target_joint: Optional[str] = None               # JOINT_STATE
    target_pos: Optional[float] = None               # JOINT_STATE (rad)
    joint_tolerance: Optional[float] = None          # JOINT_STATE (rad)
    children: List["SuccessCriterion"] = field(default_factory=list)  # COMPOSITE
    timeout_s: float = 30.0


@dataclass
class SubTask:
    """One sub-task: a natural-language prompt + success criterion."""

    prompt: str
    success: SuccessCriterion


@dataclass
class Scenario:
    """One loaded scenario script."""

    name: str
    triggers: List[str]              # STT keyword phrases
    sub_tasks: List[SubTask]


@dataclass
class TaskSrvConfig:
    """TaskSrv Provider runtime configuration."""

    scenarios_dir: str = "scenarios/"   # YAML files live here (TBD)
    tick_rate_hz: float = 10.0          # success-poll rate; TaskSrvBg target
    case_insensitive_keywords: bool = True
    # Speak Connector hook for spoken status (e.g. "실패했어요"). Optional.
    enable_speak_feedback: bool = True


@singleton
class TaskSrvProvider:
    """
    Scenario-driven sub-task orchestrator (replaces LLM Cortex).

    Lifecycle (CONV-001 Option D): instantiated and ``.start()``-ed by
    ``run.py``; dependencies are explicitly bound via ``bind(...)`` before
    ``start()``. ``TaskSrvBg`` resolves the singleton via plain
    ``TaskSrvProvider()`` and drives the success-poll loop.
    """

    def __init__(self, config: Optional[TaskSrvConfig] = None):
        self._config = config or TaskSrvConfig()
        self._scenarios: List[Scenario] = []
        self._trigger_index: Dict[str, Scenario] = {}   # built in start()
        self._state: TaskState = TaskState.IDLE
        self._active_scenario: Optional[Scenario] = None
        self._active_sub_task_idx: int = -1
        self._sub_task_dispatch_ts: float = 0.0
        self._unitree_g1 = None       # bound by bind()
        self._move_connector = None   # bound by bind()
        self._speak_connector = None  # bound by bind() (optional)
        self._on_state_change: Optional[Callable[[TaskState], None]] = None
        self._running = False
        logging.info(
            "TaskSrvProvider: skeleton initialized (tick=%.1fHz, scenarios_dir=%s)",
            self._config.tick_rate_hz,
            self._config.scenarios_dir,
        )

    # ------------------------------------------------------------------
    # Explicit dependency wiring (CONV-001 Option D)
    # ------------------------------------------------------------------
    def bind(
        self,
        unitree_g1: Any,
        move_connector: Any,
        speak_connector: Optional[Any] = None,
    ) -> None:
        """
        Wire dependencies from run.py.

        Parameters
        ----------
        unitree_g1 : UnitreeG1Provider
            Already-started singleton; read TopicCaches for success polling.
        move_connector : MoveConnector
            Already-bound connector; ``move_connector.send(prompt)`` pushes
            a sub-task to the VLA Provider. (Exact API TBD on MoveConnector side.)
        speak_connector : SpeakConnector, optional
            Optional spoken feedback channel ("성공!"/"실패했어요"). If
            ``enable_speak_feedback`` is True but this is None, feedback is
            logged only.
        """
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: assert all are started
        self._unitree_g1 = unitree_g1
        self._move_connector = move_connector
        self._speak_connector = speak_connector

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Load scenario scripts from disk, build the trigger index."""
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: scan scenarios_dir, parse YAML → Scenario list
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: build self._trigger_index (lower-case if configured)
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: self._running = True
        raise NotImplementedError("TaskSrvProvider.start: TBD [TASK-39]")

    def stop(self) -> None:
        """Cancel any active scenario, clear state."""
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: graceful abort of active sub-task
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: reset to IDLE, self._running = False
        raise NotImplementedError("TaskSrvProvider.stop: TBD [TASK-39]")

    # ------------------------------------------------------------------
    # Inbound: STT keyword router (from Sound Sensor)
    # ------------------------------------------------------------------
    def on_audio(self, text: str, ts: Optional[float] = None) -> None:
        """
        Match the transcript against scenario triggers; on hit, transition
        IDLE → ACTIVE and dispatch the first sub-task. If already ACTIVE,
        the transcript is ignored (no preemption in scaffold).
        """
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: normalise text (strip / lowercase if configured)
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: lookup self._trigger_index; ignore if no hit
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: if self._state != IDLE: log + return
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: load scenario, set ACTIVE, _dispatch_current()
        raise NotImplementedError("TaskSrvProvider.on_audio: TBD [TASK-39]")

    # ------------------------------------------------------------------
    # Periodic tick (driven by TaskSrvBg)
    # ------------------------------------------------------------------
    def tick(self) -> None:
        """
        One success-evaluation cycle. If ACTIVE: evaluate current sub-task
        criterion against UnitreeG1Provider state; advance / fail / timeout.
        If IDLE: no-op.
        """
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: if self._state != ACTIVE: return
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: read unitree_g1.uwb_pose / .joint_state caches
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: evaluate _evaluate_criterion(...) → True/False
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: on True: _advance_sub_task() or _finish_success()
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: on timeout: _finish_failure()
        raise NotImplementedError("TaskSrvProvider.tick: TBD [TASK-39]")

    # ------------------------------------------------------------------
    # Read-only state (for GUI BG)
    # ------------------------------------------------------------------
    @property
    def state(self) -> TaskState:
        """Current orchestrator state. Read-safe from any thread."""
        return self._state

    @property
    def active_sub_task(self) -> Optional[SubTask]:
        """Currently dispatched sub-task, or None when IDLE."""
        if self._active_scenario is None or self._active_sub_task_idx < 0:
            return None
        if self._active_sub_task_idx >= len(self._active_scenario.sub_tasks):
            return None
        return self._active_scenario.sub_tasks[self._active_sub_task_idx]

    def register_state_callback(
        self, callback: Callable[[TaskState], None]
    ) -> None:
        """Register a callback fired on every state transition (GUI hook)."""
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: invoke from _set_state() on each transition
        self._on_state_change = callback

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _dispatch_current(self) -> None:
        """Push current sub-task's prompt to Move Connector, record ts."""
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: self._move_connector.send(sub_task.prompt)
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: self._sub_task_dispatch_ts = time.monotonic()
        raise NotImplementedError("TaskSrvProvider._dispatch_current: TBD [TASK-39]")

    def _evaluate_criterion(
        self,
        criterion: SuccessCriterion,
        uwb_cache: Any,
        joint_cache: Any,
    ) -> bool:
        """Pure-function criterion evaluator (no side effects)."""
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: dispatch on criterion.kind
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: UWB_POSE: L2 xy + abs yaw within tolerance
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: JOINT_STATE: |actual - target| <= joint_tolerance
        # TODO(REQ-NEW-TASKSRV) [TASK-39]: COMPOSITE: all(children evaluated recursively)
        raise NotImplementedError("TaskSrvProvider._evaluate_criterion: TBD [TASK-39]")
