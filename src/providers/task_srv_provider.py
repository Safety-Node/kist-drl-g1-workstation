"""
TaskSrv Provider — KIST DRL G1 Workstation [TASK-39]
=====================================================

Scenario-driven sub-task orchestrator that **replaces the LLM Cortex** for
the KIST demo (per CONV-004, KIST mail 2026-05-22, SYS-REQ
``[DEPRECATED 2026-05-24]`` on REQ-28 / REQ-36 / REQ-40).

The provider owns:

1. A loaded **scenario script** (plain Python — see ``config/scenarios/``)
   — ordered list of ``SubTask`` records, each with a natural-language
   prompt for the VLA Provider and a ``SuccessCriterion`` to decide when
   the sub-task is done.
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

## Why Python scenarios (not YAML)

Scenarios are typed Python objects under ``config/scenarios/`` rather than
YAML files. Trade-off accepted (2026-05-24): YAML parser + validator + ad-hoc
schema (~80 LoC) replaced by Python imports + dataclasses (~0 LoC of parsing).
Cost: non-coders cannot edit scenarios — acceptable since the only authors
are the developers themselves. Benefit: full IDE / pyright autocomplete,
static type checking, polymorphic ``SuccessCriterion.evaluate(state)``
instead of string-keyed kind dispatch.

---

## Data flow

    Sound Sensor (STT transcript)                       (TASK-46)
        ↓ on_audio(text, ts)
    TaskSrv Provider                                    (this class)
        ├─ keyword match → load scenario + queue sub-tasks
        ├─ dispatch current sub-task prompt
        │       ↓ Move Connector.connect(MoveInput(action=prompt))
        │   Move Connector → VLA Provider                (TASK-40)
        │
        └─ poll UnitreeG1Provider for success
                ↑ uwb_pose (locomotion sub-tasks)
                ↑ joint_state (manipulation sub-tasks)
    TaskSrv BG (TASK-39 companion plugin)
        ↑ calls TaskSrvProvider.tick() at fixed rate

---

## Lifecycle (per CONV-001 Option D — explicit init in run.py)

    run.py:
        unitree_g1 = UnitreeG1Provider(...); unitree_g1.start()
        vla = VLAProvider(...); vla.start()
        move_conn = MoveConnector(ActionConfig())
        task_srv = TaskSrvProvider(TaskSrvConfig())
        task_srv.bind(unitree_g1=unitree_g1, move_connector=move_conn)
        task_srv.start()                                # loads scenarios

Sound Sensor (also created in run.py) wires its callback to
``task_srv.on_audio``.
"""

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .singleton import singleton


# ---------------------------------------------------------------------------
# Success criteria — polymorphic sealed sum
# ---------------------------------------------------------------------------


class SuccessCriterion(ABC):
    """
    Base class for a sub-task success rule.

    Concrete subclasses implement :meth:`evaluate` against the live
    ``UnitreeG1Provider`` state caches. ``timeout_s`` is uniform across
    all criteria and consumed by ``TaskSrvProvider.tick()``.
    """

    timeout_s: float

    @abstractmethod
    def evaluate(self, uwb_cache: Any, joint_cache: Any) -> bool:
        """Return True if the sub-task is complete given the current state."""
        raise NotImplementedError


@dataclass
class UwbPoseSuccess(SuccessCriterion):
    """
    Locomotion sub-task success: robot UWB pose is within L2 ``tolerance_xy``
    of ``target`` ``(x, y)`` and within ``tolerance_yaw`` of ``target_yaw``.

    Note: yaw tolerance is ignored if NX's UWB driver does not produce yaw
    (kp_angular=0 path). Set ``tolerance_yaw`` to a large value or
    ``math.inf`` to disable.
    """

    target: Tuple[float, float, float]   # (x, y, yaw_rad)
    tolerance_xy: float = 0.15
    tolerance_yaw: float = 0.20
    timeout_s: float = 30.0

    def evaluate(self, uwb_cache: Any, joint_cache: Any) -> bool:
        """L2 xy + abs yaw vs target, both within tolerances."""
        pose = getattr(uwb_cache, "value", None)
        if pose is None:
            return False
        x = _get(pose, "x")
        y = _get(pose, "y")
        yaw = _get(pose, "yaw")
        if x is None or y is None:
            return False
        tx, ty, tyaw = self.target
        dxy = math.hypot(x - tx, y - ty)
        if dxy > self.tolerance_xy:
            return False
        if yaw is not None and not math.isinf(self.tolerance_yaw):
            if abs(_wrap_pi(yaw - tyaw)) > self.tolerance_yaw:
                return False
        return True


@dataclass
class JointStateSuccess(SuccessCriterion):
    """
    Manipulation sub-task success: a single joint reaches a target position.

    For multi-joint criteria, compose via :class:`CompositeSuccess`.
    """

    target_joint: str
    target_pos: float
    tolerance: float = 0.05
    timeout_s: float = 10.0

    def evaluate(self, uwb_cache: Any, joint_cache: Any) -> bool:
        """|actual - target| within tolerance."""
        state = getattr(joint_cache, "value", None)
        if state is None:
            return False
        actual = _joint_pos(state, self.target_joint)
        if actual is None:
            return False
        return abs(actual - self.target_pos) <= self.tolerance


@dataclass
class CompositeSuccess(SuccessCriterion):
    """All-of: succeeds when every child criterion evaluates True."""

    children: List[SuccessCriterion] = field(default_factory=list)
    timeout_s: float = 30.0

    def evaluate(self, uwb_cache: Any, joint_cache: Any) -> bool:
        return all(c.evaluate(uwb_cache, joint_cache) for c in self.children)


# ---------------------------------------------------------------------------
# Sub-task / scenario value objects
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """One sub-task: a natural-language prompt + success criterion."""

    prompt: str
    success: SuccessCriterion


@dataclass
class Scenario:
    """One loaded scenario script."""

    name: str
    triggers: List[str]              # STT keyword phrases (substring match)
    sub_tasks: List[SubTask]


# ---------------------------------------------------------------------------
# Provider state machine
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    """High-level state of the orchestrator."""

    IDLE = "idle"
    ACTIVE = "active"
    SUCCESS = "success"   # transient — flips back to IDLE on next tick
    FAILED = "failed"     # transient — flips back to IDLE on next tick


@dataclass
class TaskSrvConfig:
    """TaskSrv Provider runtime configuration."""

    tick_rate_hz: float = 10.0
    case_insensitive_keywords: bool = True
    enable_speak_feedback: bool = True
    success_phrase: str = "성공했습니다."
    failure_phrase: str = "실패했습니다."


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
        self._trigger_index: Dict[str, Scenario] = {}
        self._state: TaskState = TaskState.IDLE
        self._active_scenario: Optional[Scenario] = None
        self._active_sub_task_idx: int = -1
        self._sub_task_dispatch_ts: float = 0.0
        self._unitree_g1 = None
        self._move_connector = None
        self._speak_connector = None
        self._running = False
        logging.info(
            "TaskSrvProvider: skeleton initialized (tick=%.1fHz)",
            self._config.tick_rate_hz,
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
        """Wire dependencies from run.py before ``start()`` is called."""
        self._unitree_g1 = unitree_g1
        self._move_connector = move_connector
        self._speak_connector = speak_connector

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, scenarios: Optional[List[Scenario]] = None) -> None:
        """
        Load scenario scripts and build the trigger index.

        ``scenarios`` lets tests inject a list directly. When None, scenarios
        are imported from ``config.scenarios.ALL`` (lazy import to avoid a
        hard dependency at module load time).
        """
        if scenarios is None:
            from config.scenarios import ALL as _ALL
            scenarios = list(_ALL)
        self._scenarios = scenarios
        self._trigger_index = self._build_trigger_index(scenarios)
        self._running = True
        logging.info(
            "TaskSrvProvider: loaded %d scenarios (%d triggers)",
            len(self._scenarios), len(self._trigger_index),
        )

    def stop(self) -> None:
        """Abort the active scenario (if any), reset to IDLE."""
        if self._state == TaskState.ACTIVE:
            logging.info(
                "TaskSrvProvider: stop() aborts active scenario '%s' at sub-task %d",
                self._active_scenario.name if self._active_scenario else "?",
                self._active_sub_task_idx,
            )
        self._reset_active()
        self._state = TaskState.IDLE
        self._running = False

    # ------------------------------------------------------------------
    # Inbound: STT keyword router (from Sound Sensor)
    # ------------------------------------------------------------------
    def on_audio(self, text: str, ts: Optional[float] = None) -> None:
        """
        Match the transcript against scenario triggers. On hit, transition
        IDLE → ACTIVE and dispatch the first sub-task. If already ACTIVE,
        the transcript is ignored (no preemption in scaffold).
        """
        if not self._running:
            logging.debug("TaskSrvProvider.on_audio: not running, ignoring")
            return
        if self._state == TaskState.ACTIVE:
            logging.debug(
                "TaskSrvProvider.on_audio: scenario active, ignoring '%s'", text
            )
            return
        matched = self._match_trigger(text)
        if matched is None:
            return
        logging.info(
            "TaskSrvProvider: trigger '%s' matched scenario '%s'",
            text, matched.name,
        )
        self._activate(matched)

    # ------------------------------------------------------------------
    # Periodic tick (driven by TaskSrvBg)
    # ------------------------------------------------------------------
    def tick(self) -> None:
        """
        One success-evaluation cycle.

        IDLE / SUCCESS / FAILED states are flipped back to IDLE here so the
        transient terminal states are observable for exactly one tick by
        the GUI poller, then cleared.
        """
        if self._state in (TaskState.SUCCESS, TaskState.FAILED):
            self._reset_active()
            self._state = TaskState.IDLE
            return
        if self._state != TaskState.ACTIVE:
            return
        sub_task = self.active_sub_task
        if sub_task is None:
            # Shouldn't happen in ACTIVE state, but guard.
            self._finish_failure(reason="no active sub-task")
            return
        # Success check
        uwb_cache = getattr(self._unitree_g1, "uwb_pose", None)
        joint_cache = getattr(self._unitree_g1, "joint_state", None)
        try:
            done = sub_task.success.evaluate(uwb_cache, joint_cache)
        except Exception:
            logging.exception("TaskSrvProvider.tick: criterion.evaluate raised")
            done = False
        if done:
            self._advance_sub_task()
            return
        # Timeout check
        elapsed = time.monotonic() - self._sub_task_dispatch_ts
        if elapsed > sub_task.success.timeout_s:
            self._finish_failure(reason=f"sub-task timeout ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Read-only state (polled by GUI BG — no push callback)
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

    @property
    def active_scenario_name(self) -> Optional[str]:
        """Name of the currently running scenario, or None when IDLE."""
        return self._active_scenario.name if self._active_scenario else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_trigger_index(
        self, scenarios: List[Scenario]
    ) -> Dict[str, Scenario]:
        index: Dict[str, Scenario] = {}
        for sc in scenarios:
            for trig in sc.triggers:
                key = trig.lower() if self._config.case_insensitive_keywords else trig
                if key in index and index[key].name != sc.name:
                    raise ValueError(
                        f"Duplicate trigger '{trig}' in scenarios "
                        f"'{index[key].name}' and '{sc.name}'"
                    )
                index[key] = sc
        return index

    def _match_trigger(self, text: str) -> Optional[Scenario]:
        haystack = text.lower() if self._config.case_insensitive_keywords else text
        # Substring match; first hit wins.
        for needle, scenario in self._trigger_index.items():
            if needle in haystack:
                return scenario
        return None

    def _activate(self, scenario: Scenario) -> None:
        self._active_scenario = scenario
        self._active_sub_task_idx = 0
        self._state = TaskState.ACTIVE
        self._dispatch_current()

    def _dispatch_current(self) -> None:
        sub_task = self.active_sub_task
        if sub_task is None:
            self._finish_failure(reason="dispatch with no active sub-task")
            return
        self._sub_task_dispatch_ts = time.monotonic()
        if self._move_connector is None:
            logging.warning(
                "TaskSrvProvider: no MoveConnector bound; would dispatch '%s'",
                sub_task.prompt,
            )
            return
        # OM1 canonical action API: ActionConnector.connect(MoveInput(action=...))
        # is async. MoveInput.action is typed MovementAction enum but the OM1
        # ros2 connector matches it as a string — we pass the VLA prompt as
        # the action string. MoveConnector implementation will adapt to the
        # whole-body VLA path (TASK-40); this just keeps the call site stable.
        from actions.move.interface import MoveInput  # type: ignore

        _schedule_coro(self._move_connector.connect(MoveInput(action=sub_task.prompt)))
        logging.info(
            "TaskSrvProvider: dispatched sub-task[%d] '%s'",
            self._active_sub_task_idx, sub_task.prompt,
        )

    def _advance_sub_task(self) -> None:
        assert self._active_scenario is not None
        self._active_sub_task_idx += 1
        if self._active_sub_task_idx >= len(self._active_scenario.sub_tasks):
            self._finish_success()
            return
        self._dispatch_current()

    def _finish_success(self) -> None:
        assert self._active_scenario is not None
        logging.info(
            "TaskSrvProvider: scenario '%s' completed successfully",
            self._active_scenario.name,
        )
        self._speak(self._config.success_phrase)
        self._state = TaskState.SUCCESS
        # Active fields retained until next tick() so GUI can read them once.

    def _finish_failure(self, reason: str) -> None:
        scenario_name = (
            self._active_scenario.name if self._active_scenario else "?"
        )
        logging.warning(
            "TaskSrvProvider: scenario '%s' failed at sub-task %d (%s)",
            scenario_name, self._active_sub_task_idx, reason,
        )
        self._speak(self._config.failure_phrase)
        self._state = TaskState.FAILED

    def _speak(self, phrase: str) -> None:
        if not self._config.enable_speak_feedback:
            return
        if self._speak_connector is None:
            logging.info("TaskSrvProvider: [speak] %s", phrase)
            return
        from actions.speak.interface import SpeakInput  # type: ignore

        _schedule_coro(self._speak_connector.connect(SpeakInput(action=phrase)))

    def _reset_active(self) -> None:
        self._active_scenario = None
        self._active_sub_task_idx = -1
        self._sub_task_dispatch_ts = 0.0


# ---------------------------------------------------------------------------
# Helpers — pose / joint state access that tolerates dict / dataclass / msg
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str) -> Optional[float]:
    """Fetch ``obj.name`` or ``obj[name]``; return None if absent."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        v = obj.get(name)
    else:
        v = getattr(obj, name, None)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _joint_pos(state: Any, joint_name: str) -> Optional[float]:
    """
    Extract a single joint position from a JointState-like value.

    Accepts:
        - sensor_msgs.msg.JointState (.name list, .position list)
        - dict {"name": [...], "position": [...]}
        - dict {"joint_a": pos_a, ...}
    """
    if state is None:
        return None
    # Mapping-of-name form
    if isinstance(state, dict) and joint_name in state and not isinstance(
        state.get("name"), (list, tuple)
    ):
        try:
            return float(state[joint_name])
        except (TypeError, ValueError):
            return None
    names = state.get("name") if isinstance(state, dict) else getattr(state, "name", None)
    positions = (
        state.get("position") if isinstance(state, dict) else getattr(state, "position", None)
    )
    if names is None or positions is None:
        return None
    try:
        idx = list(names).index(joint_name)
    except ValueError:
        return None
    try:
        return float(positions[idx])
    except (IndexError, TypeError, ValueError):
        return None


def _wrap_pi(angle: float) -> float:
    """Wrap ``angle`` to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _schedule_coro(coro: Any) -> None:
    """
    Schedule an awaitable on the running loop, or run it synchronously
    if there is no loop (the typical synchronous-test case).

    This is the only bridge between the sync ``TaskSrvProvider`` API
    surface and the async OM1 ``ActionConnector.connect()`` contract.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)
