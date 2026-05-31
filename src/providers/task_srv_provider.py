"""
TaskSrvProvider [REQ-44, TASK-39] — hook-driven scenario orchestrator.

Replaces the LLM Cortex for the KIST demo (CONV-004). A scenario is a list of
sub-tasks; each sub-task is a small lifecycle machine:

    on_create → on_start → (poll `success` each tick) → on_success | on_fail

Each hook is an ordered list of Actions (Speak / Move / ...). A connector call
is just an Action, so it is freely included or omitted per hook. Success is a
polymorphic Criterion that reads sensors (UWB / joint) AND voice transcripts,
which lets a spoken command complete a sub-task. Scenarios are JSON5 data under
``config/scenarios/*.json5`` (see CONV-013); the types they reference live in
this file.

Loop ownership: ``run(stop_event)`` owns the asyncio loop — it pumps ``tick()``
at ``tick_rate_hz`` and runs hook coroutines on the same loop. ``TaskSrvBg`` is
just the thread host that calls ``run()``.

Threading: ``on_audio`` is called from the STT thread and only enqueues
(thread-safe). All state mutation happens on the loop thread (``tick`` drains
the queue; hooks are loop tasks). One writer → no locks.

This file is intentionally one module (CONV-013): with no pytest infra
(CONV-009) the modularity payoff is nil, and one-provider-one-file matches the
rest of the codebase. Sections: resolve · context · criteria · actions ·
scenario model · loader · engine.
"""

from __future__ import annotations

import asyncio
import logging
import math
import queue
import re
import time
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import json5

from actions.move.interface import MoveInput
from actions.speak.interface import SpeakInput
from providers.singleton import singleton
from providers.unitree_g1_provider import UnitreeG1Provider


# ===========================================================================
# Template / reference resolution — replaces lambdas (JSON5 can't hold them).
#   resolve_str("{dest_ko}로 이동", bb)  -> string substitution
#   resolve_value("{dest.target}", bb)   -> the actual value (list/tuple/...)
# Dotted paths index nested dicts: {dest.target} -> bb["dest"]["target"].
# ===========================================================================

_REF_RE = re.compile(r"\{([^}]+)\}")
_WHOLE_REF_RE = re.compile(r"^\{([^}]+)\}$")


def _lookup(blackboard: T.Mapping, path: str) -> T.Any:
    cur: T.Any = blackboard
    for part in path.split("."):
        part = part.strip()
        cur = cur.get(part) if isinstance(cur, T.Mapping) else getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def resolve_str(template: T.Any, blackboard: T.Mapping) -> T.Any:
    """Substitute every ``{ref}`` in a string; non-strings pass through; missing → ''."""
    if not isinstance(template, str):
        return template
    return _REF_RE.sub(lambda m: "" if (v := _lookup(blackboard, m.group(1))) is None else str(v), template)


def resolve_value(value: T.Any, blackboard: T.Mapping) -> T.Any:
    """Resolve a param: whole-string ref → actual value; 'inf'/-inf → math.inf; else literal/template."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == "inf":
        return math.inf
    if s == "-inf":
        return -math.inf
    whole = _WHOLE_REF_RE.match(s)
    return _lookup(blackboard, whole.group(1)) if whole else resolve_str(s, blackboard)


def _as_float(value: T.Any) -> float:
    """float() that also accepts the 'inf' / '-inf' sentinels (JSON5 scalars)."""
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("inf", "-inf"):
            return math.inf if s == "inf" else -math.inf
    return float(value)


# ===========================================================================
# Blackboard + per-tick evaluation context
# ===========================================================================


class ScenarioContext(dict):
    """Per-activation mutable blackboard (e.g. a VoiceChoice stashes the chosen value)."""


@dataclass
class TickContext:
    """Bundle handed to ``Criterion.evaluate`` once per success-poll tick."""

    uwb_cache: T.Any
    joint_cache: T.Any
    transcripts: T.List[str]          # received since this sub-task started
    blackboard: ScenarioContext
    elapsed: float                    # seconds since on_start dispatch


# ===========================================================================
# Abstract bases
# ===========================================================================


class Action(ABC):
    """One declarative step in a hook list.

    ``dispatch_async`` True (Speak/Move) → the connect is fired and NOT awaited
    by the hook runner (OM1-style fire-and-forget); pacing comes from the
    dispatcher gaps in TaskSrvConfig. False (Wait/SetContext/Custom) → awaited
    inline because the effect must be sequenced (pause / blackboard write).
    """

    dispatch_async: bool = False
    delay: float = 0.0          # per-action pre-delay (s); concrete actions set it from json5

    @abstractmethod
    async def run(self, ctx: ScenarioContext, hub: "ConnectorHub") -> None: ...


class Criterion(ABC):
    """A sub-task success rule. ``timeout_s`` (uniform) drives the on_fail path."""

    timeout_s: float = 30.0

    @abstractmethod
    def evaluate(self, tc: TickContext) -> bool: ...


# ===========================================================================
# Criteria — sensor (uwb/joint/composite), voice (choice/keyword), trivial
# ===========================================================================


@dataclass
class UwbPose(Criterion):
    """Locomotion: UWB pose within tol of target. ``target`` = [x,y,yaw] or '{ref}'.
    ``tol_yaw=inf`` (default) ignores yaw (UWB kp_angular=0 path)."""

    target: T.Any = None
    tol_xy: float = 0.15
    tol_yaw: float = math.inf
    timeout_s: float = 30.0

    def evaluate(self, tc: TickContext) -> bool:
        target = resolve_value(self.target, tc.blackboard)
        if not target or len(target) < 2:
            return False
        pose = getattr(tc.uwb_cache, "value", None)
        x, y, yaw = _get(pose, "x"), _get(pose, "y"), _get(pose, "yaw")
        if x is None or y is None:
            return False
        if math.hypot(x - float(target[0]), y - float(target[1])) > self.tol_xy:
            return False
        if yaw is not None and not math.isinf(self.tol_yaw) and len(target) >= 3:
            if abs(_wrap_pi(yaw - float(target[2]))) > self.tol_yaw:
                return False
        return True


@dataclass
class JointState(Criterion):
    """Manipulation: a single joint reaches ``pos`` within ``tol``."""

    joint: str = ""
    pos: T.Any = 0.0
    tol: float = 0.05
    timeout_s: float = 10.0

    def evaluate(self, tc: TickContext) -> bool:
        actual = _joint_pos(getattr(tc.joint_cache, "value", None), self.joint)
        target = resolve_value(self.pos, tc.blackboard)
        if actual is None or target is None:
            return False
        return abs(actual - float(target)) <= self.tol


@dataclass
class Composite(Criterion):
    """All-of: succeeds when every child evaluates True."""

    children: T.List[Criterion] = field(default_factory=list)
    timeout_s: float = 30.0

    def evaluate(self, tc: TickContext) -> bool:
        return all(c.evaluate(tc) for c in self.children)


@dataclass
class VoiceChoice(Criterion):
    """Dialog: on a transcript containing one of ``choices``' keys, write the
    chosen value to blackboard[ctx_key] (+ matched key to label_key) and succeed."""

    choices: T.Dict[str, T.Any] = field(default_factory=dict)
    ctx_key: str = "choice"
    label_key: T.Optional[str] = None
    timeout_s: float = 30.0

    def evaluate(self, tc: TickContext) -> bool:
        for text in tc.transcripts:
            hay = text.lower()
            for keyword, value in self.choices.items():
                if keyword.lower() in hay:
                    tc.blackboard[self.ctx_key] = value
                    if self.label_key:
                        tc.blackboard[self.label_key] = keyword
                    return True
        return False


@dataclass
class VoiceKeyword(Criterion):
    """Succeed when a transcript contains any of ``keywords`` (no write)."""

    keywords: T.List[str] = field(default_factory=list)
    timeout_s: float = 30.0

    def evaluate(self, tc: TickContext) -> bool:
        return any(k.lower() in text.lower() for text in tc.transcripts for k in self.keywords)


@dataclass
class Always(Criterion):
    """Immediate success — speak-only / announce-and-advance steps."""

    timeout_s: float = 3600.0

    def evaluate(self, tc: TickContext) -> bool:
        return True


@dataclass
class Delay(Criterion):
    """Succeed after ``seconds`` of elapsed time."""

    seconds: float = 1.0
    timeout_s: float = 3600.0

    def evaluate(self, tc: TickContext) -> bool:
        return tc.elapsed >= self.seconds


def _get(obj: T.Any, name: str) -> T.Optional[float]:
    """obj.name / obj[name] as float, or None."""
    if obj is None:
        return None
    v = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _joint_pos(state: T.Any, joint_name: str) -> T.Optional[float]:
    """Single joint position from a JointState msg / {name:[],position:[]} / {joint:pos}."""
    if state is None:
        return None
    if isinstance(state, dict) and joint_name in state and not isinstance(state.get("name"), (list, tuple)):
        try:
            return float(state[joint_name])
        except (TypeError, ValueError):
            return None
    names = state.get("name") if isinstance(state, dict) else getattr(state, "name", None)
    positions = state.get("position") if isinstance(state, dict) else getattr(state, "position", None)
    if names is None or positions is None:
        return None
    try:
        return float(positions[list(names).index(joint_name)])
    except (ValueError, IndexError, TypeError):
        return None


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ===========================================================================
# Connector facade + actions
# ===========================================================================


class ConnectorHub:
    """Async facade over the Move / Speak connectors. Applies per-call delays
    (TTS-pacing workaround) and logs the intended call when a connector is
    unbound (standalone tests / scaffold mode)."""

    def __init__(
        self,
        move_connector: T.Optional[T.Any] = None,
        speak_connector: T.Optional[T.Any] = None,
        enable_speak: bool = True,
    ) -> None:
        self._move = move_connector
        self._speak = speak_connector
        self._enable_speak = enable_speak

    async def move(self, prompt: str) -> None:
        if self._move is None:
            logging.info("TaskSrv: [MOVE] %s", prompt)
            return
        await self._move.connect(MoveInput(action=prompt))

    async def speak(self, text: str) -> None:
        if not self._enable_speak:
            return
        if self._speak is None:
            logging.info("TaskSrv: [SPEAK] %s", text)
            return
        await self._speak.connect(SpeakInput(action=text))


# YAGNI escape hatch: {custom: "name"} → a coroutine registered by Python code.
CUSTOM_ACTIONS: T.Dict[str, T.Callable[["ScenarioContext", ConnectorHub], T.Awaitable[None]]] = {}


@dataclass
class Speak(Action):
    """Speak ``text`` (template-resolved). Fire-and-forget connect dispatch."""

    dispatch_async = True   # class var (not a dataclass field)

    text: str
    delay: float = 0.0

    async def run(self, ctx: ScenarioContext, hub: ConnectorHub) -> None:
        await hub.speak(resolve_str(self.text, ctx))


@dataclass
class Move(Action):
    """Dispatch ``prompt`` (template-resolved) to the Move connector. Fire-and-forget."""

    dispatch_async = True   # class var (not a dataclass field)

    prompt: str
    delay: float = 0.0

    async def run(self, ctx: ScenarioContext, hub: ConnectorHub) -> None:
        await hub.move(resolve_str(self.prompt, ctx))


@dataclass
class SetContext(Action):
    """Write ``value`` (resolved) to ``blackboard[key]``. Awaited inline."""

    key: str
    value: T.Any
    delay: float = 0.0

    async def run(self, ctx: ScenarioContext, hub: ConnectorHub) -> None:
        ctx[self.key] = resolve_value(self.value, ctx)


@dataclass
class Wait(Action):
    """Pause ``seconds`` — awaited inline so it actually delays the sequence."""

    seconds: float
    delay: float = 0.0

    async def run(self, ctx: ScenarioContext, hub: ConnectorHub) -> None:
        await asyncio.sleep(self.seconds)


@dataclass
class Custom(Action):
    """Run a Python-registered coroutine (``CUSTOM_ACTIONS[name]``). Awaited inline."""

    name: str
    delay: float = 0.0

    async def run(self, ctx: ScenarioContext, hub: ConnectorHub) -> None:
        fn = CUSTOM_ACTIONS.get(self.name)
        if fn is None:
            logging.warning("TaskSrv: custom action %r not registered; skipping", self.name)
            return
        await fn(ctx, hub)


# ===========================================================================
# Scenario model + provider state/config
# ===========================================================================


@dataclass
class SubTask:
    """A success criterion + four lifecycle hook lists (empty list = no-op)."""

    name: str
    success: Criterion
    on_create: T.List[Action] = field(default_factory=list)
    on_start: T.List[Action] = field(default_factory=list)
    on_success: T.List[Action] = field(default_factory=list)
    on_fail: T.List[Action] = field(default_factory=list)


@dataclass
class Scenario:
    """A loaded scenario (built from a JSON5 file)."""

    name: str
    triggers: T.List[str]
    sub_tasks: T.List[SubTask]
    priority: int = 0
    loop: bool = False                                        # re-enter sub-task 0 after the last
    exit_keywords: T.List[str] = field(default_factory=list)  # break the loop → IDLE
    on_scenario_start: T.List[Action] = field(default_factory=list)
    on_scenario_end: T.List[Action] = field(default_factory=list)


class TaskState(str, Enum):
    """External orchestrator state (read by the GUI poller). SUCCESS/FAILED are transient."""

    IDLE = "idle"
    ACTIVE = "active"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class TaskSrvConfig:
    """TaskSrv runtime configuration (populated from config/sous_chef_g1.json5)."""

    # Single active scenario — filename under config/scenarios/ (or a path).
    # TaskSrvProvider loads exactly ONE scenario file at a time (not the whole
    # folder); switch the demo by changing this.
    scenario_file: str = "move_test.json5"
    tick_rate_hz: float = 10.0
    case_insensitive_keywords: bool = True
    enable_speak_feedback: bool = True
    cancel_keywords: T.List[str] = field(default_factory=list)   # graceful abort; empty = off
    cancel_phrase: str = "취소했습니다."
    # Drop transcripts queued before a pause/resume gap longer than this.
    # Assumes ts is time.monotonic() seconds (matches TranscriptEvent.ts). 0 disables.
    stale_audio_s: float = 5.0
    # Connect pacing is per-action (Action.delay in the scenario json5), not here.
    # Loop pump (owned by TaskSrvProvider.run).
    heartbeat_every_ticks: int = 50          # INFO heartbeat cadence; CONV-009. 0 disables.
    swallow_tick_exceptions: bool = True     # log+continue vs stop the loop


# ===========================================================================
# JSON5 scenario loader + validation
#   Validates at load time so a typo is a clear error at startup, not a crash.
# ===========================================================================


class ScenarioConfigError(ValueError):
    """Raised when a scenario JSON5 file is malformed."""


def _req(node: T.Mapping, key: str, where: str) -> T.Any:
    if key not in node or node[key] is None:
        raise ScenarioConfigError(f"{where}: missing required field {key!r}")
    return node[key]


def _req_str(value: T.Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioConfigError(f"field {field_name!r} must be a non-empty string, got {value!r}")
    return value


_ACTION_KEYS: T.Dict[str, T.Callable[[T.Any, float], Action]] = {
    "speak": lambda v, d: Speak(text=_req_str(v, "speak"), delay=d),
    "move": lambda v, d: Move(prompt=_req_str(v, "move"), delay=d),
    "wait": lambda v, d: Wait(seconds=float(v), delay=d),
    "custom": lambda v, d: Custom(name=_req_str(v, "custom"), delay=d),
    "set_context": lambda v, d: _build_set_context(v, d),
}


def _build_set_context(v: T.Any, delay: float) -> Action:
    if not isinstance(v, dict) or "key" not in v or "value" not in v:
        raise ScenarioConfigError("set_context must be a mapping with 'key' and 'value'")
    return SetContext(key=_req_str(v["key"], "set_context.key"), value=v["value"], delay=delay)


_CRITERION_BUILDERS: T.Dict[str, T.Callable[[T.Mapping], Criterion]] = {
    "uwb_pose": lambda n: UwbPose(
        target=_req(n, "target", "uwb_pose"),
        tol_xy=float(n.get("tol_xy", 0.15)),
        tol_yaw=_as_float(n.get("tol_yaw", "inf")),
        timeout_s=float(n.get("timeout_s", 30.0)),
    ),
    "joint_state": lambda n: JointState(
        joint=_req_str(_req(n, "joint", "joint_state"), "joint"),
        pos=_req(n, "pos", "joint_state"),
        tol=float(n.get("tol", 0.05)),
        timeout_s=float(n.get("timeout_s", 10.0)),
    ),
    "composite": lambda n: Composite(
        children=[build_criterion(c) for c in _req_list(n, "children", "composite")],
        timeout_s=float(n.get("timeout_s", 30.0)),
    ),
    "voice_choice": lambda n: VoiceChoice(
        choices=dict(_req_mapping(n, "choices", "voice_choice")),
        ctx_key=_req_str(_req(n, "ctx_key", "voice_choice"), "ctx_key"),
        label_key=n.get("label_key"),
        timeout_s=float(n.get("timeout_s", 30.0)),
    ),
    "voice_keyword": lambda n: VoiceKeyword(
        keywords=list(_req_list(n, "keywords", "voice_keyword")),
        timeout_s=float(n.get("timeout_s", 30.0)),
    ),
    "always": lambda n: Always(timeout_s=float(n.get("timeout_s", 3600.0))),
    "delay": lambda n: Delay(seconds=float(_req(n, "seconds", "delay")), timeout_s=float(n.get("timeout_s", 3600.0))),
}


def _req_list(n: T.Mapping, key: str, where: str) -> T.List:
    v = _req(n, key, where)
    if not isinstance(v, list) or not v:
        raise ScenarioConfigError(f"{where} {key!r} must be a non-empty list")
    return v


def _req_mapping(n: T.Mapping, key: str, where: str) -> T.Mapping:
    v = _req(n, key, where)
    if not isinstance(v, dict) or not v:
        raise ScenarioConfigError(f"{where} {key!r} must be a non-empty mapping")
    return v


def build_action(node: T.Any) -> Action:
    if not isinstance(node, dict):
        raise ScenarioConfigError(f"action must be a mapping, got {node!r}")
    present = [k for k in node if k in _ACTION_KEYS]
    if len(present) != 1:
        raise ScenarioConfigError(f"action node needs exactly one of {sorted(_ACTION_KEYS)}; got {sorted(node)}")
    delay = float(node.get("delay", 0.0))
    return _ACTION_KEYS[present[0]](node[present[0]], delay)


def build_criterion(node: T.Any) -> Criterion:
    if not isinstance(node, dict):
        raise ScenarioConfigError(f"success criterion must be a mapping, got {node!r}")
    tag = node.get("type")
    if tag is None:
        raise ScenarioConfigError("success criterion missing 'type'")
    builder = _CRITERION_BUILDERS.get(tag)
    if builder is None:
        raise ScenarioConfigError(f"unknown criterion type {tag!r}; known: {sorted(_CRITERION_BUILDERS)}")
    return builder(node)


def _build_actions(raw: T.Any, where: str) -> T.List[Action]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScenarioConfigError(f"{where} must be a list of actions")
    return [build_action(a) for a in raw]


def _build_sub_task(node: T.Any, scenario: str) -> SubTask:
    if not isinstance(node, dict):
        raise ScenarioConfigError(f"scenario {scenario!r}: sub_task must be a mapping")
    name = _req_str(_req(node, "name", f"sub_task in {scenario!r}"), "name")
    where = f"scenario {scenario!r} sub_task {name!r}"
    return SubTask(
        name=name,
        success=build_criterion(_req(node, "success", where)),
        on_create=_build_actions(node.get("on_create"), f"{where}.on_create"),
        on_start=_build_actions(node.get("on_start"), f"{where}.on_start"),
        on_success=_build_actions(node.get("on_success"), f"{where}.on_success"),
        on_fail=_build_actions(node.get("on_fail"), f"{where}.on_fail"),
    )


def build_scenario(doc: T.Any, source: str = "<dict>") -> Scenario:
    if not isinstance(doc, dict):
        raise ScenarioConfigError(f"{source}: top-level scenario must be a mapping")
    name = _req_str(_req(doc, "name", source), "name")
    triggers = _req_list(doc, "triggers", f"scenario {name!r}")
    return Scenario(
        name=name,
        triggers=[str(t) for t in triggers],
        sub_tasks=[_build_sub_task(s, name) for s in _req_list(doc, "sub_tasks", f"scenario {name!r}")],
        priority=int(doc.get("priority", 0)),
        loop=bool(doc.get("loop", False)),
        exit_keywords=[str(k) for k in doc.get("exit_keywords", [])],
        on_scenario_start=_build_actions(doc.get("on_scenario_start"), f"scenario {name!r}.on_scenario_start"),
        on_scenario_end=_build_actions(doc.get("on_scenario_end"), f"scenario {name!r}.on_scenario_end"),
    )


def load_scenario_file(path: Path) -> Scenario:
    try:
        doc = json5.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as e:
        raise ScenarioConfigError(f"{Path(path).name}: JSON5 parse error: {e}") from e
    try:
        return build_scenario(doc, source=Path(path).name)
    except ScenarioConfigError as e:
        raise ScenarioConfigError(f"{Path(path).name}: {e}") from e


# ===========================================================================
# Engine
# ===========================================================================


class _Phase(Enum):
    """Internal sub-task phase (distinct from the external TaskState)."""

    NONE = 0
    ENTERING = 1     # running on_create / on_start
    RUNNING = 2      # polling success
    COMPLETING = 3   # running on_success / on_fail / cancel / exit


@singleton
class TaskSrvProvider:
    """Scenario-driven sub-task orchestrator. Owns its asyncio loop (``run``)."""

    def __init__(self, config: T.Optional[TaskSrvConfig] = None):
        self._config = config or TaskSrvConfig()
        self._scenarios: T.List[Scenario] = []
        self._trigger_index: T.Dict[str, Scenario] = {}

        self._state = TaskState.IDLE
        self._phase = _Phase.NONE
        self._active_scenario: T.Optional[Scenario] = None
        self._active_idx = -1
        self._dispatch_ts = 0.0
        self._blackboard = ScenarioContext()
        self._active_transcripts: T.List[str] = []

        self._unitree_g1 = UnitreeG1Provider()          # @singleton — sensor caches
        self._move_connector: T.Optional[T.Any] = None
        self._speak_connector: T.Optional[T.Any] = None
        self._hub = ConnectorHub()

        self._loop: T.Optional[asyncio.AbstractEventLoop] = None
        self._hook_task: T.Optional[asyncio.Task] = None

        self._running = False
        self._inbound: "queue.Queue[T.Tuple[str, T.Optional[float]]]" = queue.Queue(maxsize=32)
        logging.info("TaskSrvProvider: initialized (tick=%.1fHz)", self._config.tick_rate_hz)

    # -- wiring / lifecycle ------------------------------------------------
    def bind(self, move_connector: T.Any = None, speak_connector: T.Optional[T.Any] = None) -> None:
        """Wire the non-singleton Connectors (CONV-010). Both optional."""
        self._move_connector = move_connector
        self._speak_connector = speak_connector

    def start(self, scenarios: T.Optional[T.List[Scenario]] = None) -> None:
        """Build the connector hub + load scenarios (JSON5 when ``scenarios`` is None)."""
        self._hub = ConnectorHub(
            self._move_connector, self._speak_connector,
            enable_speak=self._config.enable_speak_feedback,
        )
        if scenarios is None:
            from config.scenarios import load as _load
            scenarios = _load(self._config.scenario_file)
        self._scenarios = list(scenarios)
        self._trigger_index = self._build_trigger_index(self._scenarios)
        self._running = True
        if not self._scenarios:
            logging.warning("TaskSrvProvider: no scenarios loaded")
        if not self._config.cancel_keywords:
            logging.warning("TaskSrvProvider: cancel_keywords empty — no graceful abort path")
        logging.info("TaskSrvProvider: loaded %d scenarios (%d triggers)",
                     len(self._scenarios), len(self._trigger_index))

    def stop(self) -> None:
        """Abort any active scenario, reset to IDLE (called on shutdown)."""
        self._cancel_hook_task()
        self._reset_active()
        self._state = TaskState.IDLE
        self._running = False

    # -- main loop (owned here; TaskSrvBg just calls this) -----------------
    def run(self, stop_event) -> None:
        """Own the asyncio loop: pump ``tick()`` at tick_rate_hz (drift-free) and
        run hook coroutines until ``stop_event`` is set."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        period = 1.0 / float(self._config.tick_rate_hz)
        hb = self._config.heartbeat_every_ticks
        sched = {"next": loop.time(), "n": 0}

        def _pump() -> None:
            if stop_event.is_set():
                loop.stop()
                return
            try:
                self.tick()
            except Exception:
                if self._config.swallow_tick_exceptions:
                    logging.exception("TaskSrvProvider: tick() raised; swallowing")
                else:
                    logging.exception("TaskSrvProvider: tick() raised; stopping loop")
                    loop.stop()
                    return
            sched["n"] += 1
            if hb and sched["n"] % hb == 0:
                logging.info("TaskSrvProvider: heartbeat tick=%d state=%s scenario=%s",
                             sched["n"], self._state.value, self.active_scenario_name)
            sched["next"] += period
            if sched["next"] - loop.time() < 0:           # overran — reset baseline
                logging.warning("TaskSrvProvider: tick overran period; resetting baseline")
                sched["next"] = loop.time()
            loop.call_at(sched["next"], _pump)

        logging.info("TaskSrvProvider: loop entering (period=%.3fs)", period)
        loop.call_soon(_pump)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        logging.info("TaskSrvProvider: loop exited (stop signalled)")

    # -- inbound STT (STT thread → enqueue only) ---------------------------
    def on_audio(self, text: str, ts: T.Optional[float] = None) -> None:
        """Enqueue a transcript for the next tick (keeps the engine single-writer)."""
        if not self._running:
            return
        try:
            self._inbound.put_nowait((text, ts))
        except queue.Full:
            logging.warning("TaskSrvProvider: inbound queue full; dropping %r", text)

    def _process_audio(self, text: str, ts: T.Optional[float]) -> None:
        if ts is not None and self._config.stale_audio_s > 0:
            age = time.monotonic() - ts
            if age > self._config.stale_audio_s:
                logging.info("TaskSrvProvider: dropping stale audio (age=%.2fs): %r", age, text)
                return
        if self._active_scenario is not None:
            if self._is_cancel_trigger(text):
                logging.info("TaskSrvProvider: cancel keyword %r → abort", text)
                self._cancel_active()
            elif self._active_scenario.exit_keywords and self._matches_any(text, self._active_scenario.exit_keywords):
                logging.info("TaskSrvProvider: exit keyword %r → end scenario", text)
                self._exit_active()
            else:
                # Buffer for the current sub-task's criterion (voice gate). A
                # phrase belonging to a consumed sub-task matches nothing here.
                self._active_transcripts.append(text)
            return
        matched = self._match_trigger(text)
        if matched is not None:
            logging.info("TaskSrvProvider: trigger %r matched scenario '%s'", text, matched.name)
            self._activate(matched)

    # -- tick: success poll (runs on the loop thread) ----------------------
    def tick(self) -> None:
        while True:
            try:
                text, ts = self._inbound.get_nowait()
            except queue.Empty:
                break
            self._process_audio(text, ts)

        if self._state in (TaskState.SUCCESS, TaskState.FAILED):
            self._state = TaskState.IDLE           # clear transient terminal state
            return
        if self._phase is not _Phase.RUNNING:
            return
        st = self.active_sub_task
        if st is None:
            self._begin_fail("no active sub-task")
            return
        tc = TickContext(
            uwb_cache=getattr(self._unitree_g1, "uwb_pose", None),
            joint_cache=getattr(self._unitree_g1, "joint_state", None),
            transcripts=self._active_transcripts,
            blackboard=self._blackboard,
            elapsed=time.monotonic() - self._dispatch_ts,
        )
        try:
            done = st.success.evaluate(tc)
        except Exception:
            logging.exception("TaskSrvProvider.tick: criterion.evaluate raised")
            done = False
        if done:
            self._begin_complete_success()
        elif tc.elapsed > st.success.timeout_s:
            self._begin_fail(f"sub-task timeout ({tc.elapsed:.1f}s)")

    # -- read-only state (GUI poller) --------------------------------------
    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def active_sub_task(self) -> T.Optional[SubTask]:
        if self._active_scenario is None or not (0 <= self._active_idx < len(self._active_scenario.sub_tasks)):
            return None
        return self._active_scenario.sub_tasks[self._active_idx]

    @property
    def active_scenario_name(self) -> T.Optional[str]:
        return self._active_scenario.name if self._active_scenario else None

    # -- activation + sub-task lifecycle (hook coroutines on the loop) -----
    def _activate(self, scenario: Scenario) -> None:
        self._active_scenario = scenario
        self._active_idx = 0
        self._blackboard = ScenarioContext()
        self._active_transcripts = []
        self._state = TaskState.ACTIVE
        self._phase = _Phase.ENTERING
        self._schedule(self._activate_coro())

    async def _activate_coro(self) -> None:
        assert self._active_scenario is not None
        await self._run_actions(self._active_scenario.on_scenario_start)
        await self._enter_coro()

    async def _enter_coro(self) -> None:
        """Enter the current sub-task: on_create → on_start → RUNNING."""
        self._phase = _Phase.ENTERING
        self._active_transcripts = []
        st = self.active_sub_task
        if st is None:
            self._finish_scenario(success=False)
            return
        await self._run_actions(st.on_create)
        self._dispatch_ts = time.monotonic()         # timeout clock starts at on_start
        await self._run_actions(st.on_start)
        self._phase = _Phase.RUNNING

    def _begin_complete_success(self) -> None:
        self._phase = _Phase.COMPLETING
        self._schedule(self._complete_success_coro())

    async def _complete_success_coro(self) -> None:
        st = self.active_sub_task
        if st is not None:
            await self._run_actions(st.on_success)
        assert self._active_scenario is not None
        self._active_idx += 1
        if self._active_idx >= len(self._active_scenario.sub_tasks):
            if self._active_scenario.loop:
                logging.info("TaskSrvProvider: scenario '%s' loop → restart", self._active_scenario.name)
                self._active_idx = 0
                await self._enter_coro()
            else:
                await self._run_actions(self._active_scenario.on_scenario_end)
                self._finish_scenario(success=True)
        else:
            await self._enter_coro()

    def _begin_fail(self, reason: str) -> None:
        logging.warning("TaskSrvProvider: scenario '%s' sub-task %d failed (%s)",
                        self._active_scenario.name if self._active_scenario else "?", self._active_idx, reason)
        self._phase = _Phase.COMPLETING
        self._schedule(self._complete_fail_coro())

    async def _complete_fail_coro(self) -> None:
        st = self.active_sub_task
        if st is not None:
            await self._run_actions(st.on_fail)
        if self._active_scenario is not None:
            await self._run_actions(self._active_scenario.on_scenario_end)
        self._finish_scenario(success=False)

    def _cancel_active(self) -> None:
        self._phase = _Phase.COMPLETING
        self._cancel_hook_task()
        self._schedule(self._cancel_coro())

    async def _cancel_coro(self) -> None:
        if self._config.cancel_phrase:
            await self._hub.speak(self._config.cancel_phrase)
        self._reset_active()
        self._state = TaskState.IDLE

    def _exit_active(self) -> None:
        self._phase = _Phase.COMPLETING
        self._cancel_hook_task()
        self._schedule(self._exit_coro())

    async def _exit_coro(self) -> None:
        if self._active_scenario is not None:
            await self._run_actions(self._active_scenario.on_scenario_end)
        self._reset_active()
        self._state = TaskState.IDLE

    def _finish_scenario(self, success: bool) -> None:
        name = self._active_scenario.name if self._active_scenario else "?"
        logging.info("TaskSrvProvider: scenario '%s' %s", name, "completed" if success else "failed")
        self._reset_active()
        self._state = TaskState.SUCCESS if success else TaskState.FAILED

    # -- hook execution ----------------------------------------------------
    async def _run_actions(self, actions: T.List[Action]) -> None:
        """Run a hook list. Speak/Move dispatch fire-and-forget as independent
        timers (each waits its own ``delay``, no cascade); Wait/SetContext/Custom
        run inline after their ``delay``."""
        for action in actions:
            if action.dispatch_async and self._loop is not None:
                self._loop.create_task(self._delayed_dispatch(action))   # independent timer
            else:
                if action.delay > 0:
                    await asyncio.sleep(action.delay)
                await self._safe_run(action)                             # inline control

    async def _delayed_dispatch(self, action: Action) -> None:
        """Wait this action's own pre-delay, then fire its connect (fire-and-forget)."""
        if action.delay > 0:
            await asyncio.sleep(action.delay)
        await self._safe_run(action)

    async def _safe_run(self, action: Action) -> None:
        try:
            await action.run(self._blackboard, self._hub)
        except Exception:
            logging.exception("TaskSrvProvider: action %r raised", action)

    def _schedule(self, coro: T.Awaitable) -> None:
        """Schedule a lifecycle coroutine on the loop (or run inline if no loop — sync tests)."""
        if self._loop is not None:
            self._hook_task = self._loop.create_task(coro)  # type: ignore[arg-type]
        else:
            asyncio.run(coro)  # type: ignore[arg-type]

    def _cancel_hook_task(self) -> None:
        if self._hook_task is not None and not self._hook_task.done():
            self._hook_task.cancel()
        self._hook_task = None

    def _reset_active(self) -> None:
        self._active_scenario = None
        self._active_idx = -1
        self._dispatch_ts = 0.0
        self._phase = _Phase.NONE
        self._active_transcripts = []

    # -- trigger / keyword matching ----------------------------------------
    def _build_trigger_index(self, scenarios: T.List[Scenario]) -> T.Dict[str, Scenario]:
        index: T.Dict[str, Scenario] = {}
        for sc in scenarios:
            if not sc.sub_tasks:
                raise ValueError(f"Scenario {sc.name!r} has no sub_tasks")
            for trig in sc.triggers:
                key = trig.lower() if self._config.case_insensitive_keywords else trig
                if key in index and index[key].name != sc.name:
                    raise ValueError(f"Duplicate trigger '{trig}' in '{index[key].name}' and '{sc.name}'")
                index[key] = sc
        return index

    def _match_trigger(self, text: str) -> T.Optional[Scenario]:
        haystack = text.lower() if self._config.case_insensitive_keywords else text
        matches = [sc for needle, sc in self._trigger_index.items() if needle in haystack]
        if not matches:
            return None
        matches.sort(key=lambda s: s.priority, reverse=True)
        return matches[0]

    def _is_cancel_trigger(self, text: str) -> bool:
        return self._matches_any(text, self._config.cancel_keywords)

    def _matches_any(self, text: str, keywords: T.List[str]) -> bool:
        if not keywords:
            return False
        if self._config.case_insensitive_keywords:
            hay = text.lower()
            return any(k.lower() in hay for k in keywords)
        return any(k in text for k in keywords)
