"""
Move Connector [TASK-44, REQ-31]

Routes TaskSrvProvider sub-task prompts (free-form text) to one of three
downstream paths (2026-05-26 split):
  1. discrete LocoClient preset (StandUp / SitDown / Damp / BalanceStand)
     via UnitreeG1.publish_loco_cmd — demo entry/exit + posture fallback
  2. navigation (continuous walking velocity) via NavigationProvider —
     PC NavigationProvider runs an internal Kalman + planner and emits
     Twist on /bridge/cmd/vel → NX motor_controller LocoClient.Move
  3. manipulation (arm + hand only post-split) via VLA Provider —
     whole-body VLA chunk stream on /bridge/cmd/{arm,low}

Routing policy (case-insensitive substring on prompt):
  - prompt key in ``_LOCO_MAP`` → discrete loco (path 1)
  - any keyword in ``_NAV_KEYWORDS`` appears → navigation (path 2)
  - otherwise → manipulation VLA (path 3)

Order matters: discrete loco is checked first because "stand up" could
otherwise be ambiguous with a future nav keyword; nav is checked before
VLA because the VLA scope shrank to arm/hand on 2026-05-26 (VLA
범위 축소 — locomotion 분리).

Threading + error policy:
  - connect() is async per OM1 ActionConnector contract.
  - Caller is TaskSrvProvider._schedule_coro — **fire-and-forget**.
    Exceptions raised here turn into "Task exception was never retrieved"
    one-line warnings and disappear (demo-debug nightmare). Implementation
    MUST try/except and log + swallow.

TODO(REQ-31) [TASK-44]: stop() lifecycle — track in-flight asyncio tasks
                        (weak-ref set), cancel on shutdown. Add Connector
                        to run.py._stop_runtime once stop() actually does
                        something.
TODO(REQ-31) [TASK-44]: E-STOP cancellation — UnitreeG1.estop edge triggers
                        cancel of in-flight VLA.infer + NavigationProvider
                        control loop (NavigationProvider also registers its
                        own estop callback for zero-Twist publish).
"""

import asyncio
import logging
import re
import weakref

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput
from providers.navigation_provider import NavigationProvider
from providers.unitree_g1_provider import UnitreeG1Provider
from providers.vla_provider import VLAProvider


# prompt substring → Unitree LocoClient command name.
# Keep keys lowercase; matcher lowercases the prompt before lookup.
_LOCO_MAP = {
    "stand up": "StandUp",
    "sit down": "SitDown",
    "damp": "Damp",
    "balance stand": "BalanceStand",
}

# Navigation keyword set — substring match against the lowercased prompt.
# Discrete LocoClient presets above ("stand up" etc.) win when both match.
_NAV_KEYWORDS = frozenset({
    "walk to", "go to", "move to", "navigate",
    "걸어", "이동", "접근",
})

_log = logging.getLogger(__name__)


class MoveConnector(ActionConnector[ActionConfig, MoveInput]):
    """Routes MoveInput.action prompts to LocoClient preset, NavigationProvider, or VLA Provider."""

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # run.py constructs Provider singletons before connectors (CONV-010),
        # so these fetches return the already-built instances.
        self._vla = VLAProvider()
        self._unitree_g1 = UnitreeG1Provider()
        self._navigation = NavigationProvider()
        # Weak-ref set of in-flight asyncio.Tasks for stop() cancellation.
        # TODO(REQ-31) [TASK-44]: wire into stop() + run.py._stop_runtime.
        self._inflight: weakref.WeakSet = weakref.WeakSet()
        _log.info("MoveConnector: initialized")

    async def connect(self, output_interface: MoveInput) -> None:
        """Route prompt to loco / nav / VLA path. Never re-raises (fire-and-forget contract)."""
        prompt = output_interface.action
        key = prompt.strip().lower()

        # Register current task for future stop() cancellation.
        task = asyncio.current_task()
        if task is not None:
            self._inflight.add(task)

        try:
            # 1. Discrete LocoClient preset — checked first (deterministic FSM)
            for kw, name in _LOCO_MAP.items():
                if kw in key:
                    _log.info("MoveConnector: loco path — %s (prompt=%r)", name, prompt)
                    self._unitree_g1.publish_loco_cmd({"name": name})
                    return

            # 2. Direct twist — "twist vx=.. vy=.. vyaw=.. duration=.."
            if key.startswith("twist"):
                await self._do_twist(prompt)
                return

            # 3. Navigation — continuous walking via NavigationProvider
            if any(kw in key for kw in _NAV_KEYWORDS):
                _log.info("MoveConnector: nav path — prompt=%r", prompt)
                self._navigation.submit_nav_subtask(prompt)
                return

            # 4. Manipulation — VLA arm/hand chunk stream
            _log.info("MoveConnector: VLA path — prompt=%r", prompt)
            await self._vla.infer(prompt)

        except asyncio.CancelledError:
            _log.info("MoveConnector: connect() cancelled (prompt=%r)", prompt)
            raise  # CancelledError must propagate for asyncio task lifecycle
        except Exception:
            _log.exception(
                "MoveConnector: connect() error (prompt=%r) — swallowing per fire-and-forget contract",
                prompt,
            )

    async def _do_twist(self, prompt: str) -> None:
        """Parse 'twist vx=.. vy=.. vyaw=.. duration=..' and publish at 10 Hz."""
        params = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration": 1.0}
        for m in re.finditer(r"(vx|vy|vyaw|duration)=([-\d.]+)", prompt):
            params[m.group(1)] = float(m.group(2))
        vx, vy, vyaw = params["vx"], params["vy"], params["vyaw"]
        duration = params["duration"]
        _log.info(
            "MoveConnector: twist path — vx=%.2f vy=%.2f vyaw=%.2f duration=%.1fs",
            vx, vy, vyaw, duration,
        )
        dt = 0.1
        steps = max(1, round(duration / dt))
        for _ in range(steps):
            self._unitree_g1.publish_twist(vx, vy, vyaw)
            await asyncio.sleep(dt)
        self._unitree_g1.publish_twist(0.0, 0.0, 0.0)
