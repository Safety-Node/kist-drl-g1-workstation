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

TODO(REQ-31) [TASK-44]: implement routing keyword dispatch in connect().
TODO(REQ-31) [TASK-44]: try/except wrap VLA.infer + publish_loco_cmd +
                        NavigationProvider.submit_nav_subtask — log +
                        swallow per fire-and-forget caller contract.
TODO(REQ-31) [TASK-44]: stop() lifecycle — track in-flight asyncio tasks
                        (weak-ref set), cancel on shutdown. Add Connector
                        to run.py._stop_runtime once stop() actually does
                        something.
TODO(REQ-31) [TASK-44]: E-STOP cancellation — UnitreeG1.estop edge triggers
                        cancel of in-flight VLA.infer + NavigationProvider
                        control loop (NavigationProvider also registers its
                        own estop callback for zero-Twist publish).
"""

import logging

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
_LOCO_KEYWORDS = tuple(_LOCO_MAP.keys())

# Navigation keyword set — substring match against the lowercased prompt.
# Continuous walking goes through NavigationProvider; the
# discrete LocoClient presets above ("stand up" etc.) win when both
# match, see ``_route`` order.
_NAV_KEYWORDS = frozenset({
    "walk", "go to", "move to", "navigate", "approach",
    "걸어", "가", "이동", "접근",
})


class MoveConnector(ActionConnector[ActionConfig, MoveInput]):
    """Routes MoveInput.action prompts to LocoClient preset, NavigationProvider, or VLA Provider."""

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # Ordering: run.py constructs all Provider singletons
        # before this connector, so the @singleton fetches return the
        # already-built instances.
        self._vla = VLAProvider()
        self._unitree_g1 = UnitreeG1Provider()
        self._navigation = NavigationProvider()
        logging.info("MoveConnector: skeleton initialized")

    async def connect(self, output_interface: MoveInput) -> None:
        # TODO(REQ-31) [TASK-44]: try/except — log + swallow, NEVER re-raise
        # TODO(REQ-31) [TASK-44]: 3-way routing — discrete loco → nav → VLA.
        #     See module docstring; same dispatch as ``_route`` below
        #     once the awaitable contract for VLA.infer is settled.
        raise NotImplementedError("MoveConnector.connect: TBD [TASK-44]")


def _route(prompt: str) -> None:
    """3-way prompt router (2026-05-26).

    Reference dispatch shared by ``MoveConnector.connect()`` (async path)
    and any synchronous test/CLI driver. Not called from production code
    until ``connect()`` is implemented — kept here so the routing policy
    has one source of truth.
    """
    key = prompt.strip().lower()
    # 1. Discrete LocoClient preset (StandUp/SitDown/...)
    for kw, name in _LOCO_MAP.items():
        if kw in key:
            UnitreeG1Provider().publish_loco_cmd({"name": name})
            return
    # 2. Navigation (continuous walking via NavigationProvider)
    if any(kw in key for kw in _NAV_KEYWORDS):
        NavigationProvider().submit_nav_subtask(prompt)
        return
    # 3. Manipulation (VLA arm/hand)
    VLAProvider().infer(prompt)
