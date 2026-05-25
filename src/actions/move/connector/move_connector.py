"""
Move Connector [TASK-44, REQ-31]

Routes TaskSrvProvider sub-task prompts (free-form text) to the VLA Provider
for whole-body manipulation + locomotion (CONV-005), with a deterministic
escape hatch via UnitreeG1.publish_loco_cmd for posture transitions that
GR00T N1.7 may not be reliably trained on (demo entry / exit / fallback).

Routing policy:
  - prompts matching ``_LOCO_KEYWORDS`` (case-insensitive substring) →
    UnitreeG1.publish_loco_cmd(StandUp / SitDown / Damp / BalanceStand)
  - all other prompts → VLA.infer(prompt) (whole-body action chunk stream)

Threading + error policy:
  - connect() is async per OM1 ActionConnector contract.
  - Caller is TaskSrvProvider._schedule_coro — **fire-and-forget**.
    Exceptions raised here turn into "Task exception was never retrieved"
    one-line warnings and disappear (demo-debug nightmare). Implementation
    MUST try/except and log + swallow.

TODO(REQ-31) [TASK-44]: implement routing keyword dispatch in connect().
TODO(REQ-31) [TASK-44]: try/except wrap VLA.infer + publish_loco_cmd —
                        log + swallow per fire-and-forget caller contract.
TODO(REQ-31) [TASK-44]: stop() lifecycle — track in-flight asyncio tasks
                        (weak-ref set), cancel on shutdown. Add Connector
                        to run.py._stop_runtime once stop() actually does
                        something.
TODO(REQ-31) [TASK-44]: E-STOP cancellation — UnitreeG1.estop edge triggers
                        cancel of in-flight VLA.infer.
"""

import logging

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput
from providers.unitree_g1_provider import UnitreeG1Provider
from providers.vla_provider import VLAProvider


_LOCO_KEYWORDS = ("stand up", "sit down", "damp", "balance stand")


class MoveConnector(ActionConnector[ActionConfig, MoveInput]):
    """Routes MoveInput.action prompts to VLA Provider or UnitreeG1 loco_cmd."""

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # CONV-001 ordering: run.py constructs both Provider singletons
        # before this connector, so the @singleton fetches return the
        # already-built instances.
        self._vla = VLAProvider()
        self._unitree_g1 = UnitreeG1Provider()
        logging.info("MoveConnector: skeleton initialized")

    async def connect(self, output_interface: MoveInput) -> None:
        # TODO(REQ-31) [TASK-44]: try/except — log + swallow, NEVER re-raise
        # TODO(REQ-31) [TASK-44]: lower = output_interface.action.lower()
        #     if any(k in lower for k in _LOCO_KEYWORDS):
        #         self._unitree_g1.publish_loco_cmd({"name": <StandUp|SitDown|Damp|BalanceStand>})
        #     else:
        #         await self._vla.infer(output_interface.action)
        raise NotImplementedError("MoveConnector.connect: TBD [TASK-44]")
