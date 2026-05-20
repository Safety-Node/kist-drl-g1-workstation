"""
Move Connector -- KIST DRL G1 Workstation
=========================================

drawio C4 Container:
    Name        : Move Connector
    Technology  : Python
    Description : Routes subtasks cmd.

Edges:
    Cortex          -> Move Connector    : SubTasks Control Cmd [Text]
    Move Connector  -> VLA Provider      : Upper Body Cmd [Text]
    Move Connector  -> UnitreeG1 Provider: Nav Cmd [Text]

TBD:
    - Sub-task parser: split LLM action plan into atomic move commands
    - Routing decision:
        * upper-body manipulation (knife / tongs / 5-finger hand) -> VLA Provider
        * locomotion / navigation goal                            -> UnitreeG1 Provider
    - Sequencing across multi-step sub-tasks
    - E-STOP propagation: cancel queued sub-tasks immediately
"""

import logging

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput


class MoveG1WorkstationConnector(ActionConnector[ActionConfig, MoveInput]):
    """
    Connector that splits Cortex's textual sub-task command into
    (a) VLA-bound upper-body commands and
    (b) UnitreeG1-bound navigation commands.
    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # TODO: get handles to VLAGrootProvider and UnitreeG1Provider
        logging.info("MoveG1WorkstationConnector: skeleton initialized")

    async def connect(self, output_interface: MoveInput) -> None:
        """Route a single MoveInput to VLA Provider and/or UnitreeG1 Provider."""
        # TODO: classify intent (locomotion vs manipulation)
        # TODO: dispatch:
        #     if locomotion: UnitreeG1Provider.publish_nav_cmd(...)
        #     if manipulation: VLAGrootProvider.infer(...)
        # TODO: await Validated Cmd from SafetyProvider before publish
        raise NotImplementedError("MoveG1WorkstationConnector.connect: TBD")
