"""
Move Connector

Routes TaskSrvProvider sub-task prompts to the VLA Provider via the OM1
``ActionConnector.connect(MoveInput)`` contract. Whole-body VLA (CONV-005)
so no locomotion / manipulation split at this layer.
"""

import logging

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput


class MoveConnector(ActionConnector[ActionConfig, MoveInput]):
    """Forwards a MoveInput to the VLA Provider."""

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # TODO: bind(vla=...) once added
        logging.info("MoveConnector: skeleton initialized")

    async def connect(self, output_interface: MoveInput) -> None:
        # TODO: await self._vla.infer(output_interface.action)
        raise NotImplementedError("MoveConnector.connect: TBD")
