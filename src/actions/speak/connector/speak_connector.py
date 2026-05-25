"""
Speak Connector

Routes a SpeakInput (text) to the TTS Provider via the OM1
``ActionConnector.connect(SpeakInput)`` contract.
"""

import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput


class SpeakConnector(ActionConnector[ActionConfig, SpeakInput]):
    """Forwards a SpeakInput to the TTS Provider."""

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # TODO: bind(tts=...) once added
        logging.info("SpeakConnector: skeleton initialized")

    async def connect(self, output_interface: SpeakInput) -> None:
        # TODO: await self._tts.synthesize(output_interface.action)
        raise NotImplementedError("SpeakConnector.connect: TBD")
