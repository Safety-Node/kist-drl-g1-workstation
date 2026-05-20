"""
Speak Connector -- KIST DRL G1 Workstation
==========================================

drawio C4 Container:
    Name        : Speak Connector
    Technology  : Python
    Description : Routes speak actions to TTS Provider.

Edges:
    Cortex            -> Speak Connector : Speak Response [text]
    Speak Connector   -> TTS Provider    : Text payload (Naver Clova)

TBD:
    - Sentence-segment for streaming TTS (lower latency)
    - Language detection / fallback (ko -> en if Korean is unavailable)
    - Interrupt handling: stop in-flight playback on new utterance
    - Backpressure: drop if queue depth exceeds threshold
    - Logging hook for safety review (every spoken phrase)
"""

import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput


class SpeakG1WorkstationConnector(ActionConnector[ActionConfig, SpeakInput]):
    """
    Connector that forwards Cortex's textual speak action to the
    workstation TTS Provider (Naver Clova).
    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # TODO: obtain NaverClovaTTSProvider handle
        logging.info("SpeakG1WorkstationConnector: skeleton initialized")

    async def connect(self, output_interface: SpeakInput) -> None:
        """Forward SpeakInput.action (text) to the TTS Provider."""
        # TODO: split into sentences for streaming synthesis
        # TODO: await NaverClovaTTSProvider.synthesize(text)
        # TODO: pipe resulting PCM via UnitreeG1Provider.publish_audio_out
        raise NotImplementedError("SpeakG1WorkstationConnector.connect: TBD")
