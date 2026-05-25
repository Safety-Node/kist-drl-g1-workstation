"""
Speak Connector [TASK-45, REQ-29]

Routes SpeakInput.action (text) to the TTS Provider for synthesis. No
routing decision — single backend.

Threading + error policy:
  - connect() is async per OM1 ActionConnector contract.
  - Caller is TaskSrvProvider._schedule_coro — **fire-and-forget**.
    Exceptions raised here turn into "Task exception was never retrieved"
    warnings and disappear; implementation MUST try/except + log + swallow.

TODO(REQ-29) [TASK-45]: connect() — await self._tts.synthesize(text);
                        try/except log + swallow per fire-and-forget caller.
TODO(REQ-29) [TASK-45]: stop() lifecycle — track in-flight asyncio tasks,
                        cancel on shutdown. Add Connector to run.py
                        _stop_runtime once stop() actually does something.
TODO(REQ-29) [TASK-45]: E-STOP cancellation — interrupt mid-utterance via
                        TTSProvider.cancel().
"""

import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.tts_provider import TTSProvider


class SpeakConnector(ActionConnector[ActionConfig, SpeakInput]):
    """Forwards SpeakInput.action to the TTS Provider."""

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # CONV-001 ordering: TTSProvider was constructed by run.py before
        # this connector, so the @singleton fetch returns that instance.
        self._tts = TTSProvider()
        logging.info("SpeakConnector: skeleton initialized")

    async def connect(self, output_interface: SpeakInput) -> None:
        # TODO(REQ-29) [TASK-45]: try/except — log + swallow, NEVER re-raise
        # TODO(REQ-29) [TASK-45]: await self._tts.synthesize(output_interface.action)
        raise NotImplementedError("SpeakConnector.connect: TBD [TASK-45]")
