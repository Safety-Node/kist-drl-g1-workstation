"""
Speak Connector [TASK-45, REQ-29].

Routes SpeakInput.action (text) to the TTS Provider for synthesis. No
routing decision — single backend.

Threading + error policy:
  - connect() is async per OM1 ActionConnector contract.
  - Caller is TaskSrvProvider._schedule_coro — **fire-and-forget**.
    Exceptions raised here turn into "Task exception was never retrieved"
    warnings and disappear; connect() therefore try/except + log + swallow
    (asyncio.CancelledError is re-raised — cooperative cancellation, not a
    synthesis failure).
  - E-STOP cancellation + backpressure are TTSProvider's responsibility
    (TTSProvider.cancel() / synthesize() gate); the connector only forwards.

TODO(REQ-29) [TASK-45]: stop() lifecycle — track in-flight asyncio tasks,
                        cancel on shutdown. Add Connector to run.py
                        _stop_runtime once stop() actually does something.
"""

import asyncio
import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.tts_provider import TTSProvider


class SpeakConnector(ActionConnector[ActionConfig, SpeakInput]):
    """Forwards SpeakInput.action to the TTS Provider."""

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        # Ordering: TTSProvider was constructed by run.py before
        # this connector, so the @singleton fetch returns that instance.
        self._tts = TTSProvider()
        logging.info("SpeakConnector: skeleton initialized")

    async def connect(self, output_interface: SpeakInput) -> None:
        """Forward the text to TTSProvider.synthesize (fire-and-forget).

        The caller (TaskSrvProvider._schedule_coro) discards the asyncio.Task,
        so any exception raised here would surface only as a "Task exception
        was never retrieved" warning and be lost. We therefore try/except +
        log + swallow, NEVER re-raise — a failed announcement must not crash
        the dispatch path. E-STOP interruption and backpressure are the
        TTSProvider's responsibility (cancel() / gate); the connector only
        forwards.

        ``asyncio.CancelledError`` is re-raised: it is cooperative task
        cancellation (stop() / shutdown), not a synthesis failure.
        """
        text = output_interface.action
        try:
            await self._tts.synthesize(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception(
                "SpeakConnector.connect: TTS synthesize failed for %r; "
                "swallowing (fire-and-forget caller)", text,
            )
