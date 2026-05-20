"""
Sound Sensor -- KIST DRL G1 Workstation
=======================================

drawio C4 Container:
    Name        : Sound Sensor
    Technology  : Python
    Description : Formats audio-related context for Cortex prompt.

Edges:
    STT Provider -> Sound Sensor : Transcribed text [text]
    Sound Sensor -> Cortex       : Audio context [text]

TBD:
    - Buffer N most-recent transcripts with timestamps
    - Speaker-diarization tag (operator vs ambient) -- nice-to-have
    - Wake-word / push-to-talk gating
    - Format into Cortex prompt block:
        "<Audio context>
           [user, 2026-05-20T16:30:11] 양상추 하나 꺼내줘
         </Audio context>"
    - Drop silent / sub-confidence transcripts
"""

import logging
from typing import List

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput


class SoundSensorConfig(SensorConfig):
    """Configuration for the Sound Sensor."""

    # TODO: declare fields (buffer_size, min_confidence, wake_word, ...)
    buffer_size: int = Field(default=5, description="Number of transcripts to retain")
    min_confidence: float = Field(default=0.3, description="Min STT confidence to keep")


class SoundSensor(FuserInput[SoundSensorConfig, str]):
    """
    Fuser input that turns raw STT transcripts into a Cortex-ready
    "Audio context" block.
    """

    def __init__(self, config: SoundSensorConfig):
        super().__init__(config)
        # TODO: subscribe to GoogleSTTProvider transcript callback
        # TODO: maintain ring buffer of recent transcripts
        self._buffer: List[str] = []
        logging.info("SoundSensor: skeleton initialized")

    async def _poll(self) -> str:
        """Pull next transcript event."""
        # TODO: await new transcript from queue
        raise NotImplementedError("SoundSensor._poll: TBD")

    async def _raw_to_text(self, raw_input: str) -> Message:
        """Format the raw transcript into a Cortex prompt fragment."""
        # TODO: append to buffer; format header/footer; trim
        raise NotImplementedError("SoundSensor._raw_to_text: TBD")
