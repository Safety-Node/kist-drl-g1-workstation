"""
Vision Sensor -- KIST DRL G1 Workstation
========================================

drawio C4 Container:
    Name        : Vision Sensor
    Technology  : Python
    Description : Formats visual context for Cortex prompt.

Edges:
    VLM Provider -> Vision Sensor : Scene Description [text]
    Vision Sensor -> Cortex       : Visual context [text]

TBD:
    - Aggregate N most recent VLM descriptions with timestamps
    - De-duplicate near-identical scene strings
    - Stable structuring of objects / spatial relations
    - Format into Cortex prompt block:
        "<Visual context>
           t=...   [fridge open, lettuce on shelf 2, ...]
         </Visual context>"
    - Latency budget: < 1 s end-to-end (VLM hop dominates)
"""

import logging
from typing import List

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput


class VisionSensorConfig(SensorConfig):
    """Configuration for the Vision Sensor."""

    buffer_size: int = Field(default=3, description="Number of scene descriptions to retain")
    dedup_window_s: float = Field(default=2.0, description="Drop near-duplicate descriptions within this window")


class VisionSensor(FuserInput[VisionSensorConfig, str]):
    """
    Fuser input that converts VLM scene descriptions into a Cortex-ready
    "Visual context" block.
    """

    def __init__(self, config: VisionSensorConfig):
        super().__init__(config)
        # TODO: subscribe to VLMCosmosProvider.describe outputs
        # TODO: maintain dedup buffer
        self._buffer: List[str] = []
        logging.info("VisionSensor: skeleton initialized")

    async def _poll(self) -> str:
        """Pull next scene description."""
        # TODO: await new description from queue
        raise NotImplementedError("VisionSensor._poll: TBD")

    async def _raw_to_text(self, raw_input: str) -> Message:
        """Format scene description into a Cortex prompt fragment."""
        # TODO: dedup + ring buffer + framing
        raise NotImplementedError("VisionSensor._raw_to_text: TBD")
