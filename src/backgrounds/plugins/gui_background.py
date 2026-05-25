"""
GUI Background [TASK-47]

Streams composite camera + task-status frames to the wall-mounted Display
System. Transport TBD (WebRTC / RTSP / MJPEG / RTMP).
"""

import logging

from backgrounds.base import Background, BackgroundConfig


class GUIBackground(Background[BackgroundConfig]):
    """Pushes the live composite stream to the Display System."""

    def __init__(self, config: BackgroundConfig):
        super().__init__(config)
        logging.info("GUIBackground: skeleton initialized")

    async def start(self) -> None:
        raise NotImplementedError("GUIBackground.start: TBD")

    async def stop(self) -> None:
        raise NotImplementedError("GUIBackground.stop: TBD")

    async def tick(self) -> None:
        raise NotImplementedError("GUIBackground.tick: TBD")
