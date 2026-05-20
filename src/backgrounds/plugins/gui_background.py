"""
GUI BackGround -- KIST DRL G1 Workstation
=========================================

drawio C4 Container:
    Name        : GUI BackGround
    Technology  : Python
    Description : Streams Video, Task Status to Display System.

Edges:
    IOProvider          -> GUI BackGround : Task / state aggregate
    UnitreeG1 Provider  -> GUI BackGround : Video Stream + Nav/Buf State [Image / RTSP]
    GUI BackGround      -> Display System : Video + Task status [Streaming]

TBD:
    - Pick the streaming transport: WebRTC vs RTSP vs MJPEG vs RTMP
    - Compose overlay: camera grid + KAPEX status pane + G1 task pane
    - Latency budget: <= 200 ms end-to-end to wall display
    - Reconnect / supervisor restart policy
    - Recording hook for post-hoc safety review
"""

import logging

from backgrounds.base import Background, BackgroundConfig


class GUIBackground(Background[BackgroundConfig]):
    """
    Background that pushes the live camera + task-status composite stream
    to the wall-mounted Display System.

    The actual rendering / encoding pipeline is intentionally TBD: the
    skeleton fixes the lifecycle hooks (``start`` / ``stop`` / ``tick``)
    so the runtime/manager can register it.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Parameters
        ----------
        config : BackgroundConfig
            Background configuration. Expected fields (TBD):
                - display_url : str
                - bitrate_kbps : int
                - layout : str
        """
        super().__init__(config)
        # TODO: build encoder pipeline (GStreamer / ffmpeg / aiortc)
        # TODO: subscribe to IOProvider for task status
        logging.info("GUIBackground: skeleton initialized")

    async def start(self) -> None:
        """Bring up the streamer."""
        # TODO: open camera subscriber, start encoder, open display socket
        raise NotImplementedError("GUIBackground.start: TBD")

    async def stop(self) -> None:
        """Tear down the streamer."""
        # TODO: flush encoder, close socket
        raise NotImplementedError("GUIBackground.stop: TBD")

    async def tick(self) -> None:
        """One streaming cycle (composition + encode + push)."""
        # TODO: pull latest frame + status, render overlay, push to display
        raise NotImplementedError("GUIBackground.tick: TBD")
