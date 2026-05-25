"""
GUI Background [TASK-47]

Composites camera frame + task status + STT/TTS/E-STOP indicators and
streams to the wall-mounted Display System.

Direct polling per CONV-010 (no IOProvider — see CONV-011):
  - UnitreeG1Provider.color / .estop
  - TaskSrvProvider.state / .active_scenario_name / .active_sub_task
  - STTProvider.state
  - TTSProvider.is_synthesizing

Transport TBD: MJPEG over HTTP is the simplest default for a single-LAN
demo; WebRTC if low-latency interaction becomes a requirement.
Latency budget: ≤ 200 ms end-to-end to the wall display.
Recording hook (TBD): mirror composite frames to disk for post-demo
safety review.

TODO(REQ-??) [TASK-47]: pick transport; implement composer + encoder.
TODO(REQ-??) [TASK-47]: define overlay schema (scenario / sub-task / state
                        badge / E-STOP banner / STT state / TTS indicator).
TODO(REQ-??) [TASK-47]: recording sink to disk.
TODO(REQ-??) [TASK-47]: optional STT live transcript via
                        ``stt.register_transcript_callback`` for caption
                        overlay (push, not poll — characters per second).
"""

import logging
import time

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.stt_provider import STTProvider
from providers.task_srv_provider import TaskSrvProvider
from providers.tts_provider import TTSProvider
from providers.unitree_g1_provider import UnitreeG1Provider


class GUIBackgroundConfig(BackgroundConfig):
    """Configuration for the GUI streaming background."""

    fps: float = Field(
        default=15.0,
        gt=0,
        description=(
            "Composite-frame publish rate to the display. 15 Hz matches "
            "VLA chunk emission and is enough for human-perceivable status; "
            "bump for smoother camera video."
        ),
    )
    display_url: str = Field(
        default="http://localhost:8081/stream",
        description="Display System endpoint (transport TBD — see module docstring).",
    )


class GUIBackground(Background[GUIBackgroundConfig]):
    """Composes Provider state into a display stream (camera + overlays)."""

    def __init__(self, config: GUIBackgroundConfig):
        super().__init__(config)
        # CONV-010: providers fetched directly (no IOProvider bridge).
        # run.py constructs all four before GUIBackground.
        self._unitree_g1 = UnitreeG1Provider()
        self._task_srv = TaskSrvProvider()
        self._stt = STTProvider()
        self._tts = TTSProvider()
        logging.info(
            "GUIBackground: skeleton initialized (fps=%.1f, display=%s)",
            config.fps, config.display_url,
        )

    def run(self) -> None:
        """Composite + push loop (matches TaskSrvBg drift-free pacing pattern)."""
        period = 1.0 / float(self.config.fps)
        next_t = time.monotonic()
        logging.info("GUIBackground: stream loop entering (period=%.3fs)", period)
        # TODO(REQ-??) [TASK-47]: open encoder + display socket here.
        # try/finally ensures cleanup runs whether the loop exits via
        # should_stop() OR sleep() interruption — encoder + socket are
        # OS resources (handles, ports), can't leak them on shutdown.
        try:
            while not self.should_stop():
                try:
                    # TODO(REQ-??) [TASK-47]: read all provider state, compose
                    #     overlay onto color frame, push to display socket.
                    pass
                except Exception:
                    logging.exception("GUIBackground: frame tick raised; continuing")
                next_t += period
                dt = next_t - time.monotonic()
                if dt > 0:
                    if not self.sleep(dt):
                        return  # stop_event fired during sleep → finally runs
                else:
                    # Overran the period; reset baseline to avoid tight-loop.
                    logging.warning(
                        "GUIBackground: frame overran by %.1f ms; resetting baseline",
                        -dt * 1000.0,
                    )
                    next_t = time.monotonic()
        finally:
            # TODO(REQ-??) [TASK-47]: close encoder + socket here. Must
            #     execute on both stop-event-during-sleep AND natural
            #     loop exit — that's why it's in finally, not after the
            #     while.
            logging.info("GUIBackground: stream loop exited (cleanup done)")

    def stop(self) -> None:
        """Base handles stop_event; encoder/socket teardown in run() after loop exit."""
        return None
