"""
Sound Sensor [TASK-46, REQ-44].

Bridges STTProvider transcripts into TaskSrvProvider's keyword router (no
Cortex prompt block anymore). Not a @singleton — one instance per mode.

Subclasses OM1 ``FuserInput`` so it slots into ``mode_config.json5: agent_inputs[]``
without framework changes. ``_poll`` / ``_raw_to_text`` are inert stubs — the
live path is the STT callback fanning out to TaskSrvProvider.
"""

import logging
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.stt_provider import STTProvider, TranscriptEvent
from providers.task_srv_provider import TaskSrvProvider


class SoundSensorConfig(SensorConfig):
    """Configuration for the Sound Sensor."""

    min_confidence: float = Field(
        default=0.3,
        description=(
            "Minimum STT confidence score to keep. Transcripts without a "
            "score pass unconditionally. NOTE: 0.3 is permissive; tune "
            "upward (0.5-0.7) after observing the actual STT confidence "
            "distribution on the demo mic — Korean + ambient noise can "
            "produce false-trigger near-homophones (e.g. 오이 ↔ 오리)."
        ),
    )
    dedupe_window_s: float = Field(
        default=1.5,
        description=(
            "Drop a transcript if an identical one arrived within this many "
            "seconds (Google STT sometimes emits duplicate finals)."
        ),
    )


class SoundSensor(FuserInput[SoundSensorConfig, str]):
    """
    Bridges STT Provider transcripts into TaskSrvProvider keyword routing.

    Not a singleton. ``run.py`` constructs one instance and calls
    ``start()``. ``stop()`` unwires the STT callback.

    Both deps (STTProvider, TaskSrvProvider) are @singletons;
    they are fetched in ``__init__``. ``run.py`` MUST construct both
    Providers before this SoundSensor.

    **Lifecycle discipline**: always call ``stop()`` before re-creating an
    instance. STT's transcript callback list holds bound methods — an
    abandoned SoundSensor without ``stop()`` leaves a dead bound method on
    the STT subscriber list, so transcripts fan out to both the dead and
    the new instance.

    The OM1 FuserInput interface (``_poll`` / ``_raw_to_text``) is
    retained as a no-op so the class still slots into the existing
    ``mode_config.json5`` ``inputs:`` plumbing without invasive
    framework changes; both methods raise ``NotImplementedError``
    because the primary path is the STT callback, not Cortex polling.
    """

    def __init__(self, config: SoundSensorConfig):
        super().__init__(config)
        # Dedupe needs only the last event, not a ring buffer; if a GUI
        # transcript-history panel later wants N events, add buffer back
        # then (YAGNI for now).
        self._last_event: Optional[TranscriptEvent] = None
        # Deps are @singletons. run.py MUST have constructed
        # both Providers before this SoundSensor.
        self._stt = STTProvider()
        self._task_srv = TaskSrvProvider()
        self._started = False
        logging.info(
            "SoundSensor: skeleton initialized (min_conf=%.2f, dedupe_window=%.1fs)",
            config.min_confidence,
            config.dedupe_window_s,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Register transcript callback with STT Provider, ready buffer."""
        self._stt.register_transcript_callback(self.on_transcript)
        self._started = True
        logging.info(
            "SoundSensor: started (min_conf=%.2f, dedupe=%.1fs)",
            self.config.min_confidence,
            self.config.dedupe_window_s,
        )

    def stop(self) -> None:
        """Unregister STT callback, clear last-event."""
        self._stt.unregister_transcript_callback(self.on_transcript)
        self._last_event = None
        self._started = False
        logging.info("SoundSensor: stopped")

    @property
    def started(self) -> bool:
        """True while the STT callback is registered (GUI status display)."""
        return self._started

    # ------------------------------------------------------------------
    # STT → TaskSrvProvider bridge (primary path)
    # ------------------------------------------------------------------
    def on_transcript(self, event: TranscriptEvent) -> None:
        """
        Callback fired by STT Provider for each transcript.

        Threading note: runs on the STT backend's response thread (gRPC
        worker). ``task_srv.on_audio`` enqueues into the inbound queue so
        all state mutation happens on the TaskSrvBg thread (T3 pattern).

        ``event.is_final`` filtering is STT-side responsibility
        (STTConfig.interim_results) — we trust the event here.

        Filter chain:
            1. Drop if confidence is not None and < min_confidence.
            2. Strip + drop empty text.
            3. Dedupe vs _last_event within dedupe_window_s.
            4. Update _last_event.
            5. Forward to TaskSrvProvider.on_audio.
        """
        # 1. Confidence filter (missing score passes unconditionally)
        if event.confidence is not None and event.confidence < self.config.min_confidence:
            logging.info(
                "SoundSensor: drop[confidence] score=%.2f < min=%.2f text=%r",
                event.confidence,
                self.config.min_confidence,
                event.text,
            )
            return

        # 2. Empty text
        text = event.text.strip()
        if not text:
            logging.info("SoundSensor: drop[empty]")
            return

        # 3. Dedupe
        if (
            self._last_event is not None
            and self._last_event.text.strip() == text
            and event.ts - self._last_event.ts < self.config.dedupe_window_s
        ):
            logging.debug(
                "SoundSensor: drop[dedupe] %r (delta=%.2fs)",
                text,
                event.ts - self._last_event.ts,
            )
            return

        # 4. Update last event
        self._last_event = event

        # 5. Forward to TaskSrvProvider keyword router
        self._task_srv.on_audio(event.text, event.ts)

    # ------------------------------------------------------------------
    # OM1 FuserInput interface (legacy / Cortex path -- not used by KIST mode)
    # ------------------------------------------------------------------
    async def _poll(self) -> str:
        """Cortex-polling path not used; TaskSrvProvider receives via callback."""
        raise NotImplementedError(
            "SoundSensor._poll: KIST mode routes via on_transcript callback; "
            "FuserInput poll path is unused."
        )

    async def _raw_to_text(self, raw_input: str) -> Message:
        """Cortex prompt-block formatter not used (no Cortex)."""
        raise NotImplementedError(
            "SoundSensor._raw_to_text: Cortex prompt block deprecated; "
            "audio context is pushed to TaskSrvProvider directly."
        )
