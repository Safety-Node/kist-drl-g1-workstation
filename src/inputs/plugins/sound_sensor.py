"""
Sound Sensor [TASK-46, REQ-44]

Bridges STTProvider transcripts into TaskSrvProvider's keyword router (CONV-004:
no Cortex prompt block anymore). Not a @singleton — one instance per mode.

Subclasses OM1 ``FuserInput`` so it slots into ``mode_config.json5: agent_inputs[]``
without framework changes. ``_poll`` / ``_raw_to_text`` are inert stubs — the
live path is the STT callback fanning out to TaskSrvProvider.

TODO(REQ-44) [TASK-46]: bind(stt, task_srv) + start() registers STT callback.
TODO(REQ-44) [TASK-46]: on_transcript — dedupe + confidence filter + dispatch.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput


@dataclass
class TranscriptEvent:
    """One transcript line with arrival timestamp (monotonic seconds)."""

    text: str
    ts: float


class SoundSensorConfig(SensorConfig):
    """Configuration for the Sound Sensor."""

    buffer_size: int = Field(
        default=5,
        description="Number of recent transcripts to retain in the ring buffer.",
    )
    min_confidence: float = Field(
        default=0.3,
        description=(
            "Minimum STT confidence score to keep. Note: confidence is "
            "optional in the STT Provider callback; transcripts without a "
            "score are accepted unconditionally."
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

    Lifecycle: not a singleton. ``run.py`` constructs one instance, calls
    ``bind(stt=..., task_srv=...)``, then ``start()``. ``stop()`` unwires
    the STT callback. Tests can construct independent instances freely.

    The OM1 FuserInput interface (``_poll`` / ``_raw_to_text``) is
    retained as a no-op so the class still slots into the existing
    ``mode_config.json5`` ``inputs:`` plumbing without invasive
    framework changes; both methods raise ``NotImplementedError``
    because the primary path is the STT callback, not Cortex polling.
    """

    def __init__(self, config: SoundSensorConfig):
        super().__init__(config)
        self._buffer: Deque[TranscriptEvent] = deque(maxlen=config.buffer_size)
        self._stt = None        # set by bind()
        self._task_srv = None   # set by bind()
        self._started = False
        logging.info(
            "SoundSensor: skeleton initialized (buffer=%d, min_conf=%.2f, dedupe_window=%.1fs)",
            config.buffer_size,
            config.min_confidence,
            config.dedupe_window_s,
        )

    # ------------------------------------------------------------------
    # Explicit dependency wiring (CONV-001 Option D)
    # ------------------------------------------------------------------
    def bind(self, stt, task_srv) -> None:
        """
        Wire dependencies after ``run.py`` has started both providers.

        Parameters
        ----------
        stt : STTProvider
            Already-started STT Provider singleton. SoundSensor will call
            ``stt.register_transcript_callback(self.on_transcript)``
            in ``start()``.
        task_srv : TaskSrvProvider
            Already-started TaskSrvProvider singleton. SoundSensor will
            push ``TranscriptEvent``s via ``task_srv.on_audio(text, ts)``
            (exact method name TBD when TaskSrvProvider lands).
        """
        # TODO(REQ-44) [TASK-46]: assert both providers are .started()
        self._stt = stt
        self._task_srv = task_srv

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Register transcript callback with STT Provider, ready buffer."""
        # TODO(REQ-44) [TASK-46]: validate bind() was called
        # TODO(REQ-44) [TASK-46]: self._stt.register_transcript_callback(self.on_transcript)
        # TODO(REQ-44) [TASK-46]: self._started = True
        raise NotImplementedError("SoundSensor.start: TBD [TASK-46]")

    def stop(self) -> None:
        """Unregister STT callback, flush ring buffer."""
        # TODO(REQ-44) [TASK-46]: STT Provider unregister (API TBD)
        # TODO(REQ-44) [TASK-46]: self._buffer.clear()
        # TODO(REQ-44) [TASK-46]: self._started = False
        raise NotImplementedError("SoundSensor.stop: TBD [TASK-46]")

    # ------------------------------------------------------------------
    # STT → TaskSrvProvider bridge (primary path)
    # ------------------------------------------------------------------
    def on_transcript(self, text: str, ts: Optional[float] = None) -> None:
        """
        Callback fired by STT Provider for each final transcript.

        Behaviour (when implemented):
            1. Ignore empty / whitespace-only text.
            2. Drop if identical to the most recent buffered entry and
               within ``dedupe_window_s`` (Google STT duplicate-final guard).
            3. Append to ring buffer.
            4. Forward to TaskSrvProvider for keyword scenario matching.
        """
        # TODO(REQ-44) [TASK-46]: strip + empty check
        # TODO(REQ-44) [TASK-46]: dedupe against self._buffer[-1] within dedupe_window_s
        # TODO(REQ-44) [TASK-46]: self._buffer.append(TranscriptEvent(text, ts))
        # TODO(REQ-44) [TASK-46]: self._task_srv.on_audio(text, ts)
        raise NotImplementedError("SoundSensor.on_transcript: TBD [TASK-46]")

    # ------------------------------------------------------------------
    # OM1 FuserInput interface (legacy / Cortex path -- not used by KIST mode)
    # ------------------------------------------------------------------
    async def _poll(self) -> str:
        """Cortex-polling path not used; TaskSrvProvider receives via callback."""
        raise NotImplementedError(
            "SoundSensor._poll: KIST mode routes via on_transcript callback "
            "(CONV-004); FuserInput poll path is unused."
        )

    async def _raw_to_text(self, raw_input: str) -> Message:
        """Cortex prompt-block formatter not used (CONV-004: no Cortex)."""
        raise NotImplementedError(
            "SoundSensor._raw_to_text: Cortex prompt block deprecated "
            "(CONV-004); audio context is pushed to TaskSrvProvider directly."
        )
