"""
Sound Sensor -- KIST DRL G1 Workstation [TASK-46]
=================================================

Bridges STT Provider transcripts into TaskSrvProvider's keyword router.

Vendor-agnostic: imports the **`STTProvider`** (vendor-agnostic name,
see CONV-003) rather than any Google/Whisper/Riva-specific class. The
backend selection lives in STT Provider's runtime config.

drawio C4 Container:
    Name        : Sound Sensor
    Technology  : Python
    Description : Formats audio-related context for TaskSrvProvider routing.

---

## Data flow (post 2026-05-22 KIST decision, see CONV-004)

    STT Provider                                       (PC, TASK-42)
        ↓ register_transcript_callback                 ← Sound Sensor hook
    Sound Sensor                                       (this class)
        ↓ on_transcript(text, ts)
    TaskSrvProvider                                    (PC, TBD)
        - keyword match → trigger scenario sub-task script

Previously the Sound Sensor formatted an `<Audio context>` block for
the OM1 LLM Cortex prompt. Cortex/VLM/Safety are deferred per
SYS-REQ `[DEPRECATED 2026-05-24]` and replaced by TaskSrvProvider;
the prompt-block formatter is gone. Audio context now lives as a
typed `TranscriptEvent` pushed to TaskSrvProvider.

---

## Lifecycle (per CONV-001 Option D — explicit init in run.py)

    run.py:
        stt = STTProvider(config=...).start()
        task_srv = TaskSrvProvider(config=...).start()
        sound_sensor = SoundSensor(config=...)        # plain object, not singleton
        sound_sensor.bind(stt=stt, task_srv=task_srv) # explicit dependency wiring
        sound_sensor.start()                          # registers STT callback

Sound Sensor is **not** a `@singleton` -- multiple modes can choose to
include/exclude it via mode_config.json5, and singleton-style sharing
across configurations would muddy that.

---

## OM1 FuserInput compatibility

The class still subclasses `FuserInput[SoundSensorConfig, str]` so it
slots into the existing OM1 `inputs:` plumbing in `mode_config.json5`
without invasive framework changes. The `_poll()` / `_raw_to_text()`
methods are NotImplementedError stubs -- they are not the primary
path. The primary path is the STT callback fanning out to
TaskSrvProvider; the FuserInput interface remains as a no-op hook for
any future Cortex-style consumer.

---

TODO(REQ-NEW-TASKSRV) [TASK-46]: bind() -- accept STTProvider + TaskSrvProvider
                                 singletons resolved by run.py. Validate both
                                 are .started() before registering callback.
TODO(REQ-NEW-TASKSRV) [TASK-46]: start() -- register on_transcript callback with
                                 STT Provider, init recent-transcript ring buffer.
TODO(REQ-NEW-TASKSRV) [TASK-46]: stop() -- unregister STT callback, flush buffer.
TODO(REQ-NEW-TASKSRV) [TASK-46]: on_transcript(text, ts) -- ring-buffer append,
                                 confidence filter, dedupe, dispatch to
                                 TaskSrvProvider.on_audio(text, ts).
TODO(REQ-NEW-TASKSRV) [TASK-46]: tests under tests/inputs/plugins/test_sound_sensor.py
                                 (stub STT Provider + stub TaskSrvProvider).
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
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: assert both providers are .started()
        self._stt = stt
        self._task_srv = task_srv

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Register transcript callback with STT Provider, ready buffer."""
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: validate bind() was called
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: self._stt.register_transcript_callback(self.on_transcript)
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: self._started = True
        raise NotImplementedError("SoundSensor.start: TBD [TASK-46]")

    def stop(self) -> None:
        """Unregister STT callback, flush ring buffer."""
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: STT Provider unregister (API TBD)
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: self._buffer.clear()
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: self._started = False
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
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: strip + empty check
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: dedupe against self._buffer[-1] within dedupe_window_s
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: self._buffer.append(TranscriptEvent(text, ts))
        # TODO(REQ-NEW-TASKSRV) [TASK-46]: self._task_srv.on_audio(text, ts)
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
