"""
STT Provider [TASK-42, REQ-27]

Vendor-agnostic Speech-to-Text streaming. Default backend: Google Cloud STT v2
streaming. PCM in (16 kHz mono from /bridge/sensors/audio_pcm) → TranscriptEvent
out via callback list (→ Sound Sensor → TaskSrvProvider, CONV-004).

Echo cancellation: drops audio while ``speaker_state.playing == True``, with
``echo_cancel_tail_ms`` tail-off after the flag clears.

TODO(REQ-27) [TASK-42]: backend abstraction; Google bidi gRPC stream.
TODO(REQ-27) [TASK-42]: audio callback from UnitreeG1; speaker_state echo gate.
TODO(REQ-27) [TASK-42]: transcript callback (finals; partials per interim_results).
TODO(REQ-27) [TASK-42]: latency p50 < 500 ms VAD→callback.
TODO(REQ-27) [TASK-42]: stream rotation — Google v2 closes streams after ~5 min;
                        open new stream at ~4 min and stitch transcript context.
TODO(REQ-27) [TASK-42]: silence injection during long TTS — drop-on-flag triggers
                        the Google idle timeout; inject zero-PCM while muted.
TODO(REQ-27) [TASK-42]: leading-edge echo — speaker_state.playing trails actual
                        audio by ~50-100 ms; add echo_cancel_lead_ms or have
                        TTSProvider raise an early flag at synthesize() start.
TODO(REQ-27) [TASK-42]: GOOGLE_APPLICATION_CREDENTIALS env (Google SDK auto).
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from .singleton import singleton
from .unitree_g1_provider import UnitreeG1Provider


class STTBackend(str, Enum):
    """Speech-to-Text backend selector."""

    GOOGLE_CLOUD = "google_cloud"
    # WHISPER = "whisper"
    # RIVA    = "riva"


class STTState(str, Enum):
    """
    Connection state, surfaced via ``STTProvider.state`` for GUI display.

    Transitions:
      start()            : IDLE → CONNECTING → STREAMING
      gRPC stream error  : STREAMING → RECONNECTING → STREAMING
                           (or → FAILED on max-retry exhaustion)
      stop()             : any → IDLE
    """

    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class TranscriptEvent:
    """One transcript line emitted by the STT backend."""

    text: str
    ts: float                            # monotonic seconds
    is_final: bool = True
    confidence: Optional[float] = None   # backend-specific; None if not reported


@dataclass
class STTConfig:
    """STT Provider runtime configuration."""

    backend: STTBackend = STTBackend.GOOGLE_CLOUD
    language_code: str = "ko-KR"
    sample_rate_hz: int = 16000          # matches NX mic_node publish format
    # interim_results filtering is the STT provider's sole responsibility:
    # when False, only is_final=True events are emitted to subscribers
    # (SoundSensor / GUI etc.). Downstream code does not re-filter.
    interim_results: bool = False
    # Drop mic input N ms AFTER speaker_state.playing clears (trailing echo).
    echo_cancel_tail_ms: int = 200
    # Drop mic input N ms BEFORE speaker_state.playing rises (leading edge):
    # the DDS speaker_state hop trails actual audio by ~50-100 ms; bump if
    # the demo shows self-transcribed TTS at utterance starts. Default 0
    # because the conservative value depends on observed LAN latency.
    echo_cancel_lead_ms: int = 0


@singleton
class STTProvider:
    """
    Vendor-agnostic STT streaming. Audio in from UnitreeG1Provider; transcripts
    out to all registered callbacks (multi-subscriber, thread-safe).
    """

    def __init__(self, config: Optional[STTConfig] = None):
        self._config = config or STTConfig()
        self._state = STTState.IDLE   # running == (state != IDLE); no separate flag
        # CONV-010: dep is a @singleton, fetched here. run.py MUST construct
        # UnitreeG1Provider first, otherwise we create it with default config
        # and run.py's later UnitreeG1Provider(...) returns this same instance.
        self._unitree_g1 = UnitreeG1Provider()
        self._callbacks: List[Callable[[TranscriptEvent], None]] = []
        self._callbacks_lock = threading.Lock()
        logging.info(
            "STTProvider: skeleton initialized (backend=%s, lang=%s, rate=%d)",
            self._config.backend.value,
            self._config.language_code,
            self._config.sample_rate_hz,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the STT streaming session and bind to UnitreeG1 audio + speaker_state."""
        # TODO(REQ-27) [TASK-42]: unitree_g1.register_audio_callback(self._on_audio_chunk)
        # TODO(REQ-27) [TASK-42]: open gRPC bidi stream to Google Cloud Speech v2
        # TODO(REQ-27) [TASK-42]: spawn worker (audio push + transcript pull)
        raise NotImplementedError("STTProvider.start: TBD [TASK-42]")

    def stop(self) -> None:
        """Cancel stream, unregister audio callback, drain callbacks."""
        # TODO(REQ-27) [TASK-42]: unitree_g1.unregister_audio_callback(self._on_audio_chunk)
        # TODO(REQ-27) [TASK-42]: close stream, join worker
        raise NotImplementedError("STTProvider.stop: TBD [TASK-42]")

    # ------------------------------------------------------------------
    # Public API — transcript subscribers (multi, thread-safe)
    # ------------------------------------------------------------------
    def register_transcript_callback(
        self, callback: Callable[[TranscriptEvent], None]
    ) -> None:
        """Append ``callback`` to the subscriber list (SoundSensor + GUI etc.)."""
        with self._callbacks_lock:
            if callback in self._callbacks:
                logging.debug(
                    "STTProvider: callback %r already registered, skipping",
                    getattr(callback, "__qualname__", callback),
                )
                return
            self._callbacks.append(callback)

    def unregister_transcript_callback(
        self, callback: Callable[[TranscriptEvent], None]
    ) -> None:
        """Remove ``callback`` from the subscriber list. No-op if not present."""
        with self._callbacks_lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    @property
    def state(self) -> STTState:
        """Connection state, polled by GUI Background for status display."""
        return self._state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _on_audio_chunk(self, pcm: bytes, ts: float) -> None:
        """Audio callback from UnitreeG1; drop while TTS plays (tail-off using ts)."""
        # TODO(REQ-27) [TASK-42]: read unitree_g1.speaker_state.value.playing
        # TODO(REQ-27) [TASK-42]: enforce tail-off window from ts (echo_cancel_tail_ms)
        # TODO(REQ-27) [TASK-42]: forward chunk to streaming_recognize sender queue
        raise NotImplementedError("STTProvider._on_audio_chunk: TBD [TASK-42]")

    def _emit_transcript(self, event: TranscriptEvent) -> None:
        """Fan ``event`` to all registered callbacks; exceptions logged, not raised."""
        with self._callbacks_lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logging.exception("STTProvider: transcript callback raised")
