"""
STT Provider — KIST DRL G1 Workstation [TASK-42]
================================================

Vendor-agnostic Speech-to-Text streaming provider. Default backend:
Google Cloud Speech-to-Text v2 streaming. ``backend`` is selectable via
``STTConfig`` so the implementation can be swapped (Whisper / Riva / ...)
without touching downstream consumers.

drawio C4 Container:
    Name        : STT Provider
    Technology  : Cloud STT (default: Google Cloud Speech-to-Text v2 streaming)
    Description : Streams microphone audio to an STT backend, emits transcripts.

---

## Data flow

    NX mic_node                                       (Onboard)
        ↓ /bridge/sensors/audio_pcm (16kHz/16bit/mono)
    UnitreeG1 Provider                                (PC, TASK-41)
        ↓ register_audio_callback                     ← STT Provider hook
    STT Provider                                      (this class)
        ↓ register_transcript_callback                ← Sound Sensor hook
    Sound Sensor → TaskSrvProvider                    (downstream)

Echo cancellation:
    UnitreeG1 Provider.speaker_state.value.playing == True
    → STT Provider drops incoming audio chunks (with tail-off) so the
    robot does not transcribe its own TTS output.

---

## ICD references

| Direction | Topic / call                                | Note                       |
|-----------|---------------------------------------------|----------------------------|
| inbound   | /bridge/sensors/audio_pcm (via UnitreeG1)   | 16kHz / 16bit / mono PCM   |
| inbound   | /bridge/audio/speaker_state (via UnitreeG1) | echo-cancel hint           |
| outbound  | Google Cloud STT v2 streaming_recognize     | HTTPS / gRPC               |
| outbound  | transcript str (callback)                   | → Sound Sensor             |

Out-of-scope (deprecated per 2026-05-22 user decision):
    - "Audio Context to Cortex" ICD — Cortex(LLM) replaced by TaskSrvProvider.
      Sound Sensor formats audio context for TaskSrvProvider directly now.

---

Vendor reference (friend's original scaffold):
    src/providers/example/google_stt_provider.py  ← Google-specific class name
    This vendor-agnostic file (``stt_provider.py``) is the active replacement.

---

TODO(REQ-27) [TASK-42]: backend abstraction — ``STTBackend`` interface, Google
                        default impl. Keep class name vendor-agnostic so other
                        backends slot in without consumer changes.
TODO(REQ-27) [TASK-42]: streaming open — gRPC bidi to Google Cloud Speech v2.
TODO(REQ-27) [TASK-42]: audio callback — receive PCM chunks from UnitreeG1
                        Provider, push to streaming_recognize sender queue.
TODO(REQ-27) [TASK-42]: speaker_state echo cancel — drop / mute audio while
                        TTS is playing, with ``echo_cancel_tail_ms`` tail-off.
TODO(REQ-27) [TASK-42]: transcript callback — emit final transcripts to the
                        registered Sound Sensor; partial transcripts optional.
TODO(REQ-27) [TASK-42]: latency budget — p50 < 500ms from VAD endpoint to
                        transcript callback (Notion Task verification step).
TODO(REQ-27) [TASK-42]: reconnect / retry on cloud STT outage (exponential
                        backoff, surface state via a property for GUI).
TODO(REQ-27) [TASK-42]: API key via env (``GOOGLE_APPLICATION_CREDENTIALS``).
TODO(REQ-27) [TASK-42]: unit/integration tests under tests/providers/.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .singleton import singleton


class STTBackend(str, Enum):
    """Speech-to-Text backend selector."""

    GOOGLE_CLOUD = "google_cloud"
    # WHISPER = "whisper"   # local OpenAI Whisper (future)
    # RIVA    = "riva"      # NVIDIA Riva (future)


@dataclass
class STTConfig:
    """STT Provider runtime configuration."""

    backend: STTBackend = STTBackend.GOOGLE_CLOUD
    language_code: str = "ko-KR"        # KIST KAPEX speaks Korean
    sample_rate_hz: int = 16000         # matches NX mic_node publish format
    interim_results: bool = False        # final transcripts only by default
    echo_cancel_tail_ms: int = 200       # ignore audio N ms after speaker stops
    google_credentials_env: str = "GOOGLE_APPLICATION_CREDENTIALS"


@singleton
class STTProvider:
    """
    Vendor-agnostic Speech-to-Text streaming provider.

    Default backend: Google Cloud Speech-to-Text v2 streaming. Subscribes to
    microphone audio via the UnitreeG1 Provider audio callback, streams PCM
    chunks to the cloud, and emits transcript strings through a registered
    callback (consumed by Sound Sensor → TaskSrvProvider).
    """

    def __init__(self, config: Optional[STTConfig] = None):
        """
        Parameters
        ----------
        config : STTConfig, optional
            Runtime configuration. Defaults to Google Cloud, ko-KR, 16kHz.
        """
        self._config = config or STTConfig()
        self._running = False
        self._on_transcript: Optional[Callable[[str], None]] = None
        # Bound late in start() so the UnitreeG1 Provider singleton is up.
        self._unitree_g1 = None
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
        """Open the STT streaming session and bind to UnitreeG1 Provider audio."""
        # TODO(REQ-27) [TASK-42]: resolve UnitreeG1 Provider singleton
        # TODO(REQ-27) [TASK-42]: register audio callback for /bridge/sensors/audio_pcm
        # TODO(REQ-27) [TASK-42]: open gRPC bidi stream to Google Cloud Speech v2
        # TODO(REQ-27) [TASK-42]: spawn worker thread (audio push + transcript pull)
        raise NotImplementedError("STTProvider.start: TBD [TASK-42]")

    def stop(self) -> None:
        """Close the streaming session, unregister callbacks, join worker."""
        # TODO(REQ-27) [TASK-42]: cancel stream, join worker, unregister callbacks
        raise NotImplementedError("STTProvider.stop: TBD [TASK-42]")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_transcript_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register the function called with each final transcript string.
        Typical consumer: Sound Sensor (forwards to TaskSrvProvider keyword
        matcher).
        """
        # TODO(REQ-27) [TASK-42]: thread-safe registration (drop-on-replace policy)
        self._on_transcript = callback

    # ------------------------------------------------------------------
    # Internals — audio chunk handling with echo cancel
    # ------------------------------------------------------------------
    def _on_audio_chunk(self, pcm: bytes) -> None:
        """
        Internal callback fed by the UnitreeG1 Provider audio subscriber.
        Drops chunks while TTS is playing (with tail-off window).
        """
        # TODO(REQ-27) [TASK-42]: poll unitree_g1.speaker_state.value.playing
        # TODO(REQ-27) [TASK-42]: enforce tail-off (echo_cancel_tail_ms)
        # TODO(REQ-27) [TASK-42]: forward chunk to streaming_recognize sender queue
        raise NotImplementedError("STTProvider._on_audio_chunk: TBD [TASK-42]")
