"""
STT Provider [TASK-42, REQ-27]

Vendor-agnostic Speech-to-Text streaming. Default backend: Google Cloud STT v2
streaming. PCM in (16 kHz mono from /bridge/sensors/audio_pcm) → transcript str
out via callback (→ Sound Sensor → TaskSrvProvider, CONV-004).

Echo cancellation: drops audio while ``speaker_state.playing == True``, with
``echo_cancel_tail_ms`` tail-off after the flag clears.

TODO(REQ-27) [TASK-42]: backend abstraction; Google bidi gRPC stream.
TODO(REQ-27) [TASK-42]: audio callback from UnitreeG1; speaker_state echo gate.
TODO(REQ-27) [TASK-42]: transcript callback (finals; partials optional).
TODO(REQ-27) [TASK-42]: latency p50 < 500 ms VAD→callback.
TODO(REQ-27) [TASK-42]: reconnect + state property for GUI.
TODO(REQ-27) [TASK-42]: GOOGLE_APPLICATION_CREDENTIALS env.
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
