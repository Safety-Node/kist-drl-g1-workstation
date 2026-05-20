"""
STT Provider (Google Cloud STT) -- KIST DRL G1 Workstation
==========================================================

drawio C4 Container:
    Name        : STT Provider
    Technology  : Google STT
    Description : Streams audio to cloud STT, send transcript text.

Edges (per docs/structure/container_option3.drawio):
    -> Sound Sensor : Transcribed text [text]
    <-> Google Cloud STT : Audio <-> Transcribed text [HTTPS / gRPC]

TBD:
    - Bind to google-cloud-speech v2 streaming API
    - Push-to-talk vs always-on toggle
    - Korean (ko-KR) language code wiring + fallback to en-US
    - Backpressure when Sound Sensor consumer is slow
    - Retry/circuit-breaker on cloud STT outage
    - Safety: PII handling, on-prem fallback path
"""

import logging
from typing import Callable, Optional

from .singleton import singleton


@singleton
class GoogleSTTProvider:
    """
    Workstation-side Google Cloud Speech-to-Text streaming provider.

    Streams microphone audio (PCM) to Google Cloud STT and emits transcript
    strings to a registered callback. Replaces / parallels the generic
    ``ASRProvider`` for the KIST DRL G1 setup.
    """

    def __init__(
        self,
        language_code: str = "ko-KR",
        sample_rate_hz: int = 16000,
        device_id: Optional[int] = None,
        gcp_credentials_path: Optional[str] = None,
    ):
        """
        Initialize the Google STT provider.

        Parameters
        ----------
        language_code : str
            BCP-47 language tag. Default "ko-KR".
        sample_rate_hz : int
            Microphone sample rate.
        device_id : int, optional
            PortAudio device index.
        gcp_credentials_path : str, optional
            Path to GCP service-account JSON.
        """
        # TODO: validate GCP credentials are reachable
        # TODO: create AudioInputStream consistent with om1_speech
        self._language_code = language_code
        self._sample_rate_hz = sample_rate_hz
        self._device_id = device_id
        self._gcp_credentials_path = gcp_credentials_path
        self._on_transcript: Optional[Callable[[str], None]] = None
        self._running = False
        logging.info("GoogleSTTProvider: skeleton initialized (lang=%s)", language_code)

    def register_transcript_callback(self, callback: Callable[[str], None]) -> None:
        """Register the callback that receives transcript strings."""
        # TODO: thread-safe registration
        self._on_transcript = callback

    def start(self) -> None:
        """Start streaming audio to Google Cloud STT."""
        # TODO: open AudioInputStream, open gRPC bidi stream, spawn worker
        self._running = True
        raise NotImplementedError("GoogleSTTProvider.start: TBD")

    def stop(self) -> None:
        """Stop streaming and release resources."""
        # TODO: close gRPC stream, stop audio thread, flush
        self._running = False
        raise NotImplementedError("GoogleSTTProvider.stop: TBD")
