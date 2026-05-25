"""
TTS Provider [TASK-43, REQ-29]

Vendor-agnostic Text-to-Speech. Default backend: Naver Clova Voice Premium REST.

PC resamples Clova native 22050/24000 Hz → 16 kHz mono int16 before publishing
to /bridge/cmd/audio_out (REQ-29 2026-05-15: PC resample responsibility).
NX speaker_node consumes (relayed onboard as /onboard/audio/playback).

TODO(REQ-29) [TASK-43]: bind(unitree_g1) + start() RuntimeError guard (mirror STT).
TODO(REQ-29) [TASK-43]: synthesize — POST to Clova /tts (X-NCP-APIGW-API-KEY-*).
TODO(REQ-29) [TASK-43]: resample 22050/24000 → 16000 Hz mono int16.
TODO(REQ-29) [TASK-43]: publish via UnitreeG1.publish_audio_out.
TODO(REQ-29) [TASK-43]: cancel() interrupts in-flight on E-STOP.
TODO(REQ-29) [TASK-43]: sentence-segment streaming (TTFB target < 600 ms).
TODO(REQ-29) [TASK-43]: env keys NCP_CLOVA_CLIENT_ID / NCP_CLOVA_CLIENT_SECRET.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .singleton import singleton


class TTSBackend(str, Enum):
    """Text-to-Speech backend selector."""

    NAVER_CLOVA = "naver_clova"
    # KOKORO     = "kokoro"      # local Kokoro fallback (future)
    # ETRI       = "etri"        # ETRI Korean TTS (future)


@dataclass
class TTSConfig:
    """TTS Provider runtime configuration."""

    backend: TTSBackend = TTSBackend.NAVER_CLOVA
    language_code: str = "ko-KR"          # KIST demo speaks Korean
    sample_rate_hz: int = 16000           # wire format: matches NX mic_node
    voice: str = "nara"                   # Clova voice id
    speed: int = 0                        # Clova [-5, +5] speed offset
    naver_api_url: str = (
        "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    )
    client_id_env: str = "NCP_CLOVA_CLIENT_ID"
    client_secret_env: str = "NCP_CLOVA_CLIENT_SECRET"
    request_timeout_s: float = 5.0
    sentence_streaming: bool = False      # split long text on sentence ends


@singleton
class TTSProvider:
    """
    Vendor-agnostic Text-to-Speech provider.

    Default backend: Naver Clova Voice Premium (REST). Accepts Korean text
    from the Speak Connector, synthesizes via the cloud TTS API, resamples
    to the NX-mic wire format (16 kHz mono int16), and publishes the PCM
    stream through the UnitreeG1 Provider so the NX speaker_node plays it.
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Parameters
        ----------
        config : TTSConfig, optional
            Runtime configuration. Defaults to Naver Clova, ko-KR, 16 kHz.
        """
        self._config = config or TTSConfig()
        self._running = False
        # Bound late in start() so the UnitreeG1 Provider singleton is up.
        self._unitree_g1 = None
        self._inflight_request = None  # cancellation handle
        logging.info(
            "TTSProvider: skeleton initialized (backend=%s, lang=%s, voice=%s, rate=%d)",
            self._config.backend.value,
            self._config.language_code,
            self._config.voice,
            self._config.sample_rate_hz,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the HTTP session and bind to UnitreeG1 Provider audio out."""
        # TODO(REQ-29) [TASK-43]: resolve UnitreeG1 Provider singleton
        # TODO(REQ-29) [TASK-43]: read NCP credentials from env, fail-fast
        # TODO(REQ-29) [TASK-43]: open persistent aiohttp session w/ timeout
        raise NotImplementedError("TTSProvider.start: TBD [TASK-43]")

    def stop(self) -> None:
        """Close the HTTP session, cancel any in-flight synthesis."""
        # TODO(REQ-29) [TASK-43]: cancel in-flight request, close session
        raise NotImplementedError("TTSProvider.stop: TBD [TASK-43]")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize ``text`` into PCM and push to the NX speaker.

        Returns the synthesized PCM bytes (16 kHz mono int16) for callers
        that want the raw audio (tests, logging). The provider also
        publishes the bytes to the UnitreeG1 Provider audio-out path so
        the NX speaker_node consumes them — callers do NOT need to
        re-publish.

        Parameters
        ----------
        text : str
            UTF-8 input text (Korean preferred).

        Returns
        -------
        bytes
            16 kHz / 16-bit / mono PCM.
        """
        # TODO(REQ-29) [TASK-43]: build POST request to Clova /tts
        # TODO(REQ-29) [TASK-43]: stream response, decode MP3 → PCM if needed
        # TODO(REQ-29) [TASK-43]: resample 22050/24000 → 16000 Hz mono
        # TODO(REQ-29) [TASK-43]: push to UnitreeG1 Provider audio-out path
        raise NotImplementedError("TTSProvider.synthesize: TBD [TASK-43]")

    def cancel(self) -> None:
        """Cancel any in-flight synthesis (called on E-STOP)."""
        # TODO(REQ-29) [TASK-43]: abort aiohttp request, drain queued PCM
        raise NotImplementedError("TTSProvider.cancel: TBD [TASK-43]")
