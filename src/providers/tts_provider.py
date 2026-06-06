"""
TTS Provider [TASK-43, REQ-29].

Vendor-agnostic Text-to-Speech. Default backend: Naver Clova Voice Premium REST.

PC resamples Clova native (22050 Hz or 24000 Hz — depends on the chosen
voice id) → 16 kHz mono int16 before publishing to /bridge/cmd/audio_out
(REQ-29 2026-05-15: PC resample responsibility). NX speaker_node consumes
(relayed onboard as /onboard/audio/playback).

Echo-cancel: TTS only publishes — NX speaker_node raises
``speaker_state.playing`` which STT consumes to mute its mic input. No
active coordination needed here.

TASK-41 status: UnitreeG1Provider.register_estop_callback /
publish_audio_out are still NotImplementedError. start() catches the
callback bind (logs WARNING, graceful degrade); synthesize() catches the
publish (logs WARNING, drops the PCM). The synthesis + resample path runs
fully regardless so the cloud round-trip is exercisable today.

Credentials: NCP_CLOVA_CLIENT_ID / NCP_CLOVA_CLIENT_SECRET (env). start()
warns (does not crash) when missing so run.py --scaffold-loop and the
exercise script still work; synthesize() then logs + drops.
"""

import asyncio
import io
import logging
import os
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .singleton import singleton
from .unitree_g1_provider import UnitreeG1Provider


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
    # Clova WAV request rate. Clova Voice Premium WAV accepts
    # [8000, 16000, 24000, 48000]; we request a high-quality native rate and
    # resample to sample_rate_hz on the PC (REQ-29 PC resample responsibility).
    clova_sample_rate_hz: int = 24000
    # Clova voice id (str, not Literal — vendor list is dozens long and
    # changes; see https://api.ncloud-docs.com/docs/ai-naver-clovavoice).
    voice: str = "nara"
    speed: int = 0                        # Clova [-5, +5] speed offset
    naver_api_url: str = (
        "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    )
    client_id_env: str = "NCP_CLOVA_CLIENT_ID"
    client_secret_env: str = "NCP_CLOVA_CLIENT_SECRET"
    # On timeout: log + drop. A missed audio cue is preferable to retry-
    # induced desync with the sub-task flow ("성공했습니다" 5 s late).
    # Diverges from VLA's 2x retry policy on purpose: TTS failure does
    # not fail a sub-task, it just skips the announcement.
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
        # UnitreeG1Provider is @singleton — fetched here.
        # run.py MUST construct UnitreeG1 before TTSProvider so this
        # returns the configured instance, not a default-config singleton.
        self._unitree_g1 = UnitreeG1Provider()
        self._inflight_request = None  # asyncio.Task while synth in progress
        self._inflight_loop: Optional[asyncio.AbstractEventLoop] = None
        # Set True by _on_estop; gates synthesize() / drops new requests.
        self._estop_active = False
        # Tracks whether we successfully registered the E-STOP callback so
        # stop() only unregisters what it bound (TASK-41 may raise).
        self._estop_cb_registered = False
        # Resolved from env in start().
        self._client_id: Optional[str] = None
        self._client_secret: Optional[str] = None
        logging.info(
            "TTSProvider: initialized (backend=%s, lang=%s, voice=%s, rate=%d)",
            self._config.backend.value,
            self._config.language_code,
            self._config.voice,
            self._config.sample_rate_hz,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Read NCP credentials and subscribe to UnitreeG1 E-STOP."""
        self._client_id = os.environ.get(self._config.client_id_env)
        self._client_secret = os.environ.get(self._config.client_secret_env)
        if not self._client_id or not self._client_secret:
            # Graceful degrade rather than fail-fast: keeps run.py --scaffold-loop
            # and exercise_tts.py (no creds) alive. synthesize() logs + drops.
            logging.warning(
                "TTSProvider: %s / %s not set — synthesize() will log + drop "
                "until credentials are present",
                self._config.client_id_env, self._config.client_secret_env,
            )

        # Bind E-STOP push callback — TASK-41 pending; degrade gracefully.
        try:
            self._unitree_g1.register_estop_callback(self._on_estop)
            self._estop_cb_registered = True
        except NotImplementedError:
            logging.warning(
                "TTSProvider: UnitreeG1.register_estop_callback NotImplementedError "
                "(TASK-41 pending) — E-STOP gate inactive; call _on_estop() directly "
                "for testing"
            )

        self._estop_active = False
        self._running = True
        logging.info("TTSProvider: started")

    def stop(self) -> None:
        """Unregister E-STOP callback and cancel any in-flight synthesis."""
        if not self._running:
            return
        if self._estop_cb_registered:
            try:
                self._unitree_g1.unregister_estop_callback(self._on_estop)
            except NotImplementedError:
                pass
            self._estop_cb_registered = False
        self.cancel()
        self._running = False
        logging.info("TTSProvider: stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def synthesize(self, text: str) -> None:
        """
        Synthesize ``text`` and publish the PCM stream to NX (fire-and-forget).

        The caller path is ``SpeakConnector.connect`` → ``_schedule_coro`` →
        here; the asyncio.Task return value is discarded. Hence ``None``
        return — re-add raw bytes later if a real consumer needs them.

        Behaviour: gate on E-STOP / not-running, POST to Clova, decode WAV,
        resample to the wire rate, publish. On timeout / error: log + drop
        (no retry — TTSConfig docstring explains the deliberate divergence
        from VLA's retry policy).
        """
        if not self._running:
            logging.warning("TTSProvider.synthesize: not running, ignoring")
            return
        if self._estop_active:
            logging.info("TTSProvider.synthesize: E-STOP active — aborting (text=%r)", text)
            return
        text = (text or "").strip()
        if not text:
            return
        if not self._client_id or not self._client_secret:
            logging.warning("TTSProvider.synthesize: no credentials — dropping (text=%r)", text)
            return

        self._inflight_request = asyncio.current_task()
        self._inflight_loop = asyncio.get_running_loop()
        try:
            wav_bytes = await self._http_post_clova(text)
            if wav_bytes is None:
                return
            # E-STOP may have fired during the network round-trip.
            if self._estop_active:
                logging.info("TTSProvider.synthesize: E-STOP fired mid-request — dropping audio")
                return
            pcm, src_rate, channels = self._decode_wav(wav_bytes)
            if pcm is None:
                return
            pcm16 = self._resample_to_wire(pcm, src_rate, channels)
            self._publish(pcm16)
        except asyncio.CancelledError:
            logging.info("TTSProvider.synthesize: cancelled (E-STOP / stop)")
            raise
        except Exception:
            # log + drop, no retry (TTSConfig policy).
            logging.exception("TTSProvider.synthesize: failed; dropping (text=%r)", text)
        finally:
            self._inflight_request = None
            self._inflight_loop = None

    def cancel(self) -> None:
        """Cancel any in-flight synthesis (called on E-STOP / stop())."""
        task = self._inflight_request
        loop = self._inflight_loop
        if task is None or loop is None:
            return
        # cancel() may run on the DDS/E-STOP thread, not the loop thread.
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            # loop already closed (e.g. asyncio.run finished) — nothing to do
            pass
        logging.info("TTSProvider: cancel() requested in-flight synthesis abort")

    # ------------------------------------------------------------------
    # Read-only state (polled by GUI BG)
    # ------------------------------------------------------------------
    @property
    def is_synthesizing(self) -> bool:
        """
        True while a synth request is in flight to the cloud TTS.

        For "is the NX speaker actually emitting sound right now?", read
        ``unitree_g1.speaker_state.value.playing`` instead — that flag is
        raised by NX speaker_node based on actual audio playback, not by
        PC-side synthesis status.
        """
        return self._inflight_request is not None

    # ------------------------------------------------------------------
    # E-STOP push callback (registered with UnitreeG1Provider in start())
    # ------------------------------------------------------------------
    def _on_estop(self, active: bool, ts: float) -> None:
        """E-STOP push callback. Cancel current synth; gate future synthesize()."""
        self._estop_active = active
        logging.info(
            "TTSProvider: E-STOP %s (ts=%.3f)", "ACTIVE" if active else "CLEARED", ts
        )
        if active:
            # Cancel in-flight + block new synthesize(). E-STOP clear does NOT
            # auto-resume the killed utterance (a G1 staying audible during
            # E-STOP would confuse the operator) — clearing just re-allows
            # future synthesize() calls.
            self.cancel()

    # ------------------------------------------------------------------
    # Backend: Naver Clova Voice Premium REST
    # ------------------------------------------------------------------
    async def _http_post_clova(self, text: str) -> Optional[bytes]:
        """POST ``text`` to Clova /tts; return raw WAV bytes (or None on failure).

        Isolated so exercise scripts can monkeypatch the network call and
        verify the decode/resample/publish path offline.
        """
        import aiohttp

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self._client_id,
            "X-NCP-APIGW-API-KEY": self._client_secret,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = {
            "speaker": self._config.voice,
            "text": text,
            "format": "wav",
            "speed": self._config.speed,
            "sampling-rate": self._config.clova_sample_rate_hz,
        }
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout_s)
        # Per-request session: TaskSrvProvider._schedule_coro runs each
        # synthesize in its own asyncio.run loop, so a persistent
        # cross-call session is not possible anyway.
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._config.naver_api_url, headers=headers, data=body) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    logging.error(
                        "TTSProvider: Clova HTTP %d — %s", resp.status, detail[:200]
                    )
                    return None
                return await resp.read()

    # ------------------------------------------------------------------
    # WAV decode + resample (PC resample responsibility, REQ-29)
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_wav(wav_bytes: bytes):
        """Return (pcm_int16_bytes, framerate, channels) from a WAV payload."""
        try:
            with io.BytesIO(wav_bytes) as buf, wave.open(buf, "rb") as wf:
                channels = wf.getnchannels()
                framerate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            if sampwidth != 2:
                logging.error("TTSProvider: unexpected WAV sample width %d (need 16-bit)", sampwidth)
                return None, 0, 0
            return frames, framerate, channels
        except Exception:
            logging.exception("TTSProvider: WAV decode failed")
            return None, 0, 0

    def _resample_to_wire(self, pcm: bytes, src_rate: int, channels: int) -> bytes:
        """Downmix to mono and resample ``pcm`` (int16) to ``sample_rate_hz``."""
        samples = np.frombuffer(pcm, dtype=np.int16)
        if channels == 2:
            # Interleaved L/R → mono average
            samples = samples.reshape(-1, 2).mean(axis=1)
        samples = samples.astype(np.float32)

        dst_rate = self._config.sample_rate_hz
        if src_rate == dst_rate or samples.size == 0:
            return samples.astype(np.int16).tobytes()

        n_dst = int(round(samples.size * dst_rate / src_rate))
        if n_dst <= 0:
            return b""
        # Linear interpolation resample (adequate for speech TTS playback).
        x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_dst, endpoint=False)
        resampled = np.interp(x_new, x_old, samples)
        return resampled.astype(np.int16).tobytes()

    def _publish(self, pcm: bytes) -> None:
        """Publish 16 kHz mono int16 PCM to NX; log + drop if TASK-41 pending."""
        try:
            self._unitree_g1.publish_audio_out(pcm)
            logging.info("TTSProvider: published %d bytes to /bridge/cmd/audio_out", len(pcm))
        except NotImplementedError:
            logging.warning(
                "TTSProvider: publish_audio_out NotImplementedError (TASK-41 pending) "
                "— dropping %d bytes (degrade)", len(pcm),
            )
