"""
STT Provider [TASK-42, REQ-27].

Vendor-agnostic Speech-to-Text streaming. Default backend: Google Cloud STT v1
bidi streaming. PCM in (16 kHz mono from /bridge/sensors/audio_pcm via
UnitreeG1Provider push callback) → TranscriptEvent out via callback list
(→ Sound Sensor → TaskSrvProvider).

Echo cancellation: drops audio while ``speaker_state.playing == True``, with
``echo_cancel_tail_ms`` tail-off after the flag clears.  Silent PCM of equal
length is injected instead to keep the Google idle timeout from killing the
stream.

notify_tts_onset() allows TTSProvider to pre-mute the mic by
``echo_cancel_lead_ms`` before actual audio hits the speaker (DDS
speaker_state hop trails real audio by ~50-100 ms).

TASK-41 status: UnitreeG1Provider.register_audio_callback /
register_estop_callback are still NotImplementedError.  start() catches
these and logs a WARNING; the provider remains functional (DUMMY
backend exercises the filter chain without live mic; GOOGLE_CLOUD will
use the same path once TASK-41 lands).
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import numpy as np
from scipy.signal import butter, sosfilt

from .singleton import singleton
from .unitree_g1_provider import UnitreeG1Provider

_MAX_RECONNECT = 10


class StreamingSpeechFilter:
    """청크 경계를 넘어 zi 상태를 유지하는 실시간 IIR 대역 필터.

    HPF(120 Hz, Butterworth 4차) + LPF(3800 Hz, Butterworth 6차) 직렬 구성.
    저주파 진동·DC 및 4 kHz 이상 광대역 노이즈를 제거하고 음성 대역을 보존한다.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        highpass_hz: float = 120.0,
        lowpass_hz: float = 3800.0,
    ) -> None:
        hp_sos = butter(4, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        lp_sos = butter(6, lowpass_hz,  btype="lowpass",  fs=sample_rate, output="sos")
        self._sos = np.vstack([hp_sos, lp_sos])
        self._zi  = np.zeros((self._sos.shape[0], 2), dtype=np.float64)

    def process(self, samples: np.ndarray, gain_linear: float = 1.0) -> np.ndarray:
        y, self._zi = sosfilt(self._sos, samples.astype(np.float64), zi=self._zi)
        return np.clip(np.rint(y * gain_linear), -32768, 32767).astype(np.int16)


class STTBackend(str, Enum):
    """Speech-to-Text backend selector."""

    GOOGLE_CLOUD = "google_cloud"
    DUMMY = "dummy"       # local verification; no credentials needed
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
    # Speech band IIR filter (HPF + LPF). Set to 0.0 to disable.
    highpass_hz: float = 120.0   # Hz — removes low-freq vibration / DC
    lowpass_hz: float  = 5500.0  # Hz — preserves Korean fricatives (ㅅ/ㅎ/ㅊ up to ~6kHz)
    # Input gain applied after filtering. Positive = amplify quiet mic.
    # +6 dB ≈ ×2 amplitude, +12 dB ≈ ×4. Clips to int16 range.
    input_gain_db: float = 0.0   # gain 없음 — 노이즈 환경에서 증폭은 역효과
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
        # Dep is a @singleton, fetched here. run.py MUST construct
        # UnitreeG1Provider first, otherwise we create it with default config
        # and run.py's later UnitreeG1Provider(...) returns this same instance.
        self._unitree_g1 = UnitreeG1Provider()
        self._callbacks: List[Callable[[TranscriptEvent], None]] = []
        self._callbacks_lock = threading.Lock()

        # Echo-cancel state (GIL-protected; each field is a single assignment)
        self._estop_active: bool = False
        self._echo_tail_end: Optional[float] = None   # monotonic tail deadline
        self._lead_mute_end: Optional[float] = None   # monotonic lead deadline

        # Speech band filter (HPF + LPF) — applied to every real audio chunk
        self._speech_filter = StreamingSpeechFilter(
            sample_rate=self._config.sample_rate_hz,
            highpass_hz=self._config.highpass_hz,
            lowpass_hz=self._config.lowpass_hz,
        )
        self._gain_linear = 10.0 ** (self._config.input_gain_db / 20.0)

        # Backend worker state
        self._audio_queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track which UnitreeG1 callbacks we successfully registered so we
        # only unregister the ones we bound (TASK-41 may raise NotImplementedError).
        self._audio_cb_registered: bool = False
        self._estop_cb_registered: bool = False

        logging.info(
            "STTProvider: initialized (backend=%s, lang=%s, rate=%d)",
            self._config.backend.value,
            self._config.language_code,
            self._config.sample_rate_hz,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open the STT streaming session and bind to UnitreeG1 audio + speaker_state."""
        if self._state != STTState.IDLE:
            logging.warning("STTProvider.start: already started (state=%s)", self._state.value)
            return

        self._state = STTState.CONNECTING
        self._stop_event.clear()
        self._audio_queue = queue.Queue(maxsize=200)

        # Bind audio push callback — TASK-41 pending; degrade gracefully on NotImplementedError.
        try:
            self._unitree_g1.register_audio_callback(self._on_audio_chunk)
            self._audio_cb_registered = True
        except NotImplementedError:
            logging.warning(
                "STTProvider: UnitreeG1.register_audio_callback NotImplementedError "
                "(TASK-41 pending) — live mic inactive; use _on_audio_chunk() directly "
                "or wait for TASK-41"
            )

        # Bind E-STOP push callback — TASK-41 pending; degrade gracefully.
        try:
            self._unitree_g1.register_estop_callback(self._on_estop)
            self._estop_cb_registered = True
        except NotImplementedError:
            logging.warning(
                "STTProvider: UnitreeG1.register_estop_callback NotImplementedError "
                "(TASK-41 pending) — E-STOP gate inactive; call _on_estop() directly "
                "for testing"
            )

        # Start backend worker thread
        if self._config.backend == STTBackend.GOOGLE_CLOUD:
            target = self._google_worker
            name = "stt_google_worker"
        elif self._config.backend == STTBackend.DUMMY:
            target = self._dummy_worker
            name = "stt_dummy_worker"
        else:
            self._state = STTState.IDLE
            raise ValueError(f"STTProvider: unknown backend {self._config.backend!r}")

        self._worker_thread = threading.Thread(target=target, name=name, daemon=True)
        self._worker_thread.start()
        self._state = STTState.STREAMING
        logging.info(
            "STTProvider: started (backend=%s, state=%s)",
            self._config.backend.value,
            self._state.value,
        )

    def stop(self) -> None:
        """Cancel stream, unregister audio callback, drain callbacks."""
        if self._state == STTState.IDLE:
            return

        # Unregister only the callbacks we successfully bound
        if self._audio_cb_registered:
            try:
                self._unitree_g1.unregister_audio_callback(self._on_audio_chunk)
            except NotImplementedError:
                pass
            self._audio_cb_registered = False

        if self._estop_cb_registered:
            try:
                self._unitree_g1.unregister_estop_callback(self._on_estop)
            except NotImplementedError:
                pass
            self._estop_cb_registered = False

        # Signal worker to stop and unblock it with a poison pill
        self._stop_event.set()
        if self._audio_queue is not None:
            try:
                self._audio_queue.put_nowait(None)
            except queue.Full:
                pass

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logging.warning("STTProvider: worker thread did not stop within 5s")
            self._worker_thread = None

        self._audio_queue = None
        self._state = STTState.IDLE
        logging.info("STTProvider: stopped")

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
    # TTS echo-cancel onset hint (called by TTSProvider before synthesis)
    # ------------------------------------------------------------------
    def notify_tts_onset(self) -> None:
        """Pre-mute the mic for ``echo_cancel_lead_ms`` starting now.

        Call this at TTS synthesis start so the mic is already muted when
        audio reaches the room (DDS speaker_state trails real playback by
        ~50-100 ms, default echo_cancel_lead_ms=0 disables this path).
        """
        if self._config.echo_cancel_lead_ms > 0:
            self._lead_mute_end = time.monotonic() + self._config.echo_cancel_lead_ms / 1000.0
            logging.debug(
                "STTProvider: TTS onset hint, leading mute for %dms",
                self._config.echo_cancel_lead_ms,
            )

    # ------------------------------------------------------------------
    # E-STOP callback (bound via UnitreeG1Provider, TASK-41)
    # ------------------------------------------------------------------
    def _on_estop(self, active: bool, ts: float) -> None:
        self._estop_active = active
        logging.info(
            "STTProvider: E-STOP %s (ts=%.3f)", "ACTIVE" if active else "CLEARED", ts
        )

    # ------------------------------------------------------------------
    # Audio callback from UnitreeG1Provider (TASK-41 push path)
    # ------------------------------------------------------------------
    def _on_audio_chunk(self, pcm: bytes, ts: float) -> None:
        """Drop while E-STOP or echo-muted; forward to backend queue otherwise."""
        # ① E-STOP gate — hard block; no silence injection needed
        if self._estop_active:
            return

        # ② Echo-cancel gate (speaker playing + tail-off + leading edge)
        if self._check_echo_mute(ts):
            # Inject silence of identical length so Google's idle timeout
            # does not terminate the stream during long TTS playback.
            if self._audio_queue is not None:
                try:
                    self._audio_queue.put_nowait(bytes(len(pcm)))
                except queue.Full:
                    pass
            return

        # ③ Apply speech filter then feed to backend queue
        if self._audio_queue is not None:
            try:
                samples  = np.frombuffer(pcm, dtype=np.int16)
                filtered = self._speech_filter.process(samples, self._gain_linear)
                self._audio_queue.put_nowait(filtered.tobytes())
            except queue.Full:
                logging.debug("STTProvider: audio queue full, dropping chunk")

    def _check_echo_mute(self, ts: float) -> bool:
        """True if the mic should be muted at audio timestamp ``ts``."""
        # Leading-edge mute (notify_tts_onset hint)
        if self._lead_mute_end is not None and ts < self._lead_mute_end:
            return True

        # Speaker playing (None guard on TopicCache.value and playing attribute)
        speaker_val = self._unitree_g1.speaker_state.value
        playing = False
        if speaker_val is not None:
            p = getattr(speaker_val, "playing", None)
            if p is not None:
                playing = bool(p)

        if playing:
            # Extend tail deadline while speaker is active
            self._echo_tail_end = ts + self._config.echo_cancel_tail_ms / 1000.0
            return True

        # Tail-off window
        if self._echo_tail_end is not None and ts < self._echo_tail_end:
            return True

        return False

    # ------------------------------------------------------------------
    # Internals — emit to all subscribers
    # ------------------------------------------------------------------
    def _emit_transcript(self, event: TranscriptEvent) -> None:
        """Fan ``event`` to all registered callbacks; exceptions logged, not raised."""
        with self._callbacks_lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(event)
            except Exception:
                logging.exception("STTProvider: transcript callback raised")

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------
    def _drain_audio_queue(self) -> int:
        """재연결 전 쌓인 stale 오디오를 모두 버린다. 버린 청크 수를 반환."""
        dropped = 0
        if self._audio_queue is None:
            return dropped
        while True:
            try:
                self._audio_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    # ------------------------------------------------------------------
    # Google Cloud STT backend
    # ------------------------------------------------------------------
    def _google_request_gen(self):
        """Yield StreamingRecognizeRequest objects consumed from the audio queue."""
        try:
            from google.cloud import speech
        except ImportError:
            logging.error(
                "STTProvider: google-cloud-speech not installed — "
                "run 'uv add google-cloud-speech>=2.21.0'"
            )
            return

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            if chunk is None:  # poison pill from stop()
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def _google_worker(self) -> None:
        """Auto-reconnect loop for Google Cloud bidi gRPC stream (~5 min limit)."""
        try:
            from google.cloud import speech
        except ImportError:
            logging.error(
                "STTProvider: google-cloud-speech not installed — "
                "run 'uv add google-cloud-speech>=2.21.0'"
            )
            self._state = STTState.FAILED
            return

        # SpeechClient 한 번만 생성 — gRPC 채널 재사용.
        # 에러로 client 자체가 망가진 경우에만 재생성.
        client = speech.SpeechClient()
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self._config.sample_rate_hz,
            language_code=self._config.language_code,
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self._config.interim_results,
        )

        reconnect_count = 0
        backoff = 1.0

        while not self._stop_event.is_set():
            if reconnect_count > _MAX_RECONNECT:
                self._state = STTState.FAILED
                logging.error(
                    "STTProvider: max reconnect attempts (%d) exhausted → FAILED",
                    _MAX_RECONNECT,
                )
                break

            try:
                if reconnect_count == 0:
                    logging.info("STTProvider: Google streaming session started")
                else:
                    self._state = STTState.RECONNECTING
                    logging.info(
                        "STTProvider: Google reconnecting (attempt %d/%d)",
                        reconnect_count, _MAX_RECONNECT,
                    )

                responses = client.streaming_recognize(
                    config=streaming_config,
                    requests=self._google_request_gen(),
                )
                self._state = STTState.STREAMING

                for response in responses:
                    if self._stop_event.is_set():
                        return
                    for result in response.results:
                        if not result.alternatives:
                            continue
                        alt = result.alternatives[0]
                        if result.is_final or self._config.interim_results:
                            confidence = alt.confidence if alt.confidence > 0 else None
                            self._emit_transcript(TranscriptEvent(
                                text=alt.transcript,
                                ts=time.monotonic(),
                                is_final=result.is_final,
                                confidence=confidence,
                            ))

                # Stream ended normally (single_utterance or ~5 min rotation)
                if not self._stop_event.is_set():
                    dropped = self._drain_audio_queue()
                    logging.info(
                        "STTProvider: Google stream ended, restarting (flushed %d stale chunks)",
                        dropped,
                    )
                    reconnect_count = 0
                    backoff = 1.0

            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._drain_audio_queue()
                reconnect_count += 1
                logging.warning(
                    "STTProvider: Google stream error (%s) — retry %d/%d in %.1fs",
                    exc, reconnect_count, _MAX_RECONNECT, backoff,
                )
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, 30.0)
                # gRPC 채널 수준 에러면 client 재생성
                try:
                    client = speech.SpeechClient()
                except Exception:
                    pass

        if self._state != STTState.FAILED:
            self._state = STTState.IDLE
        logging.info("STTProvider: Google worker stopped")

    # ------------------------------------------------------------------
    # DUMMY backend (local verification — no credentials needed)
    # ------------------------------------------------------------------
    def _dummy_worker(self) -> None:
        """Decode non-silent PCM chunks as UTF-8 and emit TranscriptEvents.

        Exercise scripts feed text bytes instead of real PCM so the filter
        chain can be exercised without a live mic or cloud credentials.
        All-zero chunks (silence injected by echo-cancel) are silently dropped.
        """
        logging.info("STTProvider: DUMMY worker started")
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:  # poison pill
                break
            # Skip echo-cancel silence injection
            if not any(chunk):
                continue
            # Decode as UTF-8 text (exercise callers feed encoded strings)
            try:
                text = chunk.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not text:
                continue
            self._emit_transcript(TranscriptEvent(
                text=text,
                ts=time.monotonic(),
                is_final=True,
                confidence=None,
            ))
        logging.info("STTProvider: DUMMY worker stopped")
