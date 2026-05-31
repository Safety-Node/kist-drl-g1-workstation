"""
Exercise TTSProvider [TASK-43] — CONV-009 log-based verification.

Verifies the decode → resample → publish path and the E-STOP gate WITHOUT
live Naver Clova credentials or a real NX speaker:
  - ``_http_post_clova`` is monkeypatched to return a canned 24 kHz WAV.
  - ``unitree_g1.publish_audio_out`` is replaced with a capturing fake
    (the real one is NotImplementedError until TASK-41).

Run from repo root:
    uv run python scripts/exercise_tts.py

Expected PASS lines:
  T1a: synthesize → publish_audio_out called once
  T1b: published PCM is 16 kHz mono int16 (resampled from 24 kHz)
  T2:  E-STOP active → synthesize aborts, no publish
  T3:  E-STOP cleared → synthesize publishes again
  T4:  is_synthesizing False at rest
"""

import asyncio
import io
import logging
import sys
import types
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def _stub_module(name: str, **attrs):
    """Create and register a lightweight stub module."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "rclpy" not in sys.modules:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        _stub_module("rclpy", ok=lambda: False, init=lambda **kw: None, shutdown=lambda: None)
        _stub_module("rclpy.node", Node=object)
        _stub_module("rclpy.executors", MultiThreadedExecutor=object)
        _stub_module("rclpy.qos", HistoryPolicy=object, QoSProfile=object, ReliabilityPolicy=object)
        _stub_module("sensor_msgs")
        _stub_module("sensor_msgs.msg", Imu=object)

import numpy as np  # noqa: E402

from providers.unitree_g1_provider import UnitreeG1Provider  # noqa: E402
from providers.tts_provider import TTSProvider, TTSConfig  # noqa: E402


def _make_wav(rate: int, seconds: float = 0.4, freq: float = 440.0) -> bytes:
    """Build a mono 16-bit WAV (sine tone) at ``rate`` Hz — stands in for Clova."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    samples = (0.3 * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _check(label: str, ok: bool) -> None:
    if ok:
        logging.info("PASS: %s", label)
    else:
        logging.error("FAIL: %s", label)


async def main() -> int:
    """Run TTSProvider decode/resample/publish + E-STOP gate verification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        force=True,
    )

    UnitreeG1Provider.reset()  # type: ignore[attr-defined]
    TTSProvider.reset()  # type: ignore[attr-defined]

    g1 = UnitreeG1Provider()
    tts = TTSProvider(TTSConfig(sample_rate_hz=16000, clova_sample_rate_hz=24000))

    # Capture published PCM (real publish_audio_out is NotImplementedError).
    published = []
    g1.publish_audio_out = lambda pcm: published.append(pcm)  # type: ignore[assignment]

    # Monkeypatch the network call: return a canned 24 kHz WAV.
    canned = _make_wav(rate=24000, seconds=0.4)

    async def _fake_post(text: str):
        return canned
    tts._http_post_clova = _fake_post  # type: ignore[assignment]

    # Inject fake credentials so synthesize() does not short-circuit on
    # "no credentials" (the real key check is irrelevant with the mock).
    tts.start()
    tts._client_id = "fake-id"
    tts._client_secret = "fake-secret"

    # ------------------------------------------------------------------
    # T1 — normal synth → publish, resampled to 16 kHz
    # ------------------------------------------------------------------
    await tts.synthesize("성공했습니다.")
    _check("T1a: synthesize → publish called once", len(published) == 1)
    if published:
        n_samples = len(published[0]) // 2  # int16
        # 0.4 s @ 16000 = 6400 samples (±1 from rounding)
        expected = int(16000 * 0.4)
        ok_rate = abs(n_samples - expected) <= 2
        logging.info("T1b: published %d samples (expected ~%d @16kHz)", n_samples, expected)
        _check("T1b: published PCM resampled to 16 kHz mono int16", ok_rate)

    # ------------------------------------------------------------------
    # T2 — E-STOP active → abort, no publish
    # ------------------------------------------------------------------
    tts._on_estop(True, ts=0.0)
    before = len(published)
    await tts.synthesize("이건 무시되어야 합니다.")
    _check("T2: E-STOP active → synthesize aborts, no publish", len(published) == before)

    # ------------------------------------------------------------------
    # T3 — E-STOP cleared → publishes again
    # ------------------------------------------------------------------
    tts._on_estop(False, ts=0.0)
    before = len(published)
    await tts.synthesize("다시 시작합니다.")
    _check("T3: E-STOP cleared → synthesize publishes again", len(published) == before + 1)

    # ------------------------------------------------------------------
    # T4 — is_synthesizing False at rest
    # ------------------------------------------------------------------
    _check("T4: is_synthesizing False at rest", tts.is_synthesizing is False)

    tts.stop()
    logging.info("exercise_tts: all cases complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
