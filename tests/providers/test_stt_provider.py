"""
Tests for STTProvider scaffold (TASK-42).

Placeholder — methods are NotImplementedError today. Once the Google Cloud
STT streaming client is wired, add:

TODO(REQ-27) [TASK-42]: fake audio source fixture — feed canned PCM into
                        ``_on_audio_chunk`` and assert the transcript
                        callback fires with the expected string (use a
                        stub backend that returns canned transcripts).
TODO(REQ-27) [TASK-42]: echo cancel — set the (mock) UnitreeG1 Provider
                        ``speaker_state.value.playing`` to True, feed audio,
                        assert chunks are dropped; flip to False, assert
                        chunks resume after ``echo_cancel_tail_ms`` only.
TODO(REQ-27) [TASK-42]: reconnect — simulate streaming_recognize disconnect,
                        assert provider re-opens within backoff window.
TODO(REQ-27) [TASK-42]: latency budget — measure callback dispatch latency
                        with a stub backend; assert p50 < 500 ms.
"""

import pytest

from providers.stt_provider import STTBackend, STTConfig, STTProvider


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test gets a fresh provider instance."""
    STTProvider.reset()
    yield
    STTProvider.reset()


def test_provider_initializes_with_defaults():
    p = STTProvider()
    assert p is not None
    assert p._config.backend == STTBackend.GOOGLE_CLOUD
    assert p._config.language_code == "ko-KR"
    assert p._config.sample_rate_hz == 16000
    assert p._config.interim_results is False
    assert p._config.echo_cancel_tail_ms == 200
    assert p._running is False
    assert p._on_transcript is None


def test_provider_initializes_with_custom_config():
    cfg = STTConfig(
        backend=STTBackend.GOOGLE_CLOUD,
        language_code="en-US",
        sample_rate_hz=48000,
        interim_results=True,
        echo_cancel_tail_ms=100,
    )
    p = STTProvider(config=cfg)
    assert p._config.language_code == "en-US"
    assert p._config.sample_rate_hz == 48000
    assert p._config.interim_results is True
    assert p._config.echo_cancel_tail_ms == 100


def test_transcript_callback_registration():
    p = STTProvider()
    received = []

    def _cb(text: str) -> None:
        received.append(text)

    p.register_transcript_callback(_cb)
    assert p._on_transcript is _cb


def test_lifecycle_methods_raise_until_implemented():
    """start/stop and audio handler are NotImplementedError stubs for now."""
    p = STTProvider()
    with pytest.raises(NotImplementedError):
        p.start()
    with pytest.raises(NotImplementedError):
        p.stop()
    with pytest.raises(NotImplementedError):
        p._on_audio_chunk(b"\x00\x00")
