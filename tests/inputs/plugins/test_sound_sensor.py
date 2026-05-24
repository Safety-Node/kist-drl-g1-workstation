"""
Tests for SoundSensor scaffold (TASK-46).

Placeholder -- the primary path (``on_transcript`` → TaskSrvProvider) is
NotImplementedError today. Once the bridge is wired, add:

TODO(REQ-NEW-TASKSRV) [TASK-46]: stub STTProvider with a
                                 ``register_transcript_callback`` no-op,
                                 stub TaskSrvProvider with an ``on_audio``
                                 list-append; bind + start the sensor and
                                 assert the callback dispatch path round-trips.
TODO(REQ-NEW-TASKSRV) [TASK-46]: empty / whitespace-only transcripts dropped.
TODO(REQ-NEW-TASKSRV) [TASK-46]: duplicate-within-dedupe-window dropped.
TODO(REQ-NEW-TASKSRV) [TASK-46]: ring buffer respects ``buffer_size``.
TODO(REQ-NEW-TASKSRV) [TASK-46]: ``start()`` before ``bind()`` raises.
"""

import pytest

from inputs.plugins.sound_sensor import (
    SoundSensor,
    SoundSensorConfig,
    TranscriptEvent,
)


def test_default_config_values():
    cfg = SoundSensorConfig()
    assert cfg.buffer_size == 5
    assert cfg.min_confidence == 0.3
    assert cfg.dedupe_window_s == 1.5


def test_custom_config_values():
    cfg = SoundSensorConfig(
        buffer_size=10,
        min_confidence=0.5,
        dedupe_window_s=2.0,
    )
    assert cfg.buffer_size == 10
    assert cfg.min_confidence == 0.5
    assert cfg.dedupe_window_s == 2.0


def test_constructor_initializes_buffer_and_deps():
    s = SoundSensor(SoundSensorConfig())
    assert len(s._buffer) == 0
    assert s._buffer.maxlen == 5
    assert s._stt is None
    assert s._task_srv is None
    assert s._started is False


def test_bind_sets_dependencies():
    s = SoundSensor(SoundSensorConfig())

    class _StubSTT:
        pass

    class _StubTaskSrv:
        pass

    stt, task_srv = _StubSTT(), _StubTaskSrv()
    s.bind(stt=stt, task_srv=task_srv)
    assert s._stt is stt
    assert s._task_srv is task_srv


def test_lifecycle_and_callback_are_stubs():
    """start/stop/on_transcript and Cortex legacy methods are NotImplementedError."""
    s = SoundSensor(SoundSensorConfig())
    with pytest.raises(NotImplementedError):
        s.start()
    with pytest.raises(NotImplementedError):
        s.stop()
    with pytest.raises(NotImplementedError):
        s.on_transcript("hello", 0.0)


@pytest.mark.asyncio
async def test_legacy_fuser_input_path_raises():
    """OM1 Cortex polling path is intentionally disabled (CONV-004)."""
    s = SoundSensor(SoundSensorConfig())
    with pytest.raises(NotImplementedError):
        await s._poll()
    with pytest.raises(NotImplementedError):
        await s._raw_to_text("anything")


def test_transcript_event_dataclass():
    ev = TranscriptEvent(text="양상추 꺼내줘", ts=123.45)
    assert ev.text == "양상추 꺼내줘"
    assert ev.ts == 123.45
