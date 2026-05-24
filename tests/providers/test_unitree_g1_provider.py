"""
Tests for UnitreeG1Provider scaffold (TASK-41).

Placeholder — methods are NotImplementedError today. Once subscribers/
publishers are wired, add:

TODO(REQ-32) [TASK-41]: fake DDS publisher fixture (e.g. via local CycloneDDS
                        loopback or a mocked rclpy node) that feeds canned
                        messages into each subscribed topic, then assert the
                        matching ``TopicCache.value`` and ``last_seen_ts``
                        are updated.
TODO(REQ-32) [TASK-41]: stale detection — set ``last_seen_ts`` in the past
                        and assert ``TopicCache.stale(now, ttl_s)`` returns
                        True; also assert a fresh cache (ts==0.0) is stale.
TODO(REQ-33) [TASK-41]: ``publish_joint_cmd_arm`` / ``publish_joint_cmd_low``
                        watchdog — when ``comm_bridge_alive()`` is False,
                        publish must raise (decide policy: raise vs no-op).
TODO(REQ-33) [TASK-41]: schema validation on outbound JointCmd dict
                        (joint_names length matches q/dq/kp/kd/tau_ff).
"""

import pytest

from providers.unitree_g1_provider import TopicCache, UnitreeG1Provider


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test gets a fresh provider instance."""
    UnitreeG1Provider.reset()
    yield
    UnitreeG1Provider.reset()


def test_provider_initializes_with_defaults():
    p = UnitreeG1Provider()
    assert p is not None
    # All caches start empty
    for prop in (
        "color", "depth", "audio_pcm", "joint_state",
        "imu_base", "imu_ankle_left", "imu_ankle_right", "uwb_pose",
        "buf_state", "speaker_state", "estop",
    ):
        cache: TopicCache = getattr(p, prop)
        assert isinstance(cache, TopicCache)
        assert cache.value is None
        assert cache.last_seen_ts == 0.0


def test_topic_cache_fresh_is_stale():
    """A never-received cache should always be stale."""
    cache = TopicCache()
    assert cache.stale(now=1000.0, ttl_s=0.5) is True


def test_topic_cache_ttl_boundary():
    """A cache populated within ttl_s should NOT be stale."""
    cache = TopicCache(value="anything", last_seen_ts=100.0)
    assert cache.stale(now=100.4, ttl_s=0.5) is False
    assert cache.stale(now=100.6, ttl_s=0.5) is True


def test_lifecycle_methods_raise_until_implemented():
    """start/stop and publishers are NotImplementedError stubs for now."""
    p = UnitreeG1Provider()
    with pytest.raises(NotImplementedError):
        p.start()
    with pytest.raises(NotImplementedError):
        p.stop()
    with pytest.raises(NotImplementedError):
        p.publish_joint_cmd_arm({})
    with pytest.raises(NotImplementedError):
        p.publish_joint_cmd_low({})
    with pytest.raises(NotImplementedError):
        p.publish_loco_cmd({})
    with pytest.raises(NotImplementedError):
        p.publish_audio_out(b"")
    with pytest.raises(NotImplementedError):
        p.comm_bridge_alive()
