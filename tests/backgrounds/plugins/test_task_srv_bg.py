"""
Tests for TaskSrvBg scaffold (TASK-39).

Placeholder — ``run()`` is NotImplementedError today. Once the tick loop
is wired, add:

TODO(REQ-NEW-TASKSRV) [TASK-39]: stub TaskSrvProvider with counting tick();
                                  start BG on a thread, sleep ~0.5s, stop;
                                  assert tick count ≈ tick_rate_hz × elapsed.
TODO(REQ-NEW-TASKSRV) [TASK-39]: exception swallow path — provider.tick raises;
                                  with swallow_tick_exceptions=True the loop
                                  keeps running; with False the run() exits.
TODO(REQ-NEW-TASKSRV) [TASK-39]: stop() promptly unblocks the loop
                                  (should_stop polled between ticks).
"""

from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig


def test_default_config_values():
    cfg = TaskSrvBgConfig()
    assert cfg.tick_rate_hz == 10.0
    assert cfg.swallow_tick_exceptions is True


def test_custom_config_values():
    cfg = TaskSrvBgConfig(tick_rate_hz=20.0, swallow_tick_exceptions=False)
    assert cfg.tick_rate_hz == 20.0
    assert cfg.swallow_tick_exceptions is False


def test_constructor_initializes_lazy_provider_ref():
    bg = TaskSrvBg(TaskSrvBgConfig())
    assert bg._task_srv is None     # resolved in run()
    assert bg.config.tick_rate_hz == 10.0


def test_run_is_stub():
    import pytest

    bg = TaskSrvBg(TaskSrvBgConfig())
    with pytest.raises(NotImplementedError):
        bg.run()


def test_stop_is_noop():
    """Base Background handles stop_event; BG-specific stop has nothing to do."""
    bg = TaskSrvBg(TaskSrvBgConfig())
    # Should not raise, should not return anything meaningful.
    assert bg.stop() is None
