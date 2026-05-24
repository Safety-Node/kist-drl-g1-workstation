"""
Tests for TaskSrvBg (TASK-39).

The BG is a fixed-rate driver around ``TaskSrvProvider.tick()``. Tests
inject a fake provider singleton via the ``TaskSrvProvider`` decorator's
reset/instance machinery so the BG resolves our fake in ``run()``.
"""

import threading
import time

import pytest

from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig
from providers.task_srv_provider import TaskSrvProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _CountingTickProvider:
    """Stand-in for the real TaskSrvProvider — just counts tick() calls."""

    def __init__(self):
        self.calls = 0
        self.raise_next = False

    def tick(self) -> None:
        self.calls += 1
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("synthetic tick failure")


@pytest.fixture
def fake_provider():
    """Swap TaskSrvProvider() singleton lookup with a counting stand-in."""
    TaskSrvProvider.reset()
    counter = _CountingTickProvider()
    # The singleton decorator stores the instance on the class; pre-seed it.
    TaskSrvProvider._singleton_class._singleton_instance = counter  # type: ignore[attr-defined]
    yield counter
    TaskSrvProvider.reset()


# ---------------------------------------------------------------------------
# Config / construction
# ---------------------------------------------------------------------------


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
    assert bg._task_srv is None
    assert bg.config.tick_rate_hz == 10.0


# ---------------------------------------------------------------------------
# run() loop behaviour
# ---------------------------------------------------------------------------


def _run_in_thread(bg: TaskSrvBg, stop_event: threading.Event):
    bg.set_stop_event(stop_event)
    t = threading.Thread(target=bg.run, name="task_srv_bg_test", daemon=True)
    t.start()
    return t


def test_run_ticks_at_configured_rate(fake_provider):
    bg = TaskSrvBg(TaskSrvBgConfig(tick_rate_hz=50.0))  # 20ms period
    stop = threading.Event()
    t = _run_in_thread(bg, stop)
    time.sleep(0.25)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()
    # ~12 ticks expected over 0.25s; allow generous margin for CI jitter.
    assert 4 <= fake_provider.calls <= 25


def test_run_continues_on_exception_when_swallow_enabled(fake_provider):
    fake_provider.raise_next = True
    bg = TaskSrvBg(TaskSrvBgConfig(tick_rate_hz=50.0, swallow_tick_exceptions=True))
    stop = threading.Event()
    t = _run_in_thread(bg, stop)
    time.sleep(0.15)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert fake_provider.calls >= 2  # at least the raising tick + one after


def test_run_exits_on_exception_when_swallow_disabled(fake_provider):
    fake_provider.raise_next = True
    bg = TaskSrvBg(TaskSrvBgConfig(tick_rate_hz=200.0, swallow_tick_exceptions=False))
    stop = threading.Event()
    t = _run_in_thread(bg, stop)
    t.join(timeout=0.5)
    assert not t.is_alive()
    assert fake_provider.calls == 1  # raised on the first tick


def test_stop_is_noop():
    bg = TaskSrvBg(TaskSrvBgConfig())
    assert bg.stop() is None
