"""
TaskSrv BG [TASK-39, REQ-44]

Fixed-rate driver for ``TaskSrvProvider.tick()``. Keeps the provider single-
threaded (no internal thread to join in tests) and pause-able by unloading
the BG without killing the provider singleton.

Resolves the provider singleton lazily in ``run()`` so run.py's explicit
startup ordering (CONV-001) is respected — BG ctors run before the provider
is bound + started.
"""

import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig


class TaskSrvBgConfig(BackgroundConfig):
    """Configuration for TaskSrv BG."""

    tick_rate_hz: float = Field(
        default=10.0,
        description=(
            "Rate at which TaskSrvProvider.tick() is called. Should match "
            "or be a divisor of TaskSrvConfig.tick_rate_hz."
        ),
    )
    swallow_tick_exceptions: bool = Field(
        default=True,
        description=(
            "If True, exceptions from TaskSrvProvider.tick() are logged "
            "and the loop continues. If False, the BG exits its thread "
            "(orchestrator restart policy decides what happens next)."
        ),
    )


class TaskSrvBg(Background[TaskSrvBgConfig]):
    """
    Calls ``TaskSrvProvider().tick()`` at ``tick_rate_hz``.

    The provider singleton is resolved lazily in ``run()`` rather than at
    construction time so that ``run.py``'s explicit startup ordering
    (CONV-001 Option D) is respected: BG instances may be constructed
    before the provider is bound + started.
    """

    def __init__(self, config: TaskSrvBgConfig):
        super().__init__(config)
        self._task_srv = None        # resolved lazily in run()
        logging.info(
            "TaskSrvBg: skeleton initialized (tick=%.1fHz, swallow_exc=%s)",
            config.tick_rate_hz,
            config.swallow_tick_exceptions,
        )

    def run(self) -> None:
        """
        Fixed-rate tick loop driving ``TaskSrvProvider.tick()`` until
        ``should_stop()`` returns True. Drift-free pacing via cumulative
        ``next_t`` so a slow tick doesn't permanently shift the schedule.
        """
        import time as _time
        from providers.task_srv_provider import TaskSrvProvider

        self._task_srv = TaskSrvProvider()  # already-started singleton
        period = 1.0 / float(self.config.tick_rate_hz)
        next_t = _time.monotonic()
        logging.info(
            "TaskSrvBg: tick loop entering (period=%.3fs, swallow_exc=%s)",
            period, self.config.swallow_tick_exceptions,
        )
        while not self.should_stop():
            try:
                self._task_srv.tick()
            except Exception:
                if self.config.swallow_tick_exceptions:
                    logging.exception("TaskSrvBg: tick() raised; swallowing")
                else:
                    logging.exception("TaskSrvBg: tick() raised; exiting loop")
                    return
            next_t += period
            dt = next_t - _time.monotonic()
            if dt > 0:
                if not self.sleep(dt):
                    return  # stop_event fired during sleep
            else:
                # Overran the period; reset baseline so we don't tight-loop.
                next_t = _time.monotonic()
        logging.info("TaskSrvBg: tick loop exited (stop signalled)")

    def stop(self) -> None:
        """Base class handles the stop_event; nothing extra to release."""
        # No-op: tick loop sees should_stop() between ticks and exits.
        return None
