"""
TaskSrv BG — KIST DRL G1 Workstation [TASK-39]
==============================================

Background runner that drives ``TaskSrvProvider.tick()`` at a fixed rate
so the provider can stay single-threaded and easy to test.

drawio C4 Container:
    Name        : TaskSrv BG
    Technology  : Python
    Description : Periodic tick driver for TaskSrvProvider success polling.

---

## Lifecycle (per CONV-001 Option D)

This BG does **not** create the TaskSrvProvider — ``run.py`` does. The BG
resolves the already-started singleton via plain ``TaskSrvProvider()`` and
just calls ``.tick()`` until ``should_stop()``.

    run.py:
        task_srv = TaskSrvProvider(config=...)
        task_srv.bind(...).start()
        # Background orchestrator picks up TaskSrvBg from mode_config.json5
        # and calls bg.run() in its own thread.

---

## Why a separate BG plugin?

Keeping the polling loop **out** of the provider makes the provider:

- single-threaded (no internal thread to join in tests),
- driven at a configurable rate (`tick_rate_hz`) without re-instantiating,
- pause-able by un-loading the BG from the mode config without killing
  the provider singleton (which other code may still reach for
  ``register_state_callback`` etc.).

---

TODO(REQ-NEW-TASKSRV) [TASK-39]: resolve TaskSrvProvider singleton in run() (lazy)
TODO(REQ-NEW-TASKSRV) [TASK-39]: rate-limited loop: time.monotonic() pacing so
                                  drift doesn't accumulate.
TODO(REQ-NEW-TASKSRV) [TASK-39]: on provider.tick() exception: log + continue
                                  (do not crash the BG, do not silently swallow).
TODO(REQ-NEW-TASKSRV) [TASK-39]: stop() — let the current tick finish, exit loop.
TODO(REQ-NEW-TASKSRV) [TASK-39]: tests under tests/backgrounds/plugins/test_task_srv_bg.py
                                  (stub provider with a counting tick()).
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
