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
import time

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.task_srv_provider import TaskSrvProvider


class TaskSrvBgConfig(BackgroundConfig):
    """Configuration for TaskSrv BG."""

    tick_rate_hz: float = Field(
        default=10.0,
        gt=0,
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
    heartbeat_every_ticks: int = Field(
        default=50,                          # 5 s at the default 10 Hz tick
        ge=0,
        description=(
            "Emit an INFO heartbeat (current TaskSrvProvider state + tick "
            "counter) every N ticks. CONV-009: logs are the verification "
            "surface — a silent loop is indistinguishable from a dead one. "
            "Set to 0 to disable."
        ),
    )


class TaskSrvBg(Background[TaskSrvBgConfig]):
    """
    Calls ``TaskSrvProvider().tick()`` at ``tick_rate_hz``.

    The provider singleton is resolved in ``run()`` (not __init__) so the
    BG can be constructed before the provider is bound + started — startup
    order ownership lives with run.py per CONV-001.
    """

    def __init__(self, config: TaskSrvBgConfig):
        super().__init__(config)
        self._task_srv = None        # resolved in run() (singleton fetch)
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
        self._task_srv = TaskSrvProvider()  # already-started singleton
        period = 1.0 / float(self.config.tick_rate_hz)
        next_t = time.monotonic()
        tick_counter = 0
        heartbeat_n = self.config.heartbeat_every_ticks
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
            tick_counter += 1
            if heartbeat_n and tick_counter % heartbeat_n == 0:
                logging.info(
                    "TaskSrvBg: heartbeat tick=%d state=%s scenario=%s",
                    tick_counter,
                    self._task_srv.state.value,
                    self._task_srv.active_scenario_name,
                )
            next_t += period
            dt = next_t - time.monotonic()
            if dt > 0:
                if not self.sleep(dt):
                    return  # stop_event fired during sleep
            else:
                # Overran the period; reset baseline so we don't tight-loop.
                # Log per CONV-009 (no pytest — logs are the verification
                # surface). Persistent overruns mean tick_rate_hz is set
                # higher than TaskSrvProvider.tick() can actually deliver.
                logging.warning(
                    "TaskSrvBg: tick overran period by %.1f ms; resetting baseline",
                    -dt * 1000.0,
                )
                next_t = time.monotonic()
        logging.info("TaskSrvBg: tick loop exited (stop signalled)")

    def stop(self) -> None:
        """Base class handles the stop_event; nothing extra to release."""
        # No-op: tick loop sees should_stop() between ticks and exits.
        return None
