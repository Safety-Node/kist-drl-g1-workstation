"""
TaskSrv BG [TASK-39, REQ-44] — thread host for TaskSrvProvider's loop.

run.py constructs + ``start()``s the provider; this Background's ``run()`` just
hands its thread to ``TaskSrvProvider.run()``, which owns the asyncio loop +
tick pump + hook scheduling. Loop tuning (tick rate, heartbeat, exception
policy) lives on ``TaskSrvConfig`` — there is nothing to configure here.
"""

import logging

from backgrounds.base import Background, BackgroundConfig
from providers.task_srv_provider import TaskSrvProvider


class TaskSrvBgConfig(BackgroundConfig):
    """No fields — loop tuning lives on TaskSrvConfig."""


class TaskSrvBg(Background[TaskSrvBgConfig]):
    """Thread host: delegates to ``TaskSrvProvider.run(stop_event)``."""

    def __init__(self, config: TaskSrvBgConfig):
        super().__init__(config)
        logging.info("TaskSrvBg: initialized")

    def run(self) -> None:
        """Run the provider's own loop on this background thread until stop."""
        TaskSrvProvider().run(self._stop_event)

    def stop(self) -> None:
        """Base class sets the stop_event; the provider loop sees it and exits."""
        return None
