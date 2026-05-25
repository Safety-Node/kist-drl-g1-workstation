"""
KIST DRL G1 Workstation entrypoint (mini-runner per CONV-001).

Explicit Provider lifecycle: replaces OM1 ``ModeCortexRuntime`` since the
KIST demo uses :class:`TaskSrvProvider` instead of an LLM Cortex (CONV-004).

Startup order: UnitreeG1 → STT / TTS / VLA → Move/Speak connectors →
TaskSrvProvider (bind + start loads scenarios) → SoundSensor (bind + start
registers STT callback) → backgrounds (TaskSrvBg, GUIBackground).
Shutdown is reverse-order. SIGINT/SIGTERM trigger the stop event.

Use ``--dry-run`` to validate the wiring graph without invoking ``.start()``
(most provider backends are still NotImplementedError during scaffold).
"""

import argparse
import asyncio
import logging
import signal
import sys
import threading
from typing import List, Optional

import dotenv

from actions.base import ActionConfig
from actions.move.connector.move_connector import MoveConnector
from actions.speak.connector.speak_connector import SpeakConnector
from backgrounds.base import Background, BackgroundConfig
from backgrounds.plugins.gui_background import GUIBackground
from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig
from inputs.plugins.sound_sensor import SoundSensor, SoundSensorConfig
from providers.stt_provider import STTConfig, STTProvider
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider
from providers.tts_provider import TTSConfig, TTSProvider
from providers.unitree_g1_provider import UnitreeG1Provider
from providers.vla_provider import VLAConfig, VLAProvider


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )


# ---------------------------------------------------------------------------
# Component construction + wiring
# ---------------------------------------------------------------------------


class _Runtime:
    """Aggregate of all live components — for reverse-order shutdown."""

    def __init__(self) -> None:
        self.providers: List[object] = []
        self.sound_sensor: Optional[SoundSensor] = None
        self.backgrounds: List[Background] = []
        self.bg_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()


def _build_runtime() -> _Runtime:
    """Construct + wire every component. Does NOT call ``.start()``."""
    rt = _Runtime()

    # Providers (each .bind() its UnitreeG1 dep before .start())
    unitree_g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig())
    stt.bind(unitree_g1=unitree_g1)
    tts = TTSProvider(TTSConfig())
    # TODO: tts.bind(unitree_g1=unitree_g1) once TTSProvider has bind()
    vla = VLAProvider(VLAConfig())
    # TODO: vla.bind(unitree_g1=unitree_g1) once VLAProvider has bind()
    rt.providers = [unitree_g1, stt, tts, vla]

    # Connectors (plain OM1 ActionConnector instances)
    move_conn = MoveConnector(ActionConfig())
    speak_conn = SpeakConnector(ActionConfig())
    # TODO: MoveConnector.bind(vla=vla, unitree_g1=unitree_g1) once that
    #       method exists on the connector (currently scaffold).
    # TODO: SpeakConnector.bind(tts=tts) likewise.

    # Orchestrator
    task_srv = TaskSrvProvider(TaskSrvConfig())
    task_srv.bind(
        unitree_g1=unitree_g1,
        move_connector=move_conn,
        speak_connector=speak_conn,
    )
    rt.providers.append(task_srv)

    # STT → TaskSrv bridge
    sensor = SoundSensor(SoundSensorConfig())
    sensor.bind(stt=stt, task_srv=task_srv)
    rt.sound_sensor = sensor

    # Backgrounds
    rt.backgrounds = [
        TaskSrvBg(TaskSrvBgConfig()),
        GUIBackground(BackgroundConfig()),
    ]
    return rt


def _start_runtime(rt: _Runtime, dry_run: bool) -> None:
    """Call ``.start()`` on each component in dependency order."""
    if dry_run:
        logging.info("Dry run: skipping .start() on all components")
        return

    for p in rt.providers[:-1]:  # all but TaskSrvProvider
        logging.info("Starting %s", type(p).__name__)
        p.start()  # type: ignore[attr-defined]
    # TaskSrvProvider.start() loads scenarios — must run last among providers
    logging.info("Starting TaskSrvProvider (loads scenarios)")
    rt.providers[-1].start()  # type: ignore[attr-defined]

    assert rt.sound_sensor is not None
    logging.info("Starting SoundSensor")
    rt.sound_sensor.start()

    for bg in rt.backgrounds:
        bg.set_stop_event(rt.stop_event)
        t = threading.Thread(target=bg.run, name=type(bg).__name__, daemon=True)
        t.start()
        rt.bg_threads.append(t)
        logging.info("Started background thread: %s", type(bg).__name__)


def _stop_runtime(rt: _Runtime) -> None:
    """Reverse-order shutdown. Best-effort: a failing .stop() is logged not raised."""
    rt.stop_event.set()
    for t in rt.bg_threads:
        t.join(timeout=2.0)
        if t.is_alive():
            logging.warning("Background thread %s did not stop within 2s", t.name)

    if rt.sound_sensor is not None:
        _safe_stop(rt.sound_sensor)

    for p in reversed(rt.providers):
        _safe_stop(p)


def _safe_stop(component: object) -> None:
    name = type(component).__name__
    try:
        component.stop()  # type: ignore[attr-defined]
        logging.info("Stopped %s", name)
    except NotImplementedError:
        logging.info("%s.stop() is still NotImplementedError (scaffold)", name)
    except Exception:
        logging.exception("%s.stop() raised", name)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="kist-drl-g1-workstation")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct + wire components but do not call .start()",
    )
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    dotenv.load_dotenv()

    rt = _build_runtime()

    # Trap SIGINT/SIGTERM → set stop event → main loop unblocks
    def _on_signal(signum, _frame):
        logging.warning("Received signal %d, shutting down", signum)
        rt.stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    try:
        _start_runtime(rt, dry_run=args.dry_run)
    except NotImplementedError as e:
        logging.error("Component not yet implemented: %s", e)
        logging.error("Use --dry-run to validate wiring without invoking .start()")
        _stop_runtime(rt)
        return 1
    except Exception:
        logging.exception("Startup failed")
        _stop_runtime(rt)
        return 1

    if args.dry_run:
        logging.info("Dry run complete (%d providers, %d backgrounds wired)",
                     len(rt.providers), len(rt.backgrounds))
        return 0

    # Block until signal
    try:
        rt.stop_event.wait()
    finally:
        _stop_runtime(rt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
