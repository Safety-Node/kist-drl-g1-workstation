"""
KIST DRL G1 Workstation entrypoint (mini-runner per CONV-001).

Explicit Provider lifecycle: replaces OM1 ``ModeCortexRuntime`` since the
KIST demo uses :class:`TaskSrvProvider` instead of an LLM Cortex (CONV-004).

Startup order: UnitreeG1 → STT / TTS / VLA → Move/Speak connectors →
TaskSrvProvider (bind + start loads scenarios) → backgrounds (TaskSrvBg,
GUIBackground) → SoundSensor (last — STT callbacks fan out only after the
drain-side TaskSrvBg is alive).
Shutdown is reverse-order. SIGINT/SIGTERM trigger the stop event.

Use ``--dry-run`` to validate the wiring graph without invoking ``.start()``
(most provider backends are still NotImplementedError during scaffold).
"""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import List, Optional, Protocol

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


REPO_ROOT = Path(__file__).resolve().parent.parent


class Startable(Protocol):
    """Provider / connector lifecycle contract used by the mini-runner."""

    def start(self) -> None: ...
    def stop(self) -> None: ...


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
        # Base providers in dependency order. start() runs over this list
        # left-to-right, stop() over reversed(). TaskSrvProvider is held in
        # its own slot below because its start() also loads scenarios and
        # must run AFTER every base provider it bind()s.
        self.providers: List[Startable] = []
        self.task_srv: Optional[TaskSrvProvider] = None
        self.sound_sensor: Optional[SoundSensor] = None
        self.backgrounds: List[Background] = []
        self.bg_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()


def _build_runtime() -> _Runtime:
    """Construct + wire every component. Does NOT call ``.start()``."""
    rt = _Runtime()

    # Base providers (each .bind() its UnitreeG1 dep before .start())
    unitree_g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig())
    stt.bind(unitree_g1=unitree_g1)
    tts = TTSProvider(TTSConfig())
    # TODO: tts.bind(unitree_g1=unitree_g1) once TTSProvider has bind()
    vla = VLAProvider(VLAConfig())
    # TODO: vla.bind(unitree_g1=unitree_g1) once VLAProvider has bind()
    rt.providers = [unitree_g1, stt, tts, vla]

    # Connectors. They are stateless adapters with no lifecycle of their
    # own; their __init__ fetches the relevant Provider singletons that
    # this function constructed above. No bind() ceremony — the @singleton
    # decorator + CONV-001 ordering guarantees the right instances.
    move_conn = MoveConnector(ActionConfig())
    speak_conn = SpeakConnector(ActionConfig())

    # Orchestrator (separate slot — started AFTER base providers since it
    # binds them, and scenarios are loaded inside start()).
    rt.task_srv = TaskSrvProvider(TaskSrvConfig())
    rt.task_srv.bind(
        unitree_g1=unitree_g1,
        move_connector=move_conn,
        speak_connector=speak_conn,
    )

    # STT → TaskSrv bridge. Started LAST (after backgrounds) so STT
    # callbacks don't fan out before TaskSrvBg is alive to drain the
    # inbound queue — see R4 in run.py review.
    sensor = SoundSensor(SoundSensorConfig())
    sensor.bind(stt=stt, task_srv=rt.task_srv)
    rt.sound_sensor = sensor

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

    for p in rt.providers:
        logging.info("Starting %s", type(p).__name__)
        p.start()

    assert rt.task_srv is not None
    logging.info("Starting TaskSrvProvider (loads scenarios)")
    rt.task_srv.start()

    # Backgrounds before SoundSensor (R4): once SoundSensor.start()
    # registers the STT callback, transcripts must have a live drain on
    # the other side or they queue up during the start gap.
    for bg in rt.backgrounds:
        bg.set_stop_event(rt.stop_event)
        t = threading.Thread(target=bg.run, name=type(bg).__name__, daemon=True)
        t.start()
        rt.bg_threads.append(t)
        logging.info("Started background thread: %s", type(bg).__name__)

    assert rt.sound_sensor is not None
    logging.info("Starting SoundSensor")
    rt.sound_sensor.start()


def _stop_runtime(rt: _Runtime) -> int:
    """Reverse-order shutdown. Returns the number of .stop() failures."""
    failures = 0
    rt.stop_event.set()
    for t in rt.bg_threads:
        t.join(timeout=2.0)
        if t.is_alive():
            logging.warning("Background thread %s did not stop within 2s", t.name)
            failures += 1

    if rt.sound_sensor is not None and not _safe_stop(rt.sound_sensor):
        failures += 1

    if rt.task_srv is not None and not _safe_stop(rt.task_srv):
        failures += 1

    for p in reversed(rt.providers):
        if not _safe_stop(p):
            failures += 1

    return failures


def _safe_stop(component: Startable) -> bool:
    """Stop ``component``; return True on clean stop (or scaffold NotImpl)."""
    name = type(component).__name__
    try:
        component.stop()
        logging.info("Stopped %s", name)
        return True
    except NotImplementedError:
        logging.info("%s.stop() is still NotImplementedError (scaffold)", name)
        return True
    except Exception:
        logging.exception("%s.stop() raised", name)
        return False


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
    # Anchor .env at the repo root rather than cwd so the runner works no
    # matter where the operator invokes it from.
    dotenv.load_dotenv(dotenv_path=REPO_ROOT / ".env")

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
        logging.info("Dry run complete (%d base providers + task_srv, %d backgrounds wired)",
                     len(rt.providers), len(rt.backgrounds))
        return 0

    # Block until signal
    try:
        rt.stop_event.wait()
    finally:
        failures = _stop_runtime(rt)
    # Non-zero exit when any component's .stop() raised (post-CONV-009 the
    # log is the verification surface — also surface it through the exit code
    # so demo wrappers / systemd can pick it up).
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
