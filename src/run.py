"""
KIST DRL G1 Workstation entrypoint (mini-runner per CONV-001).

Explicit Provider lifecycle: replaces OM1 ``ModeCortexRuntime`` since the
KIST demo uses :class:`TaskSrvProvider` instead of an LLM Cortex (CONV-004).

Startup order (CONV-001 + CONV-010): UnitreeG1 → STT / TTS / VLA →
Move/Speak connectors → TaskSrvProvider (bind connectors, start loads
scenarios) → backgrounds (TaskSrvBg, GUIBackground) → SoundSensor (last —
STT callbacks fan out only after the drain-side TaskSrvBg is alive).
All other Provider deps are @singletons fetched in consumer __init__;
the construction order above is load-bearing per CONV-010.
Shutdown is reverse-order. SIGINT/SIGTERM trigger the stop event.

Use ``--dry-run`` to validate the wiring graph without invoking ``.start()``
(most provider backends are still NotImplementedError during scaffold).

Use ``--scaffold-loop`` to keep the runtime alive in scaffold mode:
Provider/SoundSensor ``.start()`` calls that raise NotImplementedError are
logged + skipped instead of aborting. TaskSrvProvider (the only fully
implemented Provider today) still starts normally so its tick loop +
``on_audio`` queue actually run — useful for exercising the state machine
end-to-end before the backends land.
"""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import List, Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make `config/` (project-root sibling of src/) importable. TaskSrvProvider
# lazily imports ``config.scenarios.ALL`` in start(); Python's default sys.path
# only contains src/ when running ``python src/run.py``, so the lazy import
# would fail with ModuleNotFoundError without this prepend.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dotenv

from actions.base import ActionConfig
from actions.move.connector.move_connector import MoveConnector
from actions.speak.connector.speak_connector import SpeakConnector
from backgrounds.base import Background
from backgrounds.plugins.gui_background import GUIBackground, GUIBackgroundConfig
from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig
from inputs.plugins.sound_sensor import SoundSensor, SoundSensorConfig
from providers.stt_provider import STTConfig, STTProvider
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider
from providers.tts_provider import TTSConfig, TTSProvider
from providers.unitree_g1_provider import UnitreeG1Provider
from providers.vla_provider import VLAConfig, VLAProvider


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

    # Base providers. All Provider→Provider deps are @singletons (CONV-010),
    # so STT/TTS/VLA fetch UnitreeG1 inside their own __init__. The only
    # requirement is that UnitreeG1 is constructed FIRST so that fetch
    # returns the run.py-built instance instead of creating a default one.
    unitree_g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig())
    tts = TTSProvider(TTSConfig())  # CONV-010: __init__ fetches unitree_g1 (TBD)
    vla = VLAProvider(VLAConfig())
    rt.providers = [unitree_g1, stt, tts, vla]

    # Connectors. They are stateless adapters with no lifecycle of their
    # own; their __init__ fetches the relevant Provider singletons that
    # this function constructed above. No bind() ceremony — the @singleton
    # decorator + CONV-001 ordering guarantees the right instances.
    move_conn = MoveConnector(ActionConfig())
    speak_conn = SpeakConnector(ActionConfig())

    # Orchestrator (separate slot — started AFTER base providers since it
    # binds the non-singleton Connectors and loads scenarios inside start()).
    rt.task_srv = TaskSrvProvider(TaskSrvConfig())
    rt.task_srv.bind(move_connector=move_conn, speak_connector=speak_conn)

    # STT → TaskSrv bridge. SoundSensor fetches STT + TaskSrv as singletons
    # in its own __init__ (CONV-010). Started LAST (after backgrounds) so
    # STT callbacks don't fan out before TaskSrvBg is alive to drain the
    # inbound queue (R4).
    rt.sound_sensor = SoundSensor(SoundSensorConfig())

    rt.backgrounds = [
        TaskSrvBg(TaskSrvBgConfig()),
        GUIBackground(GUIBackgroundConfig()),
    ]
    return rt


def _start_component(component: Startable, scaffold_loop: bool) -> None:
    """Call ``.start()`` on ``component``; swallow NotImplementedError in scaffold-loop mode."""
    name = type(component).__name__
    logging.info("Starting %s", name)
    try:
        component.start()
    except NotImplementedError as e:
        if not scaffold_loop:
            raise
        logging.warning("Scaffold-loop: skipping %s (%s)", name, e)


def _start_runtime(rt: _Runtime, dry_run: bool, scaffold_loop: bool) -> None:
    """Call ``.start()`` on each component in dependency order."""
    if dry_run:
        logging.info("Dry run: skipping .start() on all components")
        return

    for p in rt.providers:
        _start_component(p, scaffold_loop)

    assert rt.task_srv is not None
    logging.info("Starting TaskSrvProvider (loads scenarios)")
    # TaskSrvProvider.start() is implemented; no NotImplementedError to swallow.
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
    _start_component(rt.sound_sensor, scaffold_loop)


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
    parser.add_argument(
        "--scaffold-loop",
        action="store_true",
        help=(
            "Keep the runtime alive in scaffold mode: Provider/SoundSensor "
            ".start() calls that raise NotImplementedError are logged + "
            "skipped (TaskSrvProvider still starts, tick loop runs). "
            "SIGINT exits cleanly."
        ),
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
        _start_runtime(rt, dry_run=args.dry_run, scaffold_loop=args.scaffold_loop)
    except NotImplementedError as e:
        logging.error("Component not yet implemented: %s", e)
        logging.error(
            "Use --dry-run to validate wiring only, or --scaffold-loop to keep "
            "TaskSrvProvider alive while skipping un-implemented backends."
        )
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
