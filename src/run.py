"""
KIST DRL G1 Workstation entrypoint — mini-runner replacing OM1 ModeCortexRuntime
(the demo uses TaskSrvProvider, not an LLM Cortex; CONV-001/004).

Startup order is load-bearing (CONV-001/010): UnitreeG1 → STT/TTS/VLA/Nav →
Move/Speak connectors → TaskSrvProvider(bind + start) → backgrounds → SoundSensor
(last, so STT callbacks fan out only once TaskSrvBg can drain the queue — R4).
Other Provider→Provider deps are @singletons fetched in consumers' __init__, so
construction order alone wires them. Shutdown is reverse-order.

CLI: ``python src/run.py [scenario]`` loads config/scenarios/<scenario>.json5.
  --dry-run        wire components but skip .start()
  --scaffold-loop  skip backends still raising NotImplementedError; keep
                   TaskSrvProvider's loop alive to exercise the state machine.
"""

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import List, Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent
# Put repo root on sys.path so TaskSrvProvider.start()'s lazy `import
# config.scenarios` resolves when run as `python src/run.py`.
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
from providers.navigation_provider import NavigationProvider, NavigationProviderConfig
from providers.stt_provider import STTConfig, STTProvider
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider
from providers.tts_provider import TTSConfig, TTSProvider
from providers.unitree_g1_provider import UnitreeG1Provider
from providers.vla_provider import VLAConfig, VLAProvider


class Startable(Protocol):
    """Provider / connector lifecycle contract used by the mini-runner."""

    def start(self) -> None: ...
    def stop(self) -> None: ...


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )


class _Runtime:
    """Aggregate of live components for reverse-order shutdown."""

    def __init__(self) -> None:
        # task_srv is its own slot (not in `providers`) because its start()
        # loads scenarios and must run after every base provider it bind()s.
        self.providers: List[Startable] = []
        self.task_srv: Optional[TaskSrvProvider] = None
        self.sound_sensor: Optional[SoundSensor] = None
        self.backgrounds: List[Background] = []
        self.bg_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()


def _build_runtime(scenario_file: Optional[str] = None) -> _Runtime:
    """Construct + wire every component (no ``.start()``).

    ``scenario_file`` (CLI positional) picks which config/scenarios/*.json5
    TaskSrvProvider loads; None keeps TaskSrvConfig's default.
    """
    rt = _Runtime()

    # UnitreeG1 FIRST so the others' @singleton fetch (in their __init__)
    # returns this instance, not a fresh default (CONV-010).
    unitree_g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig())
    tts = TTSProvider(TTSConfig())
    vla = VLAProvider(VLAConfig())
    navigation = NavigationProvider(NavigationProviderConfig())  # CONV-012: loco split off VLA
    rt.providers = [unitree_g1, stt, tts, vla, navigation]

    # Connectors: stateless adapters; their __init__ fetches provider singletons.
    move_conn = MoveConnector(ActionConfig())
    speak_conn = SpeakConnector(ActionConfig())

    # Orchestrator — non-singleton connectors injected via bind() (CONV-010).
    task_cfg = TaskSrvConfig() if scenario_file is None else TaskSrvConfig(scenario_file=scenario_file)
    rt.task_srv = TaskSrvProvider(task_cfg)
    rt.task_srv.bind(move_connector=move_conn, speak_connector=speak_conn)
    logging.info("TaskSrvProvider scenario: %s", task_cfg.scenario_file)

    # STT → TaskSrv bridge; started LAST (after backgrounds) — see R4 below.
    rt.sound_sensor = SoundSensor(SoundSensorConfig())

    rt.backgrounds = [TaskSrvBg(TaskSrvBgConfig()), GUIBackground(GUIBackgroundConfig())]
    return rt


def _start_component(component: Startable, scaffold_loop: bool) -> None:
    """``.start()`` ``component``; swallow NotImplementedError in scaffold-loop mode."""
    name = type(component).__name__
    logging.info("Starting %s", name)
    try:
        component.start()
    except NotImplementedError as e:
        if not scaffold_loop:
            raise
        logging.warning("Scaffold-loop: skipping %s (%s)", name, e)


def _start_runtime(rt: _Runtime, dry_run: bool, scaffold_loop: bool) -> None:
    """``.start()`` each component in dependency order."""
    if dry_run:
        logging.info("Dry run: skipping .start() on all components")
        return

    for p in rt.providers:
        _start_component(p, scaffold_loop)

    assert rt.task_srv is not None
    logging.info("Starting TaskSrvProvider (loads scenarios)")
    rt.task_srv.start()

    # Backgrounds before SoundSensor (R4): the STT callback SoundSensor.start()
    # registers needs a live TaskSrvBg drain, or transcripts pile up.
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
    """Stop ``component``; True on clean stop (or scaffold NotImplementedError)."""
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="kist-drl-g1-workstation")
    parser.add_argument(
        "scenario",
        nargs="?",
        default=None,
        help=(
            "Scenario to run — a file under config/scenarios/ (e.g. 'move_test' "
            "or 'move_test.json5'; the .json5 suffix is optional). Omit for the "
            "TaskSrvConfig default (move_test.json5)."
        ),
    )
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
            "Scaffold mode: skip Provider/SoundSensor .start() calls that raise "
            "NotImplementedError; TaskSrvProvider still starts and its loop runs. "
            "SIGINT exits cleanly."
        ),
    )
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)
    # Anchor .env at repo root so the runner works from any cwd.
    dotenv.load_dotenv(dotenv_path=REPO_ROOT / ".env")

    scenario_file = args.scenario
    if scenario_file and not scenario_file.endswith(".json5"):
        scenario_file += ".json5"
    rt = _build_runtime(scenario_file)

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
        logging.error("Use --dry-run (wiring only) or --scaffold-loop (skip un-implemented backends).")
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

    try:
        rt.stop_event.wait()       # block until SIGINT/SIGTERM
    finally:
        failures = _stop_runtime(rt)
    # Surface .stop() failures via exit code (CONV-009: logs are the verification surface).
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
