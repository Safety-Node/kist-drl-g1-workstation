"""
KIST DRL G1 demo scenarios — JSON5 data (REQ-44 redesign 2026-05-30).

Each ``*.json5`` file here defines one Scenario: name, triggers, and a list of
sub_tasks, where every sub_task has a success criterion plus optional
lifecycle hook lists (``on_create`` / ``on_start`` / ``on_success`` /
``on_fail``). A connector call (speak / move) is just an action inside a hook,
so it can be freely included or omitted. See ``move_test.json5`` for a
linear voice-commanded (await keyword → drive → arrive), consume-once example.

JSON5 matches the project's config format (``config/sous_chef_g1.json5``):
comments, trailing commas, unquoted keys.

**TaskSrvProvider loads exactly ONE scenario file** — the active file is
chosen by ``TaskSrvConfig.scenario_file`` (default ``move_test.json5``) and
loaded via :func:`load`. Dropping a new ``.json5`` here does NOT auto-activate
it; point ``scenario_file`` at it. The loader validates schema + trigger
uniqueness, raising a clear error at startup instead of crashing mid-demo.
"""

from pathlib import Path

from providers.task_srv_provider import ScenarioConfigError, load_scenario_file

SCENARIOS_DIR = Path(__file__).resolve().parent


def load(filename: str):
    """Load + validate the ONE active scenario file.

    ``filename`` is resolved under this directory unless it is an absolute
    path. Returns a single-element list so the engine's trigger-index code is
    unchanged.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = SCENARIOS_DIR / path
    if not path.exists():
        available = ", ".join(p.name for p in sorted(SCENARIOS_DIR.glob("*.json5"))) or "(none)"
        raise ScenarioConfigError(
            f"scenario file not found: {path.name} (available in {SCENARIOS_DIR.name}/: {available})"
        )
    return [load_scenario_file(path)]
