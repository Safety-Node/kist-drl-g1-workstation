# kist-drl-g1-workstation

**KIST DRL — Unitree G1 PC Workstation Stack**

OpenMind OM1 fork for the PC (RTX 4090) running KIST G1 collaborative demo (2026).
Speech I/O, VLA inference, task orchestration, GUI streaming.

> Target HW: PC (Ubuntu 22.04, RTX 4090)
> Companion repo: `kist-drl-g1-onboard` (NX side — sensors / safety / motors)

> 🚧 **Scaffold.** Most providers/connectors are still placeholders. Implementation
> markers `TODO(REQ-XX) [TASK-XX]` link back to the Notion DBs.

---

## Components (workstation, PC-side)

| # | Component | Path | Notes |
|---|---|---|---|
| 1 | STT Provider | `src/providers/stt_provider.py` | Google Cloud STT default (TASK-42 scaffold). Vendor reference: `src/providers/example/google_stt_provider.py` |
| 2 | TTS Provider | `src/providers/tts_provider.py` | Naver Clova default (TASK-43 scaffold). Vendor reference: `src/providers/example/naver_clova_tts_provider.py` |
| 3 | UnitreeG1 Provider | `src/providers/unitree_g1_provider.py` | DDS facade — rclpy + CycloneDDS direct (TASK-41 scaffold) |
| 4 | VLA Provider | `src/providers/vla_provider.py` | GR00T N1.7 + GearSonic placeholder (TASK-40 scaffold). Vendor reference: `src/providers/example/vla_groot_provider.py` |
| 5 | TaskSrvProvider + TaskSrvBg | `src/providers/task_srv_provider.py` + `src/backgrounds/plugins/task_srv_bg.py` | Hook-based scenario state machine — JSON5 scenarios in `config/scenarios/`, replaces LLM Cortex |
| 6 | Move Connector | `src/actions/move/connector/move_connector.py` | Routes sub-task prompts to VLA / LocoCommand |
| 7 | Speak Connector | `src/actions/speak/connector/speak_connector.py` | Text → TTS Provider |
| 8 | Sound Sensor | `src/inputs/plugins/sound_sensor.py` | STT transcript → TaskSrvProvider |
| 9 | GUI Background | `src/backgrounds/plugins/gui_background.py` | Streams video + task status to Display System |
| 10 | IOProvider | `src/providers/io_provider.py` | OM1 infra — NOT used in KIST flow |

Deferred (kept in repo as `[DEPRECATED]` only in spec; not implemented):
- VLM Provider (Cosmos), Safety Provider, LLM Cortex — see SYS-REQ `[DEPRECATED 2026-05-24]`.

---

## Install

PC requirements:
- Ubuntu 22.04 (must match NX onboard distro)
- ROS 2 humble (must match NX onboard distro)
- Python 3.10 (Humble system Python; uv pinned to match — rclpy ABI is tied to the system Python minor version)

```bash
# System deps
sudo apt-get update && sudo apt-get install -y \
    portaudio19-dev python3-dev ffmpeg \
    ros-humble-rmw-cyclonedds-cpp \
    pybind11-dev libyaml-cpp-dev

# ROS 2 — https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
# Repo-local activation (recommended)
source env.sh

# Python deps (CycloneDDS Python bindings build against the C lib above)
uv sync --extra dds

# Submodules
# - src/unitree: Unitree SDK (required)
# - third_party/route_planner: C++ route planning pipeline (required)
# Note: src/ubtech/ is unused in KIST flow — not initialized.
git submodule update --init src/unitree third_party/route_planner

# Build route-planner (C++ → pybind11 .so)
# Must be re-run whenever third_party/route_planner is updated.
(cd third_party/route_planner && colcon build --packages-select route_planner)

# Credentials — loaded from repo-root .env by src/run.py (python-dotenv).
# Required keys: NCP_CLOVA_CLIENT_ID / NCP_CLOVA_CLIENT_SECRET (TTS),
# GOOGLE_APPLICATION_CREDENTIALS (path to GCP service-account JSON for STT).
cp .env.example .env
$EDITOR .env
```

---

## Run

Three scaffold-stage modes, ordered by increasing scope:

```bash
# 1. Wiring smoke test — construct + wire every component, skip .start().
#    Catches import / @singleton order / pydantic validation problems.
uv run python src/run.py --dry-run

# 2. Exercise TaskSrvProvider end-to-end with stubbed Connectors and an
#    injected trigger; full state machine cycle in ~5 s.
uv run python scripts/exercise_task_srv.py

# 3. Live scaffold runtime — TaskSrvProvider + backgrounds run; un-implemented
#    Provider .start() calls are logged + skipped. Ctrl+C exits cleanly.
uv run python src/run.py move_test --scaffold-loop

# (Future) Full run, once backends land
uv run python src/run.py move_test
```

Scenario: `src/run.py [scenario]` loads exactly one JSON5 file from `config/scenarios/`
(e.g. `move_test` → `move_test.json5`); omit the arg for the default. Add a scenario by
dropping a `.json5` file there — see `move_test.json5` for the hook/criteria schema.

PC↔NX transport: rclpy + CycloneDDS direct (no Zenoh bridge daemon).
PC and NX must share the same DDS domain + `cyclonedds.xml` network interface.

`safety_monitor` / `motor_controller` run on the NX (see `kist-drl-g1-onboard`).

---

## Where the spec lives

| Layer | Notion |
|---|---|
| Requirements | [SYS-REQ DB](https://www.notion.so/d7d7c9b9943b4018a4bce2afb904d706) |
| Interface contracts | [ICD DB](https://www.notion.so/b319b5cec8f2429389fb5fac8c042503) |
| Work items | [Tasks DB](https://www.notion.so/cd779d7a54b343b6a9e5449f4620a44c) |
| Verification | [Tests DB](https://www.notion.so/a67e62ef1cfc4f85be29a340107846b6) |

Each `TODO(REQ-XX) [TASK-XX]` in code links to the matching Notion page.

---

## Difference vs upstream OM1

- Non-G1 platforms removed (Go2 / Turtlebot4 / Yanshee / Booster / LimX K1·Tron / Spot / Cubly)
- Multi-vendor backends pruned (VLM / ASR / TTS / Web3 / Telegram / Twitter / Discord / Tesla / GPS / RTK ...)
- LLM Cortex deferred → TaskSrvProvider scripted orchestration
- IOProvider unused in KIST data flow; direct singleton polling
- No pytest infra — `system_hw_test/` + dev logging only
- Upstream merge path inactive — no `upstream` git remote configured. OM1 changes are
  not auto-tracked; any future absorption is a manual cherry-pick.
- Submodules:
  - `src/unitree/` — Unitree SDK (upstream OM1, used by `src/runtime/robotics.py`, **required**)
  - `src/ubtech/` — UBTech (Yanshee etc.) SDK, **unused in KIST flow** — slated for removal alongside `src/providers/io_provider.py`.
  - `third_party/route_planner/` — KIST-added C++ route planning pipeline (PointCloud2 → EDT Costmap → A*), exposed to Python via pybind11. **Required** for `NavigationProvider`.

---

## Contributing

PRs are squash-merged to `main`. Conventions enforced in CI:

- Branch name: `TASK-{number}` (Notion-linked work) or `chore/{description}` (non-task housekeeping — e.g. `chore/fix-typo-in-readme`, `chore/bump-cyclonedds-dep`)
- PR title: `[TASK-{number}] <type>(<scope>)?: <subject>` or `[chore] <type>(<scope>)?: <subject>` (Conventional Commits, lowercase casing)

`chore/...` branches are for housekeeping that doesn't warrant a Notion Task (typo fixes, dep bumps, comment cleanup). Notion tracking is by-pass for these.

---

## License

MIT (succeeded from OM1).
