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
| 1 | STT Provider | `src/providers/stt_provider.py` (TBD) | Default backend: Google Cloud STT. Vendor reference: `src/providers/example/google_stt_provider.py` |
| 2 | TTS Provider | `src/providers/tts_provider.py` (TBD) | Default backend: Naver Clova. Vendor reference: `src/providers/example/naver_clova_tts_provider.py` |
| 3 | UnitreeG1 Provider | `src/providers/unitree_g1_provider.py` | DDS facade via Zenoh bridge (TASK-41 scaffold) |
| 4 | VLA Provider | `src/providers/vla_provider.py` (TBD) | GR00T N1.7 + GearSonic (placement TBD). Vendor reference: `src/providers/example/vla_groot_provider.py` |
| 5 | TaskSrvProvider + TaskSrvBg | `src/providers/task_srv_provider.py` + `src/backgrounds/plugins/task_srv_bg.py` (TASK-39 scaffold) | Scenario-driven sub-task orchestrator (replaces LLM Cortex) |
| 6 | Move Connector | `src/actions/move/connector/g1_workstation.py` | Routes sub-task prompts to VLA |
| 7 | Speak Connector | `src/actions/speak/connector/g1_workstation.py` | Text → TTS Provider |
| 8 | Sound Sensor | `src/inputs/plugins/sound_sensor.py` | STT transcript → TaskSrvProvider |
| 9 | GUI Background | `src/backgrounds/plugins/gui_background.py` | Streams video + task status to Display System |
| 10 | IOProvider | `src/providers/io_provider.py` | OM1 infra |

Deferred (kept in repo as `[DEPRECATED]` only in spec; not implemented):
- VLM Provider (Cosmos), Safety Provider, LLM Cortex — see SYS-REQ `[DEPRECATED 2026-05-24]`.

---

## Build & Run

PC requirements:
- Ubuntu 22.04
- ROS 2 humble (for `rclpy` PC↔NX comm with the onboard's CycloneDDS network)
- Python 3.12 (managed via `uv`)

```bash
# system deps
sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-dev ffmpeg

# ROS 2 humble (one-time)
# follow https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
# then in every shell:
source /opt/ros/humble/setup.bash
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds/cyclonedds.xml
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# python deps + run
uv venv
uv run src/run.py kist_g1_demo   # TBD mode config
```

PC↔NX transport: **rclpy + CycloneDDS direct** (no Zenoh bridge daemon on the NX).
PC and NX must share the same DDS domain and `cyclonedds.xml` network interface.
OM1's internal Zenoh (mode manager, config provider) remains but stays loopback-only.

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
- LLM Cortex deferred → TaskSrvProvider scripted orchestration (KIST 2026-05-22)
- KIST workstation components scaffolded

Upstream remains accessible:

```bash
git remote -v   # origin / upstream
git fetch upstream && git merge upstream/main   # if needed
```

---

## Contributing

PRs are squash-merged to `main`. Conventions enforced in CI (TBD):

- Branch name: `TASK-{number}` (no description suffix)
- PR title: `[TASK-{number}] <type>(<scope>)?: <subject>` (Conventional Commits, lowercase casing)

---

## License

MIT (succeeded from OM1).
