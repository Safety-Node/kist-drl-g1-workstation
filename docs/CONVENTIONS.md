# Conventions — kist-drl-g1-workstation

Code-level architectural decisions for the workstation (PC) side. The system
spec (requirements, ICDs, change log) lives in Notion; this file captures
**engineering rules** that contributors should follow when writing or
reviewing code.

If you make a new decision that affects multiple files or future tasks,
append it here in the same format (see *Pattern* at the bottom).

---

## CONV-001 — Provider lifecycle: explicit init in `src/run.py`

**Status**: Accepted · **Date**: 2026-05-24

### Context
OM1 default pattern (`mode_config.json5 → mode → background → Provider() in __init__`)
proved fragile: 4-layer indirection, unclear dependency order, hard to test a
single provider in isolation, "where does this provider get initialised?"
trace expensive.

### Decision
Providers stay as `@singleton` (so `Provider()` from any caller returns the
same instance) but **lifecycle is owned by `src/run.py`** (or
`src/bootstrap.py` if extracted later):

1. `run.py` instantiates each provider in dependency order:
   `UnitreeG1 → (STT, TTS, VLA) → TaskSrvProvider`.
2. `run.py` calls `.start()` on each, then hands off to `ModeCortexRuntime`.
3. On shutdown, `run.py` calls `.stop()` in reverse order.
4. Backgrounds (`TaskSrvBg`, `GUI`, …) reach providers via `Provider()` —
   they get the already-started singleton; no init responsibility.
5. `mode_config.json5` does **not** declare providers (only backgrounds /
   inputs / actions toggled per mode).

### Consequences
- ✅ One file (`run.py`) shows entire startup sequence + config + dep order.
- ✅ Singleton pattern lets BGs/connectors keep clean `Provider()` access.
- ✅ Easy unit test: `Provider.reset(); p = Provider(config=...)` works.
- ⚠️ `run.py` grows. Acceptable; extract `src/bootstrap.py` if unwieldy.
- ⚠️ Mode hot-reload doesn't re-init providers (intentional — providers span modes).

---

## CONV-002 — PC↔NX transport: rclpy + CycloneDDS direct

**Status**: Accepted · **Date**: 2026-05-24

### Context
OM1 default uses `zenoh-bridge-ros2dds` daemon on NX + `eclipse-zenoh` Python
on PC. Considered for the KIST demo but rejected: team is already familiar
with ROS 2 + CycloneDDS from onboard work, single-LAN single-robot demo
gains nothing from Zenoh's cross-network features, and `ros2 cli` /
`ros2 bag` / `rqt` work out of the box with rclpy.

### Decision
PC subscribes/publishes `/bridge/*` topics via **`rclpy` + CycloneDDS** in
the same DDS domain as NX. PC requires:
- ROS 2 humble install (Ubuntu 22.04)
- `source /opt/ros/humble/setup.bash` in every shell
- `CYCLONEDDS_URI=file://$(pwd)/cyclonedds/cyclonedds.xml`
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`

NX side: no changes. `zenoh-bridge-ros2dds` is NOT deployed.

### Consequences
- ✅ Standard ROS 2 toolchain (ros2 topic / ros2 bag / rqt / rviz2).
- ✅ Direct DDS multicast peer discovery on the LAN.
- ⚠️ PC needs ROS 2 install (~3-5 GB).
- ⚠️ OM1's `zenoh_listener_provider` / `zenoh_publisher_provider` are unused
  for PC↔NX traffic → moved to `src/providers/example/`.
- ⚠️ OM1 framework internals (`runtime/manager.py`, `runtime/config.py`,
  `providers/config_provider.py`) still import `zenoh` — loopback only, not
  exposed on the wire. Acceptable cost.

### Affected
- `src/providers/unitree_g1_provider.py` (DDS facade for `/bridge/*` topics)
- `cyclonedds/cyclonedds.xml` (network interface)
- `README.md` (PC prereqs)

---

## CONV-003 — Vendor-agnostic provider naming

**Status**: Accepted · **Date**: 2026-05-24

### Context
Initial scaffolds used vendor-prefixed names (`google_stt_provider.py`,
`naver_clova_tts_provider.py`, `vla_groot_provider.py`). Hardcodes the
vendor into the class name and import path, painful to swap later
(Whisper, ETRI TTS, non-GR00T VLA).

### Decision
Provider files in `src/providers/` use **vendor-agnostic names**:
`stt_provider.py`, `tts_provider.py`, `vla_provider.py`. Backend selection
is runtime config (e.g. `STTConfig.backend = STTBackend.GOOGLE_CLOUD`).

Vendor-specific implementations either live as a backend strategy inside
the file (one file, multiple classes) or as a separate adapter that the
provider composes.

### Consequences
- ✅ Consumers (`Sound Sensor`, `Speak Connector`, etc.) import a stable name.
- ✅ Swap a backend by changing config, not by touching multiple files.
- ⚠️ Vendor-prefixed scaffolds preserved under `src/providers/example/` as
  reference for the default backend wiring.

---

## CONV-004 — Cortex / VLM / Safety Provider deferred → TaskSrvProvider

**Status**: Accepted · **Date**: 2026-05-22 (KIST mail) + 2026-05-24 (REQ deprecation)

### Context
LLM Cortex (`runtime/cortex.py`) + VLM Provider + Safety Provider deferred
per KIST schedule (insufficient development time for VLM/LLM in this
milestone).

### Decision
`TaskSrvProvider` + `TaskSrvBg` replace the LLM Cortex for scenario-driven
orchestration:
- Load pre-defined sub-task script (YAML or Python).
- STT keyword match triggers a scenario.
- Push sub-tasks (natural-language prompts) to VLA Provider via Move Connector.
- Poll `UnitreeG1Provider.uwb_pose` / `.joint_state` for sub-task success.

Long-term hook: keep `runtime/cortex.py` in place; switch via mode config
when LLM returns.

### Consequences
- ✅ Demo can run without LLM/VLM/Safety Provider development.
- ⚠️ `runtime/cortex.py` (770 LoC, OM1) stays but is bypassed by KIST mode.
- ⚠️ Sound Sensor formats audio context for TaskSrvProvider (not Cortex).

### Affected SYS-REQ (Notion)
- REQ-28 `자연어 태스크 자율 오케스트레이션` `[DEPRECATED 2026-05-24]`
- REQ-36 `VLM 장면 묘사` `[DEPRECATED 2026-05-24]`
- REQ-40 `LLM I/O 시멘틱 안전 검증` `[DEPRECATED 2026-05-24]`
- NEW: `사전 정의 sub-task 시퀀스 실행 및 성공 판정 (TaskSrvProvider)` (P0)

---

## CONV-005 — Whole-body VLA (locomotion + manipulation in one inference)

**Status**: Accepted · **Date**: 2026-05-22 (KIST mail)

### Context
Original split: navigation = high-level `LocoClient.Move` (NX),
manipulation = low-level VLA joint cmd. KIST observed the high↔low
transitions caused visible BalanceStand discontinuities and minimum
100 Hz control was required.

### Decision
Locomotion + manipulation are both driven by the **whole-body VLA**. NX
`navigation` package retired. Joint cmds split across two SDK paths:
- `/bridge/cmd/arm` (rt/arm_sdk, upper body, weight respected)
- `/bridge/cmd/low` (rt/lowcmd, lower body, weight ignored)

### Consequences
- ✅ Continuous whole-body motion, no rock-standing discontinuity.
- ⚠️ VLA Provider includes a `GearSonic` balance-correction stage post-VLA
  (placement TBD — see CONV-007).
- ⚠️ Onboard `navigation` package deleted (separate decision in onboard repo).

---

## CONV-006 — Control loop @ 100 Hz step replay (VLA chunk @ ~15 Hz)

**Status**: Accepted · **Date**: 2026-05-22 (KIST mail)

### Context
KIST requires ≥ 100 Hz low-level control. GR00T N1.7 chunk inference is
~63.9 ms on L40 (≈15 Hz chunk emission, 16-step chunks). Misreading this
as "100 Hz VLA inference" led to wrong REQ wording (corrected 2026-05-22).

### Decision
- **VLA chunk emission**: ~15 Hz (model property, fixed).
- **Step replay**: chunks unpacked into 16 step-level JointCmd messages at
  **100 Hz** (NX motor loop rate).
- **Motor control loop**: 100 Hz / 10 ms tick on NX `motor_controller`.
- **E-STOP path**: ≤ 200 ms regardless.

### Consequences
- ✅ KIST 100 Hz requirement met without requiring a 100 Hz VLA model.
- ⚠️ Receding-horizon: ~6-7 steps of every chunk get replayed before the
  next chunk arrives; the rest overwritten. Crossfade handled at PC VLA
  Provider side.

---

## CONV-007 — GearSonic placement: undecided

**Status**: Open · **Date**: 2026-05-22

### Context
GearSonic = presumed NVIDIA GR00T-WholeBodyControl GEAR-SONIC RL whole-body
controller. Takes VLA joint targets + IMU(base + ankle L/R) + joint state,
outputs balance-corrected joint targets. Needs GPU.

### Decision
**Three candidates open**, decision deferred:
1. PC co-tenant on RTX 4090 (resource contention with VLA + VLM if revived)
2. Separate Jetson (extra device cost, simplest isolation)
3. NX onboard (latency win, depends on NX TOPS budget)

Until decided, GearSonic logic lives **inside `VLAProvider`** as a post-VLA
stage. External interface (joint cmd out) is identical regardless of
placement; only deployment changes.

### Consequences
- ✅ Diagram + downstream code unchanged if placement flips.
- ⚠️ Re-extract into its own container when placement is settled.

### Owner
KIST 단장님 학생들 (per 2026-05-22 mail).

---

## CONV-008 — Stay close to OM1 shape; document deviations

**Status**: Accepted · **Date**: 2026-05-25

### Context
Team (3 devs) is familiar with OM1 patterns from upstream. Silent deviations
from OM1 file/section/symbol naming cost onboarding time — a new contributor
opens `mode_config.json5` expecting OM1 schema and is lost if they find a
parallel design instead. CONV-001..007 already encode our deliberate
deviations; this CONV makes the principle itself explicit so future PRs
don't accumulate undocumented drift.

### Decision
1. **Config files** (`mode_config.json5` and successors) match OM1 section
   names verbatim — even sections we no longer wire (e.g. `cortex_llm: null`)
   stay in the schema as placeholders, not omitted.
2. **Code-level deviations** require a CONV entry (CONV-001..007 are the
   precedent). When the deviation is unavoidable, mirror OM1 names where
   possible (e.g. `ActionConnector.connect(MoveInput)` kept verbatim even
   though our TaskSrvProvider also drives it).
3. **When in doubt, mirror OM1 shape** even if our runtime ignores it.
   OM1-familiar muscle memory should transfer to this repo by default.

### Consequences
- ✅ Devs reading our config immediately know which knob is which.
- ✅ Future LLM revival (CONV-004 reversal path) just enables sections
  that are already present.
- ⚠️ Adds a one-line "why this section is null" comment when sections are
  intentionally inert.

---

## CONV-009 — HW-system-test-only verification; no pytest infra

**Status**: Accepted · **Date**: 2026-05-25

### Context
3-person team, demo deadline. Maintaining a unit-test suite alongside
scaffold churn was negative-ROI: the tests mostly asserted
``NotImplementedError`` stubs and dataclass shapes that flux with every
PR. Real correctness signal comes from running on the actual G1 + KAPEX
setup and inspecting logs.

### Decision
- No ``pytest`` (removed dep, config, CI job, and ``tests/`` tree).
- Verification = run the demo or the relevant sub-system on real
  hardware; capture logs; attach the log path to the Notion Test page.
- The Notion **Tests DB** is the authoritative test surface — each test
  row has an "사전 조건 / 절차 / 기대 결과" triplet that maps to a
  hardware run, not a function call.
- Branch protection still enforces ``pr-meta.yml`` (branch + PR title
  regex) — those are zero-maintenance.

### Consequences
- ✅ Zero test-maintenance overhead during scaffold churn.
- ✅ Verification matches what actually matters (HW behaviour, not
  Python type annotations).
- ⚠️ Import-error regressions are caught only at ``run.py --dry-run``
  time; we lose the cheap automated guard. Recommended habit: run
  ``uv run python src/run.py --dry-run`` before pushing.
- ⚠️ When LLM revives (CONV-004 reversal), the kept-but-untested
  ``runtime/cortex.py`` / ``manager.py`` / ``llm/`` / ``fuser/`` have
  no automated coverage; HW runs are again the verification surface.

### Affected
- Removed ``tests/`` tree, ``pytest*`` dev deps, ``[tool.pytest.*]``
  and ``[tool.coverage.*]`` from ``pyproject.toml``.
- Removed ``.github/workflows/ci.yml`` (unit-test was its only job;
  lint had already been deferred per scaffold note).
- ``pr-meta.yml`` retained.

---

## CONV-010 — DI by singleton-fetch; bind() only when deps aren't singletons

**Status**: Accepted · **Date**: 2026-05-25

### Context
Early scaffold inconsistently applied ``bind(...)`` to STT, SoundSensor,
TaskSrvProvider on the rationale "CONV-001 explicit DI". That conflated
two things: CONV-001 owns *lifecycle order* (when ``start()`` runs), not
*DI mechanism* (how a component obtains its deps). For every dep that is
itself a ``@singleton``, ``bind`` is pure ceremony — ``Provider()`` from
the consumer's ``__init__`` returns the same instance ``run.py`` built.

### Decision
The real split is **whether the dep is a ``@singleton``**:

- **@singleton dep** → consumer fetches in ``__init__`` (or ``run()`` for
  background threads):
  ``self._dep = DepProvider()``
  No ``bind`` method, no ``start()`` RuntimeError guard for that dep.
- **Non-singleton dep** (plain instances — currently the two
  ``ActionConnector`` subclasses, ``MoveConnector`` / ``SpeakConnector``)
  → ``run.py`` is the only context that can hand the instance over;
  the consumer exposes ``bind(...)`` and validates in ``start()``.

In KIST today only **``TaskSrvProvider.bind(move_connector,
speak_connector)``** survives because the Connectors are plain instances.
Every other consumer (STT, TTS, VLA, SoundSensor, TaskSrvBg, the
Connectors themselves) fetches its deps via singleton.

CONV-001 still owns startup order: each Provider must be **constructed**
in ``run.py`` before any consumer that references it.

### Consequences
- ✅ One rule to teach: "if the dep is a Provider, just ``()`` it."
- ✅ ``run.py`` wiring shrinks (no ``stt.bind``, no ``sensor.bind``).
- ✅ One bind signature left to maintain (``TaskSrvProvider.bind``).
- ⚠️ Footgun: constructing a consumer **before** its Provider in
  ``run.py`` means the consumer's ``__init__`` creates the singleton
  with **default** config; the subsequent
  ``Provider(custom_config)`` silently returns the same default-config
  instance and the custom config is dropped. CONV-001 construction
  order is load-bearing — preserve it when refactoring ``run.py``.
- ⚠️ No automated test catches mis-ordered ``run.py`` after CONV-009;
  ``run.py --dry-run`` exercises the order and is the cheap gate.

---

## CONV-011 — IOProvider is OM1 infra, NOT used in KIST data flow

**Status**: Accepted · **Date**: 2026-05-25

### Context
OM1 ``IOProvider`` was the LLM Fuser's I/O bulletin board (system prompt,
fuser inputs, llm prompt, mode transition input, generic ``_inputs`` /
``_variables`` dict). CONV-004 deferred the LLM Cortex, so the LLM-shaped
fields are dead.

We briefly considered repurposing ``IOProvider._inputs`` as a generic
Provider → GUI bulletin board (Providers ``add_input("task.state", ...)``,
GUI ``get_input(...)``). Rejected — see Decision.

### Decision
KIST components do NOT read from or write to ``IOProvider``. Inter-component
state flows via direct ``@singleton`` polling (CONV-010). ``GUIBackground``
is the only consumer of cross-provider state and reads each provider's
own public property surface (``TaskSrvProvider.state``,
``UnitreeG1Provider.estop``, ``STTProvider.state``,
``TTSProvider.is_synthesizing``, etc.).

``IOProvider`` remains in ``src/providers/io_provider.py`` as untouched OM1
infrastructure for the same reason ``runtime/cortex.py`` does (CONV-004):
in case the LLM Cortex / mode system is revived later.

### Consequences
- ✅ Single pattern (CONV-010) — no new "bulletin board" concept to teach.
- ✅ Providers stay write-side single-responsibility (own state only).
- ✅ GUI dependency surface is explicit at construction.
- ⚠️ Adding a new GUI element means importing one more Provider in
  ``GUIBackground``. Acceptable for the demo's component count.

---

## Pattern for new conventions

When a decision affects multiple tasks or future code review, add a new
section with:

```
## CONV-NNN — Title
**Status**: Accepted | Open | Superseded by CONV-MMM · **Date**: YYYY-MM-DD

### Context
What problem are we solving? Constraints, alternatives considered briefly.

### Decision
What we chose. Concrete, code-pointing if possible.

### Consequences
Trade-offs, follow-ups, what it constrains in the future.

### Affected (optional)
Files / Notion SYS-REQ / other CONVs touched.
```

Reference: the per-decision history is also reflected in the Notion
**Meta Data → Spec Change Log** row dated when it was first agreed.
