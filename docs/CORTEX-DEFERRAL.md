# Cortex Deferral & TaskSrvProvider Architecture

Onboarding note for engineers joining the KIST DRL G1 workstation repo.
The repo is an OM1 fork; the architecture deviates from upstream in
specific, documented ways. This document explains the largest deviation
— **the LLM Cortex loop is not running** — and how the demo actually
executes today.

## TL;DR

- OM1's LLM Cortex loop is **off**. KIST 2026-05-22 decision deferred
  VLM, LLM, and Safety Provider development to a later milestone.
- The main loop is owned by `src/run.py` (mini-runner) +
  `TaskSrvProvider` (orchestration) + `TaskSrvBg` (tick driver).
- `config/sous_chef_g1.json5` keeps the OM1 `mode_config` shape so it
  reads natural to OM1-familiar contributors — but the live reader is
  our `run.py` (TODO), not `ModeCortexRuntime`.
- OM1 LLM code is preserved (`runtime/cortex.py`, `runtime/manager.py`,
  `llm/`, `fuser/`). LLM revival is a per-mode toggle, not a revert.

## What changed vs upstream OM1

| Area | OM1 upstream | KIST (today) |
|---|---|---|
| Main loop | `ModeCortexRuntime._run_cortex_loop` (LLM tick) | `TaskSrvBg.run` (success-poll tick) |
| Action selection | LLM picks from `agent_actions[]` | `TaskSrvProvider` runs scripted scenarios |
| Scenario trigger | LLM intent inference | STT keyword substring match |
| Success judgement | LLM implicit | `SuccessCriterion.evaluate(state)` explicit (UWB pose / joint state) |
| Provider lifecycle | Lazy init via `mode_config` consumers | Explicit `.start()` in `run.py` (CONV-001) |
| Mode transitions | `ModeManager` + transition rules | Single mode (`sous_chef_g1`) — no transitions |

## Why (KIST 2026-05-22 mail + Meta Data 2026-05-24)

VLM (Cosmos), Safety Provider, and LLM Cortex were assessed as
unattainable for the demo milestone. The alternative locked in:

- Scenario sequences become fixed Python scripts (`config/scenarios/*.py`,
  consumed by `TaskSrvProvider`).
- "Intent → action" LLM reasoning is replaced by "STT keyword →
  scenario" substring matching.

SYS-REQ consequences:

- **REQ-28** (자연어 태스크 자율 오케스트레이션) → `[DEPRECATED 2026-05-24]`
- **REQ-36** (VLM 장면 묘사) → `[DEPRECATED 2026-05-24]`
- **REQ-40** (LLM I/O 시멘틱 안전 검증) → `[DEPRECATED 2026-05-24]`
- **REQ-44** (TaskSrvProvider) → new, **P0**

## Execution flow

### Startup (`src/run.py`)

```
1. argparse + dotenv
2. Instantiate providers in dep order, then .start():
     UnitreeG1 → STT → TTS → VLA → TaskSrvProvider
3. Wire MoveConnector / SpeakConnector
4. SoundSensor.bind(stt, task_srv) + start()
5. Spawn worker threads for TaskSrvBg + GUIBackground
6. Main thread waits on SIGINT / SIGTERM
```

### Runtime (demo)

```
KAPEX speaks: "오이 가져와 부탁해요"
  → NX mic_node → /bridge/sensors/audio_pcm
  → STT Provider (Google STT v2 stream)
  → transcript callback
  → SoundSensor.on_transcript
  → TaskSrvProvider.on_audio(text)
  → _match_trigger(text)  → Scenario activated
  → _dispatch_current     → MoveConnector.connect(MoveInput(prompt))
  → VLAProvider.infer(prompt) → action_horizon-step chunk @ ~15 Hz
                                (default 16, KIST L40 ~63.9 ms / chunk
                                assumed; RTX 4090 unmeasured, TBD)
                              → arm/low frozenset split (PC)
  → UnitreeG1.publish_joint_chunk_arm + publish_joint_chunk_low
                              → JointCmdChunk on /bridge/cmd/{arm,low}
                              → NX motor_controller paces 100 Hz +
                                queue_aggregate.crossfade() (CONV-006 REVISED)
  → NX safety_monitor → motor_controller → G1 motors
```

### Tick (`TaskSrvBg`, 10 Hz)

```
TaskSrvProvider.tick()
  → SuccessCriterion.evaluate(uwb_cache, joint_cache)
  → True  → _advance_sub_task() (next sub-task dispatch)
  → False & timeout exceeded → _finish_failure()
  → all sub-tasks done → _finish_success()
```

OM1 `ModeCortexRuntime._run_cortex_loop` is **never called**.
`runtime/cortex.py` (1.2k LoC) is dormant.

## Config file shape

`config/sous_chef_g1.json5` mirrors OM1 `mode_config` 1:1:

- `hertz`, `name`, `system_prompt_base`
- `agent_inputs[]`, `cortex_llm`, `agent_actions[]`, `backgrounds[]`

Plus one KIST-only extension: `providers: {...}` (CONV-001 explicit
lifecycle — OM1 does lazy init in input/action/background ctors, we
instantiate in `run.py` instead).

`cortex_llm: null` is set explicitly (CONV-008: prefer null over
omission so reviewers see exactly which slot the LLM goes into when
it returns).

**Today:** the file is documentation. `run.py` uses dataclass defaults.
**Next step:** add `--config` flag to `run.py` and apply key-by-key
overrides (keys are 1:1 with dataclass field names).

## LLM revival path

Preserved as of HEAD (commit `bd22bac`):

| Path | LoC | State |
|---|---|---|
| `src/runtime/cortex.py` | ~1200 | MCP branches removed; rest intact |
| `src/runtime/manager.py` | ~1000 | unchanged |
| `src/runtime/config.py` | ~700 | `mcp_servers` field stripped; LLM fields intact |
| `src/llm/` | – | LLM backend abstraction (`function_schemas`, `output_model`, etc.) |
| `src/fuser/` | – | prompt builder (MCP descriptions block removed) |

Removed entirely: `src/mcp_servers/` only. MCP (Model Context Protocol
— OM1's LLM-tool-calling layer) is the one piece we judged worth
redesigning from scratch when LLM returns.

### Revival procedure

1. Populate `cortex_llm` in `mode_config`:
   ```json5
   cortex_llm: { type: "openai", config: { ... } }
   ```
2. Add `--mode llm` branch to `src/run.py` that hands off to
   `ModeCortexRuntime(mode_config).run()` instead of the mini-runner.
3. TaskSrvProvider and LLM Cortex coexist as two modes that the
   operator toggles at startup.

## Related conventions

- **CONV-001** — Provider explicit lifecycle in `run.py`
  ([Notion CONV page](https://app.notion.com/p/377b39de7dd780b391f3ceec30226a0e))
- **CONV-004** — Cortex / VLM / Safety Provider deferred → TaskSrvProvider
- **CONV-008** — Stay close to OM1 shape; document deviations

## Related Notion

- [Meta Data → Spec Change Log](https://www.notion.so/Meta-Data-353b39de7dd7801498cce3dbbea06f91) — 2026-05-22 and 2026-05-24 rows
- REQ-44 (TaskSrvProvider) in SYS-REQ DB
