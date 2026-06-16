# verify_full_loop.py — 타임로깅 기준 다이어그램

`scripts/verify_full_loop.py`가 측정하는 타이밍 probe(T1~T5)의 위치와 각 구간의 의미.

> **전제**: TASK-41 `UnitreeG1Provider` 구현 완료 + NX `mic_node` / `speaker_node` /
> `comm_bridge` 가동. 모든 타임스탬프는 **desktop `time.monotonic()` 한 시계** 기준 —
> NX와 시계 동기화 불필요(마지막 구간 T4→T5는 왕복 측정).

---

## 전체 흐름과 probe 위치

```
 [NX]                          [Desktop (이 스크립트, 정식 경로)]                               [NX]
┌──────────┐   DDS    ┌────────────────────────────────────────────────────────────┐   DDS   ┌───────────┐
│ mic_node  │ ──────► │ UnitreeG1Provider ─► STT ─► SoundSensor ─► TaskSrvProvider │ ──────► │speaker_node│
│(comm_bridge│ audio_  │  (audio 콜백 push)   │          │              │  trigger  │ audio_  │ (comm_bridge│
│  outbound)│  pcm    │                      ▼          ▼              ▼  매칭→speak│  out    │  inbound)  │
└──────────┘         │                    ●T1        ●T2          ●T2b            │         └─────┬─────┘
                      │                                                  │          │               │ 재생
                      │            TTSProvider.synthesize ◄── SpeakConnector        │               ▼
                      │                  ●T3                                        │        speaker_state
                      │                   │ Clova 합성→리샘플                         │         playing=True
                      │                   ▼                                         │               │
                      │            g1.publish_audio_out ●T4                         │◄──────────────┘
                      │                                                             │   (relay 복귀)
                      │            g1.speaker_state 폴링 ●T5                         │
                      └────────────────────────────────────────────────────────────┘
```

## probe 정의 (코드 기준)

| Probe | 시점 | 심은 위치 (전부 측정 전용 래퍼 — 동작 불변) |
|---|---|---|
| **T1** | STT transcript 방출 | `stt.register_transcript_callback(...)` — SoundSensor보다 **먼저** 등록 (STT fan-out은 등록 순서) |
| **T2** | TaskSrv 인바운드 큐 진입 | `task_srv.on_audio` 래퍼 (SoundSensor가 호출하는 그 지점) |
| **T2b** | 시나리오 trigger 매칭/활성 | `task_srv._activate` 래퍼 (tick 스레드가 큐에서 꺼내 매칭 성공 시) |
| **T3** | TTS 합성 진입 | `tts.synthesize` 래퍼 (SpeakConnector→`_schedule_coro` 경유 호출 시점) |
| **T4** | PCM 발행 | `g1.publish_audio_out` 래퍼 (`/bridge/cmd/audio_out` 직전) |
| **T5** | 실제 재생 시작 신호 | `g1.speaker_state.value.playing == True` 폴링 (NX speaker_node 발행 → relay 복귀) |

## 구간 의미와 기대값

```
T1 ──► T2 ──► T2b ──► T3 ──► T4 ──────► T5
 SoundSensor  tick픽업   dispatch  TTS합성     relay+큐+재생
   전달      +trigger매칭  →synth   →publish     (왕복)
```

| 구간 | 무엇을 재나 | 기대값 | 크면 의심 지점 |
|---|---|---|---|
| T1→T2 | SoundSensor 콜백 전달 (동일 스레드) | ~0 ms | SoundSensor 필터/예외 |
| T2→T2b | 인바운드 큐 → tick 픽업 + trigger 매칭 | ≤ ~100 ms (TaskSrvBg 10 Hz tick) | TaskSrvBg 미가동/tick 정체 |
| T2b→T3 | `on_scenario_start` speak dispatch → synthesize 진입 | 수 ms | `_schedule_coro`/스레드 스케줄링 |
| T3→T4 | **Clova 왕복 + WAV 디코드 + 16k 리샘플** | 수백 ms (병목 1순위) | Clova 네트워크/자격증명 |
| T4→T5 | comm_bridge relay → speaker 큐 → PlayStream 진입 → speaker_state 복귀 (**왕복**) | ~100–300 ms | comm_bridge / speaker_node / AudioClient |
| **T1→T5** | **전체 응답** (transcript → 로봇 재생 시작) | < ~1.5 s 목표 | 위 구간별로 분해 |

> **측정 못 하는 것**: 사용자가 *말을 시작한 절대 시점* → T1 앞 구간(발화→transcript)은
> Google STT 스트리밍 내부라 이 스크립트로는 분해 불가. T1을 사이클 기준점(0)으로 삼는다.
> 또한 T4→T5는 **왕복**(편도 아님) — NX가 정확히 언제 소리를 냈는지(편도)는 시계 동기화(PTP) 필요.

## 사이클 규칙

- **T1(새 transcript)마다 새 사이클 시작** — 이전 사이클이 미완(T5 미수신)이면 버림.
- 사이클당 각 probe는 **최초 1회만** 기록 (TTS가 PCM을 여러 번 publish해도 T4는 첫 호출).
- T4 후 **15 s** 안에 `playing=True`가 안 오면 `T5 미수신`으로 요약 출력 (speaker/relay 실패 신호).
- trigger 미매칭 발화(예: 잡음 인식)는 T2까지만 찍히고 T2b 이후가 비어 요약에 `--`로 표시.

## 실행 / 예상 로그

```bash
# desktop, repo root
uv run python scripts/verify_full_loop.py
# 마이크: "오디오 테스트"  →  로봇 응답(cycle 1)
# 마이크: "테스트 끝"      →  응답 + 시나리오 종료(cycle 2)
```

```
TRANSCRIPT: '오디오 테스트'
==== cycle #1 timing  ('오디오 테스트') ====
  T1→T2  SoundSensor 전달      :       0 ms
  T2→T2b tick 픽업+trigger 매칭:      62 ms
  T2b→T3 dispatch→synthesize  :       3 ms
  T3→T4  TTS 합성→publish      :     648 ms
  T4→T5  publish→재생(왕복)    :     141 ms
  T1→T5  전체 응답              :     854 ms
==============================================
```

## 관련 파일

- 스크립트: `scripts/verify_full_loop.py`
- 전용 시나리오: `config/scenarios/audio_loop_test.json5` (speak 전용, move/VLA 미사용)
- 단독 검증(선행 권장): `verify_mic_live.py` / `verify_stt_live.py` / `verify_tts_live.py`(+`--tone`),
  onboard `exercise_speaker_live.py`
