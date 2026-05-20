# KIST DRL G1 Workstation -- Architecture Components

이 문서는 `docs/structure/container_option3.drawio`의 **PC (RTX 4090) Workstation** 컨테이너 박스에 표시된 13개 컴포넌트를 본 repo의 실제 코드 경로에 매핑한다. 각 컴포넌트는 _placeholder skeleton_ 상태이며, 구현 작업은 표 우측의 **TBD** 항목을 참조한다.

> drawio 출처: [`docs/structure/container_option3.drawio`](../structure/container_option3.drawio)
> 상위 안전 산출물: `safety_artifacts/` (별도 repo의 SSOT)

---

## Workstation 컴포넌트 매핑

| # | drawio 이름 | 종류 | 기술 스택 | 코드 경로 | 상태 |
|---|---|---|---|---|---|
| 1 | STT Provider | Container | Google STT | [`src/providers/google_stt_provider.py`](../../src/providers/google_stt_provider.py) | skeleton |
| 2 | TTS Provider | Container | Naver Clova | [`src/providers/naver_clova_tts_provider.py`](../../src/providers/naver_clova_tts_provider.py) | skeleton |
| 3 | UnitreeG1 Provider | Container | LAN / Python | [`src/providers/unitree_g1_provider.py`](../../src/providers/unitree_g1_provider.py) | skeleton (composes 기존 `unitree_g1_*_provider`) |
| 4 | Cortex | Container | LLM | [`src/runtime/cortex.py`](../../src/runtime/cortex.py) | OM1 기존 구현 (수정 TBD) |
| 5 | VLM Provider | Container | COSMOS | [`src/providers/vlm_cosmos_provider.py`](../../src/providers/vlm_cosmos_provider.py) | skeleton |
| 6 | VLA Provider | Container | Python / GPU (Gr00t N1.7 3B) | [`src/providers/vla_groot_provider.py`](../../src/providers/vla_groot_provider.py) | skeleton |
| 7 | GUI BackGround | Container | Python | [`src/backgrounds/plugins/gui_background.py`](../../src/backgrounds/plugins/gui_background.py) | skeleton |
| 8 | Sound Sensor | Container | Python | [`src/inputs/plugins/sound_sensor.py`](../../src/inputs/plugins/sound_sensor.py) | skeleton |
| 9 | Vision Sensor | Container | Python | [`src/inputs/plugins/vision_sensor.py`](../../src/inputs/plugins/vision_sensor.py) | skeleton |
| 10 | Move Connector | Container | Python | [`src/actions/move/connector/g1_workstation.py`](../../src/actions/move/connector/g1_workstation.py) | skeleton |
| 11 | Speak Connector | Container | Python | [`src/actions/speak/connector/g1_workstation.py`](../../src/actions/speak/connector/g1_workstation.py) | skeleton |
| 12 | Safety Provider | Container | Python | [`src/providers/safety_provider.py`](../../src/providers/safety_provider.py) | skeleton |
| 13 | IOProvider* | Container | Python | [`src/providers/io_provider.py`](../../src/providers/io_provider.py) | OM1 기존 구현 (drawio 오타: "IOPorvider") |

*drawio에는 `IOPorvider`로 표기되어 있으나, OM1 코드 기준 `IOProvider`가 정칙.

---

## External 컴포넌트 (Workstation 밖 / 참조용)

| drawio 이름 | 위치 | 비고 |
|---|---|---|
| KIST KAPEX | 외부 시스템 | 음성 명령 송수신 |
| Google Cloud STT | 외부 SaaS | STT Provider가 호출 |
| Naver CLOVA TTS | 외부 SaaS | TTS Provider가 호출 |
| GPT API | 외부 SaaS | Cortex가 호출 |
| Display System | 외부 시스템 | GUI BackGround → 송출 |
| G1 Onboard (Orin NX) | G1 본체 | `sensors / comm_bridge / navigation / safety_monitor / motor_controller` (이 repo 범위 외) |

---

## 데이터 흐름 (drawio edges 요약)

```
KAPEX --(Voice cmd, Air)--> G1 mic --> sensors --> comm_bridge --> UnitreeG1 Provider
                                                                      |
   Google Cloud STT <----(HTTPS/gRPC)---- STT Provider <---- UnitreeG1 Provider (audio)
   STT Provider --(Transcribed text)--> Sound Sensor --(Audio context)--> Cortex
   UnitreeG1 Provider --(Camera image + Buf State)--> VLM Provider
   VLM Provider --(Scene Description)--> Vision Sensor --(Visual context)--> Cortex

   Cortex <==(Fused prompt + Action plan)==> Safety Provider
   Cortex --(Speak Response)--> Speak Connector --(Text)--> TTS Provider <--HTTPS--> Naver CLOVA TTS
   Cortex --(SubTasks Control Cmd)--> Move Connector --(Upper Body Cmd)--> VLA Provider
                                       Move Connector --(Nav Cmd)--> UnitreeG1 Provider

   VLA Provider --(Upper Body Cmd)--> UnitreeG1 Provider
   VLA Provider --(VLA Sync Sig)--> VLM Provider
   TTS Provider --(Audio data PCM)--> UnitreeG1 Provider (speaker)

   IOProvider <--> (모든 컴포넌트, 상태 집계)
   IOProvider + UnitreeG1 Provider --> GUI BackGround --> Display System
```

---

## 공통 TBD (cross-cutting)

- **추적성**: 각 컴포넌트의 결정/거부/저하(degrade) 이벤트를 `safety_artifacts/00_master/decision_log.md`와 정합되는 형식으로 emit
- **테스트**: skeleton 별 `tests/providers/` `tests/inputs/plugins/` 하위 단위테스트 (현재 placeholder도 미작성)
- **config**: `config/unitree_g1_*.json5`에 13개 컴포넌트 enable/disable + 파라미터 정의
- **Prometheus metrics**: 각 컴포넌트의 latency/qps/error counter
- **Safety hooks**: Safety Provider와의 동기 호출 지점 명시 (Cortex in/out, Move Connector pre-publish, TTS pre-play)

자세한 TBD는 각 skeleton 파일 상단 docstring 참고.
