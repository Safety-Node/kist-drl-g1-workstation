# kist-drl-g1-workstation

KIST DRL G1 휴머노이드 협업 시연(2026)의 **워크스테이션(PC, RTX 4090)** 측 런타임. OpenMind [`OM1`](https://github.com/OpenMind/OM1)을 기반으로 KIST KAPEX-G1 시나리오용으로 축소·재구성한 fork이다.


---

## Architecture

전체 컨테이너 다이어그램은 [`docs/structure/container_option3.drawio`](docs/structure/container_option3.drawio).
워크스테이션 13개 Architecture Component의 코드 매핑은 [`docs/architecture/workstation_components.md`](docs/architecture/workstation_components.md).

요약:

| # | Component | Code |
|---|---|---|
| 1 | STT Provider (Google STT) | `src/providers/google_stt_provider.py` |
| 2 | TTS Provider (Naver Clova) | `src/providers/naver_clova_tts_provider.py` |
| 3 | UnitreeG1 Provider | `src/providers/unitree_g1_provider.py` |
| 4 | Cortex (LLM) | `src/runtime/cortex.py` (OM1 기존) |
| 5 | VLM Provider (COSMOS) | `src/providers/vlm_cosmos_provider.py` |
| 6 | VLA Provider (Gr00t N1.7) | `src/providers/vla_groot_provider.py` |
| 7 | GUI BackGround | `src/backgrounds/plugins/gui_background.py` |
| 8 | Sound Sensor | `src/inputs/plugins/sound_sensor.py` |
| 9 | Vision Sensor | `src/inputs/plugins/vision_sensor.py` |
| 10 | Move Connector | `src/actions/move/connector/g1_workstation.py` |
| 11 | Speak Connector | `src/actions/speak/connector/g1_workstation.py` |
| 12 | Safety Provider | `src/providers/safety_provider.py` |
| 13 | IOProvider | `src/providers/io_provider.py` (OM1 기존) |

신규로 추가한 컴포넌트(#1–3, 5–12 중 신규 7건)는 모두 **placeholder skeleton** 상태이며, 각 파일 상단 docstring의 `TBD` 항목이 구현 백로그다.

## 차이 vs upstream OM1

- 타 로봇 플랫폼(Unitree Go2 / Turtlebot4 / UBTech Yanshee / Booster / LimX K1·Tron / Spot / Cubly) 제거
- 13개 워크스테이션 Architecture Component skeleton 추가
- `docs/structure/`, `docs/architecture/` 안전 아키텍처 문서 동기화

OM1 upstream은 `upstream` remote로 연결되어 있다:

```bash
git remote -v
# origin    https://github.com/Safety-Node/kist-drl-g1-workstation.git
# upstream  https://github.com/OpenMind/OM1.git
git fetch upstream
git merge upstream/main   # 필요 시 동기화
```

## Status

| Phase | Status |
|---|---|
| OM1 import | ✅ |
| Non-G1 platform prune (moderate) | ✅ |
| 13 component skeletons | ✅ |
| Component 구현 | ☐ TBD |
| 단위 테스트 | ☐ TBD |
| 통합 / 시연 | ☐ TBD |

---

## Getting Started (OM1 원본 quick start 발췌)

```bash
# macOS
brew install portaudio ffmpeg

# Linux
sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-dev ffmpeg

git submodule update --init
uv venv
uv run src/run.py unitree_g1_conversation   # G1 conversation 데모
```

자세한 설정·LLM API 키·BrainPack·Isaac Sim 통합은 OM1 원본 문서를 참조한다: <https://docs.openmind.com/>.

## License

MIT. OM1의 라이선스를 승계한다.
