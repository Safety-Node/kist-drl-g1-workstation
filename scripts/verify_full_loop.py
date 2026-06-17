"""
Full-loop verification:  NX-mic → STT → SoundSensor → TaskSrvProvider → TTS → NX-speaker.

전제(필수): TASK-41 UnitreeG1Provider 구현 완료 + NX 측 mic_node / speaker_node /
comm_bridge 가동. 어댑터/monkeypatch 로 ROS I/O 를 대체하던 verify_stt_live /
verify_tts_live 와 달리, 이 스크립트는 run.py 와 동일한 **정식 경로**로 컴포넌트를
와이어링한다 (UnitreeG1Provider 가 /bridge/* 구독·발행을 전부 소유).

타임로깅 (전부 desktop monotonic 시계 — NX 시계 동기화 불필요, 끝 구간은 왕복):
    T1  STT transcript 방출        (probe: register_transcript_callback)
    T2  TaskSrvProvider.on_audio   (probe: on_audio wrapper)
    T2b 시나리오 trigger 매칭/활성  (probe: _activate wrapper)
    T3  TTS synthesize 진입        (probe: synthesize wrapper)
    T4  PCM publish(audio_out)     (probe: publish_audio_out wrapper)
    T5  speaker playing=True       (probe: g1.speaker_state 폴링)
구간 의미는 docs/VERIFY_FULL_LOOP_TIMING.md 다이어그램 참조.

probe 는 전부 "원본 호출을 그대로 통과시키는 측정 래퍼" — 동작 변경 없음.

전제 환경:
    source env.sh
    export GOOGLE_APPLICATION_CREDENTIALS=...      # STT
    export NCP_CLOVA_CLIENT_ID=... NCP_CLOVA_CLIENT_SECRET=...   # TTS
실행 (desktop, repo root):
    uv run python scripts/verify_full_loop.py
    uv run python scripts/verify_full_loop.py --scenario audio_loop_test
사용법: 실행 후 마이크에 "오디오 테스트" → 로봇이 응답하면 cycle 1 성공.
        "테스트 끝" → 응답 + 시나리오 종료(cycle 2). Ctrl-C 로 종료.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import dotenv  # noqa: E402
dotenv.load_dotenv(dotenv_path=_ROOT / ".env")

from actions.base import ActionConfig                                  # noqa: E402
from actions.speak.connector.speak_connector import SpeakConnector     # noqa: E402
from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig # noqa: E402
from inputs.plugins.sound_sensor import SoundSensor, SoundSensorConfig # noqa: E402
from providers.stt_provider import STTConfig, STTProvider, TranscriptEvent  # noqa: E402
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider # noqa: E402
from providers.tts_provider import TTSConfig, TTSProvider              # noqa: E402
from providers.unitree_g1_provider import UnitreeG1Provider            # noqa: E402

T5_TIMEOUT_S = 15.0   # T4 이후 playing=True 대기 한도 (이 안에 안 오면 미수신 처리)


class CycleProbe:
    """한 발화 사이클(T1~T5)의 타임스탬프 수집 + 요약 출력. thread-safe."""

    STAGES = ("T1_transcript", "T2_on_audio", "T2b_activate",
              "T3_synthesize", "T4_publish", "T5_playing")

    def __init__(self):
        self._lock = threading.Lock()
        self._cycle = 0
        self._ts: dict = {}
        self._text: str = ""

    def mark(self, stage: str, text: str = "") -> None:
        now = time.monotonic()
        with self._lock:
            if stage == "T1_transcript":
                # 새 transcript = 새 사이클 시작 (이전 사이클 미완이면 그대로 버림)
                self._cycle += 1
                self._ts = {}
                self._text = text
            # 사이클당 각 stage 최초 1회만 기록
            if stage not in self._ts:
                self._ts[stage] = now

    def has(self, stage: str) -> bool:
        with self._lock:
            return stage in self._ts

    def t4_age(self) -> Optional[float]:
        with self._lock:
            t4 = self._ts.get("T4_publish")
        return (time.monotonic() - t4) if t4 is not None else None

    def summarize(self, t5_missing: bool = False) -> None:
        with self._lock:
            ts, cyc, text = dict(self._ts), self._cycle, self._text
            self._ts = {}
        if "T1_transcript" not in ts:
            return
        t1 = ts["T1_transcript"]

        def d(a: str, b: str) -> str:
            if a in ts and b in ts:
                return f"{(ts[b] - ts[a]) * 1000:7.0f} ms"
            return "     -- "

        logging.info("==== cycle #%d timing  (%r) ====", cyc, text)
        logging.info("  T1→T2  SoundSensor 전달      : %s", d("T1_transcript", "T2_on_audio"))
        logging.info("  T2→T2b tick 픽업+trigger 매칭: %s", d("T2_on_audio", "T2b_activate"))
        logging.info("  T2b→T3 dispatch→synthesize  : %s", d("T2b_activate", "T3_synthesize"))
        logging.info("  T3→T4  TTS 합성→publish      : %s", d("T3_synthesize", "T4_publish"))
        logging.info("  T4→T5  publish→재생(왕복)    : %s", d("T4_publish", "T5_playing"))
        if "T5_playing" in ts:
            logging.info("  T1→T5  전체 응답              : %7.0f ms", (ts["T5_playing"] - t1) * 1000)
        elif t5_missing:
            logging.warning("  T5 미수신 — speaker_node/comm_bridge(outbound) 또는 재생 실패 의심")
        logging.info("=" * 46)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="audio_loop_test.json5",
                    help="config/scenarios/<이름>.json5 (기본: audio_loop_test)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s", force=True)

    probe = CycleProbe()

    # ── 와이어링 (run.py CONV-010 순서: UnitreeG1 FIRST) ───────────────────
    g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig())
    tts = TTSProvider(TTSConfig())
    speak_conn = SpeakConnector(ActionConfig())
    task_srv = TaskSrvProvider(TaskSrvConfig(scenario_file=args.scenario))
    # audio_loop_test 시나리오는 speak 전용 — move/VLA 미사용.
    task_srv.bind(move_connector=None, speak_connector=speak_conn)
    sound_sensor = SoundSensor(SoundSensorConfig())

    # ── 타이밍 probe 설치 (측정만, 동작 불변 — 원본을 그대로 통과) ────────
    _orig_on_audio = task_srv.on_audio
    _orig_activate = task_srv._activate
    _orig_synth = tts.synthesize
    _orig_publish = g1.publish_audio_out
    _orig_stt_chunk = stt._on_audio_chunk        # T0 진단: 마이크 청크가 STT 에 도달하는지

    # T0 진단용 카운터 (오디오가 실제로 STT 에 흘러들어오는지 확인). 첫 청크는
    # 즉시 로그, 이후 100청크(~2s)마다 한 번씩 누적값을 출력. transcript 가 안
    # 보일 때 이 로그가 0이면 오디오 경로(comm_bridge/G1 sub) 문제, 늘어나는
    # 데 transcript 가 없으면 STT/Google 문제로 좁힐 수 있다.
    _stt_chunk_count = {"n": 0}

    def _p_on_audio(text, ts=None):
        probe.mark("T2_on_audio")
        return _orig_on_audio(text, ts)

    def _p_activate(scenario):
        probe.mark("T2b_activate")
        return _orig_activate(scenario)

    async def _p_synth(text):
        probe.mark("T3_synthesize")
        return await _orig_synth(text)

    def _p_publish(pcm):
        probe.mark("T4_publish")
        return _orig_publish(pcm)

    def _p_stt_chunk(pcm, ts):
        n = _stt_chunk_count["n"] + 1
        _stt_chunk_count["n"] = n
        if n == 1:
            logging.info("T0 STT first audio chunk: %d bytes (audio path OK)", len(pcm))
        elif n % 100 == 0:
            logging.info("T0 STT chunks received: %d", n)
        return _orig_stt_chunk(pcm, ts)

    task_srv.on_audio = _p_on_audio              # type: ignore[assignment]
    task_srv._activate = _p_activate             # type: ignore[assignment]
    tts.synthesize = _p_synth                    # type: ignore[assignment]
    g1.publish_audio_out = _p_publish            # type: ignore[assignment]
    stt._on_audio_chunk = _p_stt_chunk           # type: ignore[assignment]

    # ── 기동 (run.py 의존 순서: providers → task_srv → bg → sound_sensor) ─
    stop_event = threading.Event()
    g1.start()
    stt.start()
    tts.start()

    # 전제(TASK-41) 충족 확인 — STT 가 g1 audio 콜백에 실제로 바인딩됐는가
    if not getattr(stt, "_audio_cb_registered", False):
        logging.error(
            "전제 미충족: UnitreeG1Provider.register_audio_callback 미구현(TASK-41). "
            "이 스크립트는 정식 경로 전용 — TASK-41 머지 전에는 verify_stt_live / "
            "verify_tts_live (어댑터 버전) 를 사용하세요.")
        g1.stop()
        return 2

    task_srv.start()

    bg = TaskSrvBg(TaskSrvBgConfig())
    bg.set_stop_event(stop_event)
    bg_thread = threading.Thread(target=bg.run, name="TaskSrvBg", daemon=True)
    bg_thread.start()

    # T1 probe — SoundSensor 보다 먼저 등록(STT fan-out 은 등록 순서대로 호출)
    def _on_transcript(ev: TranscriptEvent) -> None:
        probe.mark("T1_transcript", ev.text)
        logging.info("TRANSCRIPT: %r", ev.text)

    stt.register_transcript_callback(_on_transcript)
    sound_sensor.start()

    logging.info("full loop ready (scenario=%s) — 마이크에 \"오디오 테스트\" 라고 말하세요. "
                 "Ctrl-C 종료.", args.scenario)

    # ── 메인 루프: T5(speaker playing) 폴링 + 사이클 요약 ─────────────────
    try:
        while True:
            time.sleep(0.02)
            # T5: g1 이 구독한 speaker_state 캐시에서 playing 전이 감지
            if probe.has("T4_publish") and not probe.has("T5_playing"):
                sv = getattr(g1.speaker_state, "value", None)
                if sv is not None and bool(getattr(sv, "playing", False)):
                    probe.mark("T5_playing")
                    probe.summarize()
                else:
                    age = probe.t4_age()
                    if age is not None and age > T5_TIMEOUT_S:
                        probe.summarize(t5_missing=True)
    except KeyboardInterrupt:
        pass
    finally:
        # 역순 종료 (run.py _stop_runtime 와 동일한 순서)
        stop_event.set()
        bg_thread.join(timeout=2.0)
        for c in (sound_sensor, task_srv, tts, stt, g1):
            try:
                c.stop()
            except Exception:
                logging.exception("stop 실패: %s", type(c).__name__)
        logging.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
