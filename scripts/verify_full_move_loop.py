"""
Full move-loop verification: NX-mic → STT → SoundSensor → TaskSrvProvider →
**MoveConnector(wrap)** → UnitreeG1.publish_twist → NX motor_controller
LocoClient.Move(vx, vy, vyaw).

verify_full_loop.py 의 speak-only 변형. MoveConnector 의 기본 path 3종
(loco preset / nav / VLA) 은 nav 가 NotImplementedError 라 1차시연 검증이
어려움 — 본 스크립트는 MoveConnector.connect 를 wrap 해서 "twist ..."
prefix prompt 를 직접 ``g1.publish_twist`` 로 보내는 path 를 임시 추가한다.
운영 MoveConnector / NavigationProvider 코드는 안 건드림.

지원 프롬프트 (시나리오의 ``move`` action value):
    "twist vx=0.2 duration=2.0"
    "twist vx=-0.15 duration=2.0"
    "twist vyaw=0.3 duration=2.0"
    "twist vx=0.2 vyaw=0.1 duration=3.0"
  - 명시 안 한 필드는 0. duration 동안 10Hz publish, 끝나면 zero twist 1회.

타임로깅 (전부 desktop monotonic 시계):
    T1  STT transcript 방출         (probe: register_transcript_callback)
    T2  TaskSrvProvider.on_audio    (probe: on_audio wrapper)
    T2b 시나리오 trigger 매칭/활성  (probe: _activate wrapper)
    T3  MoveConnector.connect 진입  (probe: connect wrapper)
    T4  g1.publish_twist 첫 호출    (probe: publish_twist wrapper)
    T5  buf_state heartbeat 갱신     (probe: g1.buf_state 폴링 — NX 응답)

probe 는 전부 "원본 호출을 그대로 통과시키는 측정 래퍼" — 동작 변경 없음.

전제 환경:
    source env.sh
    export GOOGLE_APPLICATION_CREDENTIALS=...      # STT
    export NCP_CLOVA_CLIENT_ID=... NCP_CLOVA_CLIENT_SECRET=...   # TTS
    (NX 측 mic_node / speaker_node / motor_controller / comm_bridge 가동)
    로봇이 BalanceStand 자세에 있어야 LocoClient.Move(vel) 가 받는다.

실행 (desktop, repo root):
    uv run python scripts/verify_full_move_loop.py
    uv run python scripts/verify_full_move_loop.py --scenario audio_move_loop_test.json5

사용법:
    실행 후 마이크에:
      1) "이동 테스트 시작"           → 시나리오 활성, 음성 안내 재생
      2) "앞으로" / "전진"            → vx=+0.2  m/s, 2초 전진
      3) "뒤로" / "후진"              → vx=-0.15 m/s, 2초 후진
      4) "왼쪽으로 돌아" / "좌회전"   → vyaw=+0.3 rad/s, 2초 회전
      5) "오른쪽으로 돌아" / "우회전" → vyaw=-0.3 rad/s, 2초 회전
      6) "시연 끝" 또는 Ctrl-C        → 종료
"""

import argparse
import asyncio
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
from actions.move.connector.move_connector import MoveConnector        # noqa: E402
from actions.speak.connector.speak_connector import SpeakConnector     # noqa: E402
from backgrounds.plugins.task_srv_bg import TaskSrvBg, TaskSrvBgConfig # noqa: E402
from inputs.plugins.sound_sensor import SoundSensor, SoundSensorConfig # noqa: E402
from providers.stt_provider import STTConfig, STTProvider, TranscriptEvent  # noqa: E402
from providers.task_srv_provider import TaskSrvConfig, TaskSrvProvider # noqa: E402
from providers.tts_provider import TTSConfig, TTSProvider              # noqa: E402
from providers.unitree_g1_provider import UnitreeG1Provider            # noqa: E402

T5_TIMEOUT_S = 8.0       # T4 이후 buf_state 갱신 대기 한도
TWIST_PUBLISH_HZ = 10.0  # twist 연속 발행 주기 (LocoClient.Move 가 streaming 입력)


def _parse_twist_prompt(prompt: str) -> Optional[dict]:
    """``twist vx=... vy=... vyaw=... duration=...`` 파싱. 미인식 시 None."""
    s = prompt.strip().lower()
    if not s.startswith("twist"):
        return None
    out = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "duration": 1.0}
    for tok in s.split()[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in out:
            try:
                out[k] = float(v)
            except ValueError:
                logging.warning("twist parse: %r 의 %s 값 %r 파싱 실패", prompt, k, v)
    # 안전 클램프
    out["vx"] = max(-0.5, min(0.5, out["vx"]))
    out["vy"] = max(-0.3, min(0.3, out["vy"]))
    out["vyaw"] = max(-0.8, min(0.8, out["vyaw"]))
    out["duration"] = max(0.1, min(5.0, out["duration"]))
    return out


class MoveCycleProbe:
    """한 음성-이동 사이클(T1~T5)의 타임스탬프 수집 + 요약 출력. thread-safe."""

    STAGES = ("T1_transcript", "T2_on_audio", "T2b_activate",
              "T3_move_connect", "T4_publish_twist", "T5_buf_state")

    def __init__(self):
        self._lock = threading.Lock()
        self._cycle = 0
        self._ts: dict = {}
        self._text: str = ""
        self._twist_label: str = ""

    def mark(self, stage: str, text: str = "", twist_label: str = "") -> None:
        now = time.monotonic()
        with self._lock:
            if stage == "T1_transcript":
                self._cycle += 1
                self._ts = {}
                self._text = text
                self._twist_label = ""
            if twist_label and not self._twist_label:
                self._twist_label = twist_label
            if stage not in self._ts:
                self._ts[stage] = now

    def has(self, stage: str) -> bool:
        with self._lock:
            return stage in self._ts

    def t4_age(self) -> Optional[float]:
        with self._lock:
            t4 = self._ts.get("T4_publish_twist")
        return (time.monotonic() - t4) if t4 is not None else None

    def t4_baseline_buf_ts(self) -> Optional[float]:
        with self._lock:
            return self._ts.get("_buf_baseline_ts")

    def set_buf_baseline(self, buf_seen_ts: Optional[float]) -> None:
        with self._lock:
            if "_buf_baseline_ts" not in self._ts:
                self._ts["_buf_baseline_ts"] = buf_seen_ts

    def summarize(self, t5_missing: bool = False) -> None:
        with self._lock:
            ts, cyc, text, label = dict(self._ts), self._cycle, self._text, self._twist_label
            self._ts = {}
        if "T1_transcript" not in ts:
            return
        t1 = ts["T1_transcript"]

        def d(a: str, b: str) -> str:
            if a in ts and b in ts:
                return f"{(ts[b] - ts[a]) * 1000:7.0f} ms"
            return "     -- "

        head = f" [{label}]" if label else ""
        logging.info("==== cycle #%d timing  (%r)%s ====", cyc, text, head)
        logging.info("  T1→T2  SoundSensor 전달        : %s", d("T1_transcript", "T2_on_audio"))
        logging.info("  T2→T2b tick 픽업+trigger 매칭  : %s", d("T2_on_audio", "T2b_activate"))
        logging.info("  T2b→T3 dispatch→connect       : %s", d("T2b_activate", "T3_move_connect"))
        logging.info("  T3→T4  routing→publish_twist  : %s", d("T3_move_connect", "T4_publish_twist"))
        logging.info("  T4→T5  publish→buf_state 갱신 : %s", d("T4_publish_twist", "T5_buf_state"))
        if "T5_buf_state" in ts:
            logging.info("  T1→T5  전체 응답               : %7.0f ms", (ts["T5_buf_state"] - t1) * 1000)
        elif t5_missing:
            logging.warning(
                "  T5 미수신 — motor_controller/comm_bridge 미동작 또는 buf_state 미발행 의심"
            )
        logging.info("=" * 48)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="audio_move_loop_test.json5",
                    help="config/scenarios/<이름>.json5 (기본: audio_move_loop_test.json5)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s", force=True)

    probe = MoveCycleProbe()

    # ── 와이어링 (run.py CONV-010 순서: UnitreeG1 FIRST) ───────────────────
    g1 = UnitreeG1Provider()
    stt = STTProvider(STTConfig())
    tts = TTSProvider(TTSConfig())
    speak_conn = SpeakConnector(ActionConfig())
    # MoveConnector __init__ 가 VLA/Navigation 싱글턴을 만들지만 둘 다 본
    # 시나리오에서는 호출 안 됨 — wrap 한 connect 가 twist prompt 만 처리.
    move_conn = MoveConnector(ActionConfig())
    task_srv = TaskSrvProvider(TaskSrvConfig(scenario_file=args.scenario))
    task_srv.bind(move_connector=move_conn, speak_connector=speak_conn)
    sound_sensor = SoundSensor(SoundSensorConfig())

    # ── 타이밍 probe 설치 ─────────────────────────────────────────────────
    _orig_on_audio = task_srv.on_audio
    _orig_activate = task_srv._activate
    _orig_move_connect = move_conn.connect
    _orig_publish_twist = g1.publish_twist
    _orig_stt_chunk = stt._on_audio_chunk

    # T0 진단 — STT 에 PCM chunk 들어오는지 확인용 카운터
    _stt_chunk_count = {"n": 0}

    def _p_on_audio(text, ts=None):
        probe.mark("T2_on_audio")
        return _orig_on_audio(text, ts)

    def _p_activate(scenario):
        probe.mark("T2b_activate")
        return _orig_activate(scenario)

    def _p_publish_twist(vx, vy, vyaw):
        # 한 burst 의 첫 publish 만 T4 로 기록 (twist 는 duration 동안 반복 호출됨)
        if not probe.has("T4_publish_twist"):
            label = f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}"
            probe.mark("T4_publish_twist", twist_label=label)
            cache = getattr(g1, "buf_state", None)
            baseline = getattr(cache, "last_seen_ts", 0.0) if cache is not None else 0.0
            probe.set_buf_baseline(baseline)
            logging.info("publish_twist[first] → vx=%+.2f vy=%+.2f vyaw=%+.2f", vx, vy, vyaw)
        return _orig_publish_twist(vx, vy, vyaw)

    async def _twist_burst(params: dict) -> None:
        """params 의 (vx,vy,vyaw) 를 TWIST_PUBLISH_HZ 로 duration 초 publish, 종료 시 zero."""
        period = 1.0 / TWIST_PUBLISH_HZ
        end = time.monotonic() + params["duration"]
        try:
            while time.monotonic() < end:
                g1.publish_twist(params["vx"], params["vy"], params["vyaw"])
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            logging.info("twist burst: cancelled")
            raise
        finally:
            # 정지 — 잔여 속도 누적 방지. 한 번만 publish (LocoClient 가 zero 받으면 정지)
            try:
                g1.publish_twist(0.0, 0.0, 0.0)
            except Exception:
                logging.exception("twist zero publish 실패")

    async def _p_move_connect(output_interface):
        probe.mark("T3_move_connect")
        prompt = output_interface.action
        logging.info("MoveConnector dispatch: %r", prompt)
        # twist prompt 만 가로채서 publish_twist 로 직접 분기.
        # 그 외 prompt 는 원래 MoveConnector 로 통과 (loco/nav/VLA 라우팅 유지).
        params = _parse_twist_prompt(prompt)
        if params is not None:
            logging.info(
                "twist path: vx=%+.2f vy=%+.2f vyaw=%+.2f dur=%.2fs (%.0fHz)",
                params["vx"], params["vy"], params["vyaw"], params["duration"], TWIST_PUBLISH_HZ,
            )
            await _twist_burst(params)
            return
        return await _orig_move_connect(output_interface)

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
    move_conn.connect = _p_move_connect          # type: ignore[assignment]
    g1.publish_twist = _p_publish_twist          # type: ignore[assignment]
    stt._on_audio_chunk = _p_stt_chunk           # type: ignore[assignment]

    # ── 기동 (run.py 의존 순서) ───────────────────────────────────────────
    stop_event = threading.Event()
    g1.start()
    stt.start()
    tts.start()

    if not getattr(stt, "_audio_cb_registered", False):
        logging.error(
            "전제 미충족: UnitreeG1Provider.register_audio_callback 미구현(TASK-41). "
            "TASK-41 머지 후 다시 실행하세요.")
        g1.stop()
        return 2

    task_srv.start()

    bg = TaskSrvBg(TaskSrvBgConfig())
    bg.set_stop_event(stop_event)
    bg_thread = threading.Thread(target=bg.run, name="TaskSrvBg", daemon=True)
    bg_thread.start()

    def _on_transcript(ev: TranscriptEvent) -> None:
        probe.mark("T1_transcript", ev.text)
        logging.info("TRANSCRIPT: %r", ev.text)

    stt.register_transcript_callback(_on_transcript)
    sound_sensor.start()

    logging.info(
        "full move-loop ready (scenario=%s) — 로봇이 BalanceStand 자세여야 합니다. "
        "마이크에 \"이동 테스트 시작\" 후 \"앞으로/뒤로/왼쪽으로 돌아/오른쪽으로 돌아\". "
        "Ctrl-C 종료.", args.scenario,
    )

    # ── 메인 루프: T5(buf_state 갱신) 폴링 + 사이클 요약 ─────────────────
    try:
        while True:
            time.sleep(0.02)
            if probe.has("T4_publish_twist") and not probe.has("T5_buf_state"):
                cache = getattr(g1, "buf_state", None)
                cur_ts = getattr(cache, "last_seen_ts", 0.0) if cache is not None else 0.0
                baseline = probe.t4_baseline_buf_ts() or 0.0
                if cur_ts > 0.0 and cur_ts > baseline:
                    probe.mark("T5_buf_state")
                    probe.summarize()
                else:
                    age = probe.t4_age()
                    if age is not None and age > T5_TIMEOUT_S:
                        probe.summarize(t5_missing=True)
    except KeyboardInterrupt:
        pass
    finally:
        # 안전 — 종료 직전 zero twist 한 번 보내 잔여 속도 방지
        try:
            _orig_publish_twist(0.0, 0.0, 0.0)
        except Exception:
            pass
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
