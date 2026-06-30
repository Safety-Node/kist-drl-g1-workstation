"""
TeleopControlLoop

50 Hz 제어 루프:
    G1ObsProvider.update()
    → TeleopPolicyProvider.build()  (encoder 1762→64, decoder 994→29)
    → q_target[29]
    → JointCmdChunk (1 step)
    → UnitreeG1Provider.publish_joint_chunk_low()
    → onboard motor_controller → Unitree SDK PD 제어

사용법:
    loop = TeleopControlLoop(g1_provider, vr_coord, g1_obs)
    loop.run()          # Ctrl-C 로 종료
    loop.stop()         # 외부에서 종료
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from .g1_obs_provider import ALL_JOINT_NAMES
from .policy_params import KPS, KDS
from .teleop_encoder_input_provider import TeleopEncoderInputProvider
from .teleop_policy_provider import TeleopPolicyProvider, TeleopPolicyOutput
from .g1_obs_provider import G1ObsProvider
from .vr_coord_provider import VRCoordProvider

logger = logging.getLogger(__name__)

CONTROL_HZ = 50
_DT = 1.0 / CONTROL_HZ

# chunk_id: 1~255, 0 은 skip (wrap rule)
_CHUNK_ID_MAX = 255


def _make_joint_cmd(
    joint_names: list,
    q_target: np.ndarray,
    kps: np.ndarray,
    kds: np.ndarray,
    chunk_id: int,
) -> dict:
    """q_target (29,) → JointCmd (단일 스텝, onboard inbound_relay 타입)."""
    n = len(joint_names)
    return {
        "joint_names": joint_names,
        "q":      q_target.tolist(),
        "dq":     [0.0] * n,
        "kp":     kps.tolist(),
        "kd":     kds.tolist(),
        "tau_ff": [0.0] * n,
        "mode":   1,       # position PD
        "weight": 1.0,
        "chunk_id":   chunk_id,
        "step_index": 0,
    }


class TeleopControlLoop:
    """
    VR teleop 50 Hz 제어 루프.

    g1_provider.publish_joint_chunk_low() 로 /bridge/cmd/low 에 JointCmdChunk 를 전송한다.
    onboard motor_controller 가 수신해 Unitree SDK PD 제어를 실행한다.
    """

    def __init__(
        self,
        g1_provider,            # UnitreeG1Provider (이미 start() 된 상태)
        vr_coord: VRCoordProvider,
        g1_obs: G1ObsProvider,
        model_dir: Optional[str] = None,
        control_hz: float = CONTROL_HZ,
    ):
        self._g1 = g1_provider
        self._g1_obs = g1_obs
        self._hz = control_hz
        self._dt = 1.0 / control_hz

        enc_prov = TeleopEncoderInputProvider(vr_coord, g1_obs)
        kwargs = {"model_dir": model_dir} if model_dir else {}
        self._policy = TeleopPolicyProvider(enc_prov, g1_obs, **kwargs)

        self._running = False
        self._chunk_id = 1
        self._step_count = 0
        self._last_out: Optional[TeleopPolicyOutput] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """블로킹 제어 루프. Ctrl-C 또는 stop() 으로 종료."""
        self._running = True
        logger.info("TeleopControlLoop: started at %.0f Hz", self._hz)
        deadline = time.monotonic()

        try:
            while self._running:
                deadline += self._dt
                self._step()
                sleep = deadline - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                elif sleep < -self._dt:
                    logger.warning(
                        "TeleopControlLoop: 루프 지연 %.1f ms (step %d)",
                        -sleep * 1000, self._step_count,
                    )
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            logger.info(
                "TeleopControlLoop: stopped (steps=%d)", self._step_count,
            )

    def start_background(self) -> threading.Thread:
        """백그라운드 스레드로 제어 루프를 시작한다."""
        t = threading.Thread(target=self.run, name="teleop_ctrl", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        """제어 루프를 중단한다."""
        self._running = False

    @property
    def last_output(self) -> Optional[TeleopPolicyOutput]:
        """마지막 추론 결과 (진단용)."""
        return self._last_out

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step(self) -> None:
        self._g1_obs.update()

        out = self._policy.build()
        if out is None:
            logger.debug("TeleopControlLoop: policy.build() returned None, skip")
            return

        self._last_out = out
        self._step_count += 1

        cmd = _make_joint_cmd(
            joint_names=list(ALL_JOINT_NAMES),
            q_target=out.q_target,
            kps=KPS,
            kds=KDS,
            chunk_id=self._chunk_id,
        )
        self._g1.publish_joint_cmd_low(cmd)

        # chunk_id wrap: 1~255, 0 skip
        self._chunk_id = (self._chunk_id % _CHUNK_ID_MAX) + 1
        if self._chunk_id == 0:
            self._chunk_id = 1
