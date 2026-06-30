"""
TeleopControlLoop

50 Hz 제어 루프:
    G1ObsProvider.update()
    → TeleopPolicyProvider.build()  (encoder 1762→64, decoder 994→29)
    → q_target[29]
    → JointCmd (1 step)
    → UnitreeG1Provider.publish_joint_cmd_low()
    → onboard motor_controller → Unitree SDK PD 제어

arm_only=True 시 팔 관절(15-28)만 전송 — 하체는 loco SDK 가 담당.

사용법:
    loop = TeleopControlLoop(g1_provider, vr_coord, g1_obs)
    loop = TeleopControlLoop(g1_provider, vr_coord, g1_obs, arm_only=True)
    loop.run()   # Ctrl-C 로 종료
"""

import logging
import threading
import time
from typing import List, Optional

import numpy as np

from .g1_obs_provider import ALL_JOINT_NAMES
from .policy_params import KPS, KDS
from .teleop_encoder_input_provider import TeleopEncoderInputProvider
from .teleop_policy_provider import TeleopPolicyProvider, TeleopPolicyOutput
from .g1_obs_provider import G1ObsProvider
from .vr_coord_provider import VRCoordProvider

logger = logging.getLogger(__name__)

CONTROL_HZ = 50
_CHUNK_ID_MAX = 255

# 팔 관절 인덱스 (MuJoCo 순서, waist 포함)
_ARM_INDICES = list(range(12, 29))   # 12-14 waist, 15-28 arms
_ARM_NAMES = [ALL_JOINT_NAMES[i] for i in _ARM_INDICES]
_ARM_KPS   = KPS[_ARM_INDICES]
_ARM_KDS   = KDS[_ARM_INDICES]


def _make_joint_cmd(
    joint_names: List[str],
    q_target: np.ndarray,
    kps: np.ndarray,
    kds: np.ndarray,
    chunk_id: int,
) -> dict:
    """q_target → JointCmd (onboard inbound_relay 타입)."""
    n = len(joint_names)
    return {
        "joint_names": joint_names,
        "q":      q_target.tolist(),
        "dq":     [0.0] * n,
        "kp":     kps.tolist(),
        "kd":     kds.tolist(),
        "tau_ff": [0.0] * n,
        "mode":   1,
        "weight": 1.0,
        "chunk_id":   chunk_id,
        "step_index": 0,
    }


class TeleopControlLoop:
    """
    VR teleop 50 Hz 제어 루프.

    arm_only=True: 팔 관절(waist+arms, 17 joints)만 전송.
                   하체는 loco SDK 가 별도로 제어.
    arm_only=False: 전체 29 관절 전송 (whole-body policy).
    """

    def __init__(
        self,
        g1_provider,
        vr_coord: VRCoordProvider,
        g1_obs: G1ObsProvider,
        model_dir: Optional[str] = None,
        control_hz: float = CONTROL_HZ,
        arm_only: bool = False,
    ):
        self._g1 = g1_provider
        self._g1_obs = g1_obs
        self._hz = control_hz
        self._dt = 1.0 / control_hz
        self._arm_only = arm_only

        enc_prov = TeleopEncoderInputProvider(vr_coord, g1_obs)
        kwargs = {"model_dir": model_dir} if model_dir else {}
        self._policy = TeleopPolicyProvider(enc_prov, g1_obs, **kwargs)

        self._running = False
        self._chunk_id = 1
        self._step_count = 0
        self._last_out: Optional[TeleopPolicyOutput] = None

    def run(self) -> None:
        """블로킹 제어 루프. Ctrl-C 또는 stop() 으로 종료."""
        self._running = True
        mode_str = "arm-only" if self._arm_only else "whole-body"
        logger.info("TeleopControlLoop: started at %.0f Hz (%s)", self._hz, mode_str)
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
            logger.info("TeleopControlLoop: stopped (steps=%d)", self._step_count)

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="teleop_ctrl", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._running = False

    @property
    def last_output(self) -> Optional[TeleopPolicyOutput]:
        return self._last_out

    @property
    def step_count(self) -> int:
        return self._step_count

    def _step(self) -> None:
        self._g1_obs.update()

        out = self._policy.build()
        if out is None:
            logger.debug("TeleopControlLoop: policy.build() returned None, skip")
            return

        self._last_out = out
        self._step_count += 1

        if self._arm_only:
            q = out.q_target[_ARM_INDICES]
            cmd = _make_joint_cmd(_ARM_NAMES, q, _ARM_KPS, _ARM_KDS, self._chunk_id)
        else:
            cmd = _make_joint_cmd(
                list(ALL_JOINT_NAMES), out.q_target, KPS, KDS, self._chunk_id,
            )

        self._g1.publish_joint_cmd_low(cmd)

        self._chunk_id = (self._chunk_id % _CHUNK_ID_MAX) + 1
        if self._chunk_id == 0:
            self._chunk_id = 1
