"""
TeleopPolicyProvider

TeleopEncoderInputProvider (encoder input 1762 dims) 와
G1ObsProvider (decoder history) 를 합쳐 ONNX encoder + decoder 를 돌려
action (29 joints, MuJoCo 순서) 을 생성한다.

inference 흐름:
    1. TeleopEncoderInputProvider.build()  → encoder_input (1762,)
    2. ONNX encoder                        → token (64,)
    3. decoder input 조립 (994,):
         [  0:  64]  token_state                           [64]
         [ 64:  94]  his_base_angular_velocity_10f_s1      [30] = 10×3
         [ 94: 384]  his_body_joint_positions_10f_s1       [290] = 10×29
         [384: 674]  his_body_joint_velocities_10f_s1      [290]
         [674: 964]  his_last_actions_10f_s1               [290] = 10×29
         [964: 994]  his_gravity_dir_10f_s1                [30] = 10×3
    4. ONNX decoder                        → action_raw (29,) MuJoCo 순서
    5. q_target = DEFAULT_ANGLES + action_raw * ACTION_SCALE  (policy_params.py)

action_history 버퍼 (maxlen=10) 는 매 build() 호출 후 갱신.
G1ObsData 에서 his_* 를 직접 사용하므로 별도 버퍼 불필요 (joint pos/vel/gravity/ang_vel).
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .policy_params import ACTION_SCALE, DEFAULT_ANGLES

logger = logging.getLogger(__name__)

# ONNX 모델 기본 경로 (src/policy 디렉토리)
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "policy")

DECODER_INPUT_DIM = 994
ACTION_DIM = 29
N_FRAMES = 10


@dataclass
class TeleopPolicyOutput:
    """한 스텝 policy 추론 결과."""

    action: np.ndarray     # (29,) float32, raw decoder 출력 (MuJoCo 순서)
    q_target: np.ndarray   # (29,) float32, DEFAULT_ANGLES + action * ACTION_SCALE
    token: np.ndarray      # (64,) float32, 디버깅용
    vr_ok: bool
    g1_ok: bool
    inference_time_ms: float


class TeleopPolicyProvider:
    """
    ONNX encoder + decoder 를 사용해 teleop action (29,) 을 추론한다.

    사용법:
        policy = TeleopPolicyProvider(enc_prov, g1_obs)
        # 제어 루프
        g1_obs.update()
        out: TeleopPolicyOutput | None = policy.build()
        if out:
            apply_action(out.action)
    """

    def __init__(
        self,
        enc_prov,    # TeleopEncoderInputProvider
        g1_obs,      # G1ObsProvider
        model_dir: str = _DEFAULT_MODEL_DIR,
    ):
        self._enc_prov = enc_prov
        self._g1_obs = g1_obs

        import onnxruntime as ort
        encoder_path = os.path.join(model_dir, "model_encoder.onnx")
        decoder_path = os.path.join(model_dir, "model_decoder.onnx")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2

        self._encoder = ort.InferenceSession(encoder_path, sess_options=opts)
        self._decoder = ort.InferenceSession(decoder_path, sess_options=opts)

        logger.info("TeleopPolicyProvider: loaded encoder=%s decoder=%s", encoder_path, decoder_path)

        # action 히스토리 버퍼 (appendleft → [0] = 최신)
        self._action_history: deque = deque(maxlen=N_FRAMES)
        _zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(N_FRAMES):
            self._action_history.appendleft(_zero_action.copy())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Optional[TeleopPolicyOutput]:
        """
        최신 데이터로 encoder → decoder 추론을 수행하고 action (29,) 을 반환한다.

        반환: TeleopPolicyOutput, 또는 encoder input 생성 실패 시 None.
        """
        t0 = time.monotonic()

        # 1. encoder input 조립
        enc_input = self._enc_prov.build()
        if enc_input is None:
            return None

        # 2. encoder 추론 → token (64,)
        token = self._encoder.run(
            None,
            {"obs_dict": enc_input.encoder_input[np.newaxis, :].astype(np.float32)},
        )[0][0]   # (64,)

        # 3. decoder input 조립 (994,)
        g1_data = self._g1_obs.get_latest()

        if g1_data is not None:
            ang_vel_10f = g1_data.base_ang_vel_10f_s1    # (30,)
            pos_all_10f = g1_data.joint_pos_all_10f_s1   # (290,)
            vel_all_10f = g1_data.joint_vel_all_10f_s1   # (290,)
            gravity_10f = g1_data.gravity_dir_10f_s1     # (30,)
        else:
            ang_vel_10f = np.zeros(30, dtype=np.float32)
            pos_all_10f = np.zeros(290, dtype=np.float32)
            vel_all_10f = np.zeros(290, dtype=np.float32)
            gravity_10f = np.tile([0.0, 0.0, -1.0], 10).astype(np.float32)

        action_10f = np.concatenate(list(self._action_history)).astype(np.float32)  # (290,)

        decoder_input = np.concatenate([
            token,        # [  0:  64]
            ang_vel_10f,  # [ 64:  94]
            pos_all_10f,  # [ 94: 384]
            vel_all_10f,  # [384: 674]
            action_10f,   # [674: 964]
            gravity_10f,  # [964: 994]
        ]).astype(np.float32)

        assert decoder_input.shape[0] == DECODER_INPUT_DIM, \
            f"decoder input dim mismatch: {decoder_input.shape[0]} != {DECODER_INPUT_DIM}"

        # 4. decoder 추론 → action (29,)
        action = self._decoder.run(
            None,
            {"obs_dict": decoder_input[np.newaxis, :].astype(np.float32)},
        )[0][0]   # (29,)

        # 5. q_target 계산
        action_f32 = action.astype(np.float32)
        q_target = DEFAULT_ANGLES + action_f32 * ACTION_SCALE

        # 6. action history 갱신 (최신 action 을 앞에 추가)
        self._action_history.appendleft(action_f32.copy())

        inf_ms = (time.monotonic() - t0) * 1000.0

        return TeleopPolicyOutput(
            action=action_f32,
            q_target=q_target,
            token=token.astype(np.float32),
            vr_ok=enc_input.vr_ok,
            g1_ok=enc_input.g1_ok,
            inference_time_ms=inf_ms,
        )

    def reset_action_history(self) -> None:
        """action 히스토리를 zeros 로 초기화한다 (에피소드 재시작 시 사용)."""
        self._action_history.clear()
        _zero = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(N_FRAMES):
            self._action_history.appendleft(_zero.copy())
