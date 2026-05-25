"""
VLA Provider [TASK-40, REQ-31/39/43]

Vendor-agnostic VLA client. Default backend: NVIDIA GR00T N1.7 (3B) whole-body,
served by KIST Model Server.

- Whole-body locomotion + manipulation (CONV-005).
- ~15 Hz chunk emission, 16-step chunks → 100 Hz step replay on PC (CONV-006).
  JointCmd carries (chunk_id, step_index) so NX can detect chunk boundaries
  for an optional crossfade fallback (default OFF — canonical crossfade is here).
- GearSonic balance-correction lives inside this provider (CONV-007).
  Placement TBD (PC GPU / separate Jetson / NX); external interface stable.

TODO(REQ-39) [TASK-40]: __init__ should fetch UnitreeG1Provider() (CONV-010);
                        no bind() needed since dep is @singleton.
TODO(REQ-39) [TASK-40]: connect to KIST Model Server (gRPC/HTTP).
TODO(REQ-31) [TASK-40]: obs assembly (RGB + 29-DoF joint + IMU base/ankle L/R).
TODO(REQ-31) [TASK-40]: chunk decode → 100 Hz step replay + PC-side crossfade.
TODO(REQ-43) [TASK-40]: GearSonic stage (spec deferred — KIST 단장님 학생들).
TODO(REQ-39) [TASK-40]: safety clip (joint_delta_clip_rad) before publish.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .singleton import singleton


class VLABackend(str, Enum):
    """Vision-Language-Action backend selector."""

    GROOT_N17 = "groot_n17"
    # OPENVLA   = "openvla"     # OpenVLA (future)
    # PI0       = "pi0"          # π0 (future)


@dataclass
class VLAConfig:
    """VLA Provider runtime configuration."""

    backend: VLABackend = VLABackend.GROOT_N17
    # Connection to KIST Model Server (KIST owns server, we own client).
    model_server_url: str = "http://localhost:8000"
    model_server_protocol: str = "http"      # "http" | "grpc"
    request_timeout_s: float = 1.0
    # Inference + replay rates (CONV-006).
    action_horizon: int = 16                 # steps per chunk
    chunk_emit_rate_hz: float = 15.0         # GR00T property, fixed-ish
    step_replay_rate_hz: float = 100.0       # NX motor loop rate
    # GearSonic balance-correction stage (CONV-007 — spec deferred).
    gearsonic_enabled: bool = True           # identity passthrough until wired
    gearsonic_device: str = "cuda:0"         # "cuda:0" | "cuda:1" | "cpu"
    # Safety envelope (rejected before leaving the provider).
    joint_delta_clip_rad: float = 0.25
    # Auth / device for the local client side.
    api_key_env: str = "KIST_VLA_API_KEY"


@singleton
class VLAProvider:
    """
    Vendor-agnostic Vision-Language-Action provider.

    Default backend: KIST GR00T N1.7 Model Server (REQ-39). Consumes
    sub-task prompts from the Move Connector, bundles observation
    (camera + 29-DoF joint state + IMU base/ankle L/R) from the UnitreeG1
    Provider, requests a 16-step action chunk from the model server,
    passes the chunk through the GearSonic balance-correction stage
    (CONV-007), then unpacks the chunk into step-level JointCmd messages
    at 100 Hz for the UnitreeG1 Provider publish path.
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        """
        Parameters
        ----------
        config : VLAConfig, optional
            Runtime configuration. Defaults to GR00T N1.7, 15 Hz chunks,
            100 Hz step replay, GearSonic enabled.
        """
        self._config = config or VLAConfig()
        self._running = False
        # Bound late in start() so the UnitreeG1 Provider singleton is up.
        self._unitree_g1 = None
        self._current_chunk_id = 0
        logging.info(
            "VLAProvider: skeleton initialized (backend=%s, server=%s, "
            "chunk_hz=%.1f, replay_hz=%.1f, gearsonic=%s)",
            self._config.backend.value,
            self._config.model_server_url,
            self._config.chunk_emit_rate_hz,
            self._config.step_replay_rate_hz,
            self._config.gearsonic_enabled,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Connect to the KIST Model Server and bind to UnitreeG1 Provider."""
        # TODO(REQ-39) [TASK-40]: resolve UnitreeG1 Provider singleton
        # TODO(REQ-39) [TASK-40]: open gRPC/HTTP client to model server
        # TODO(REQ-43) [TASK-40]: load GearSonic model (if local) or open
        #                         RPC to the remote device (if separate)
        # TODO(REQ-31) [TASK-40]: spawn step-replay worker thread @ 100 Hz
        raise NotImplementedError("VLAProvider.start: TBD [TASK-40]")

    def stop(self) -> None:
        """Cancel in-flight inference, drain replay queue, close clients."""
        # TODO(REQ-39) [TASK-40]: cancel request, join worker, close client
        raise NotImplementedError("VLAProvider.stop: TBD [TASK-40]")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def infer(self, sub_task_prompt: str) -> int:
        """
        Request one VLA chunk for ``sub_task_prompt`` and enqueue its
        steps for the 100 Hz replay loop.

        Observation (camera + joint state + IMU) is read live from the
        UnitreeG1 Provider — callers do NOT pass it in. This keeps the
        Move Connector API minimal (prompt-only).

        Parameters
        ----------
        sub_task_prompt : str
            Natural-language sub-task from the Move Connector
            (TaskSrvProvider scenario step).

        Returns
        -------
        int
            ``chunk_id`` assigned to the resulting 16-step chunk. The
            ``step_index`` runs 0..15 inside the chunk. Step-level
            JointCmd messages will carry ``(chunk_id, step_index)``.
        """
        # TODO(REQ-31) [TASK-40]: snapshot obs from UnitreeG1 Provider
        # TODO(REQ-39) [TASK-40]: send (prompt, obs) to KIST Model Server
        # TODO(REQ-43) [TASK-40]: pipe chunk through GearSonic stage
        # TODO(REQ-39) [TASK-40]: safety clip (joint_delta_clip_rad)
        # TODO(REQ-31) [TASK-40]: enqueue 16 steps with new chunk_id; the
        #                         replay loop will publish them at 100 Hz
        raise NotImplementedError("VLAProvider.infer: TBD [TASK-40]")

    def cancel_chunk(self) -> None:
        """Drop the currently replaying chunk (called on E-STOP / new prompt)."""
        # TODO(REQ-31) [TASK-40]: flush replay queue, hold joint state
        raise NotImplementedError("VLAProvider.cancel_chunk: TBD [TASK-40]")

    # ------------------------------------------------------------------
    # Internals — GearSonic seam (CONV-007)
    # ------------------------------------------------------------------
    def _gearsonic_correct(self, chunk: Any) -> Any:
        """
        Post-VLA balance correction (GearSonic stage).

        Spec deferred — see CONV-007 + REQ-43 (KIST 단장님 학생들 담당).
        For now this is the seam where the model hooks in; default
        behaviour is identity passthrough so the rest of the pipeline
        can be built and tested.
        """
        # TODO(REQ-43) [TASK-40]: when spec lands, run model here:
        #     corrected = self._gearsonic(chunk, imu_base, imu_ankle_l, imu_ankle_r)
        # TODO(REQ-43) [TASK-40]: if placement moves off-PC (Jetson / NX),
        #                         keep this method's signature; only the
        #                         transport changes (local call → RPC).
        return chunk  # identity passthrough placeholder

    # ------------------------------------------------------------------
    # Internals — chunk → step unpack (CONV-006)
    # ------------------------------------------------------------------
    def _enqueue_chunk_steps(self, chunk: Any, chunk_id: int) -> None:
        """
        Unpack a 16-step chunk into the 100 Hz replay queue.

        Each enqueued step carries ``(chunk_id, step_index)`` so the NX
        ``motor_controller`` ring-buffer can detect chunk boundaries for
        the optional crossfade fallback (default OFF per CONV-006).
        """
        # TODO(REQ-31) [TASK-40]: drain pending queue, push 16 fresh steps
        # TODO(REQ-31) [TASK-40]: PC-side crossfade between chunk_id N and
        #                         N+1 (canonical per CONV-006)
        raise NotImplementedError("VLAProvider._enqueue_chunk_steps: TBD [TASK-40]")
