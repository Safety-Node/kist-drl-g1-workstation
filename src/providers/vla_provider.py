"""
VLA Provider [TASK-40, REQ-31/39/43]

Vendor-agnostic VLA client. Default backend: NVIDIA GR00T N1.7 (3B) whole-body,
served by KIST Model Server.

- Whole-body locomotion + manipulation (CONV-005).
- ~15 Hz chunk emission (assumed — KIST L40 ~63.9 ms / chunk, RTX 4090
  unmeasured; source TBD), `action_horizon`-step chunks (default 16 per
  KIST fine-tune; NVIDIA example default 8) → 100 Hz step replay on PC
  (CONV-006 is canonical). JointCmd carries (chunk_id, step_index) so NX
  can detect chunk boundaries for an optional crossfade fallback
  (default OFF — canonical crossfade is here).
- GearSonic balance-correction lives inside this provider (CONV-007).
  Placement TBD (PC GPU / separate Jetson / NX); external interface stable.

Pipeline ordering (LOCKED for safety):
    VLA chunk  →  GearSonic correct  →  safety clip  →  enqueue  →  publish
GearSonic precedes the clip so the balance-corrected joint targets are still
subject to the safety envelope; clipping the VLA output before GearSonic
would let the balance stage produce out-of-envelope corrections silently.

TODO(REQ-39) [TASK-40]: connect to KIST Model Server (gRPC/HTTP).
TODO(REQ-31) [TASK-40]: obs assembly (RGB + 29-DoF joint + IMU base/ankle L/R).
TODO(REQ-31) [TASK-40]: chunk decode → 100 Hz step replay + PC-side crossfade.
TODO(REQ-43) [TASK-40]: GearSonic stage (spec deferred — KIST 단장님 학생들).
TODO(REQ-39) [TASK-40]: safety clip (joint_delta_clip_rad) — AFTER GearSonic.
TODO(REQ-39) [TASK-40]: E-STOP — start() subscribes self._on_estop via
                        unitree_g1.register_estop_callback. On active=True:
                        cancel_chunk() + block subsequent infer() calls.
                        E-STOP clear does NOT auto-resume (TaskSrvProvider
                        decides next scenario / posture).
TODO(REQ-39) [TASK-40]: arm vs low joint split — define _ARM_JOINTS /
                        _LOW_JOINTS frozensets (per ICD IF-6 + 2026-05-22
                        lower-body addition) and route step joint subsets
                        to publish_joint_cmd_arm / publish_joint_cmd_low.
TODO(REQ-39) [TASK-40]: request_timeout_s policy — on model-server timeout
                        the loop should retry up to 2x (~1 s total) before
                        surfacing the failure to TaskSrvProvider; a single
                        lag spike must NOT fail the sub-task. Last-known
                        chunk keeps replaying during the retry window.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from .singleton import singleton
from .unitree_g1_provider import UnitreeG1Provider


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
    model_server_protocol: Literal["http", "grpc"] = "http"
    # Per-request timeout. On exceeded: retry up to 2 times before failing
    # the sub-task (see TODO in module docstring); last-known chunk keeps
    # replaying during the retry window.
    request_timeout_s: float = 1.0
    # Inference + replay rates (CONV-006 — canonical wording).
    # Steps per chunk. Currently 16 per KIST fine-tune choice (NVIDIA's
    # public example default is 8 — source for KIST's 16 is TBD).
    action_horizon: int = 16
    # GR00T model property — actual emission rate is set by model-server
    # inference latency: KIST L40 measured ~63.9 ms / chunk (~15.6 Hz,
    # source TBD); RTX 4090 unmeasured. NVIDIA-published H100 TensorRT
    # 27.9 ms / 35.9 Hz; with TensorRT applied on the 4090, 30-50 Hz is
    # plausible. Setting this value higher does NOT make the model emit
    # faster; informational only, kept so config reviewers see the
    # design assumption (conservative).
    chunk_emit_rate_hz: float = 15.0
    step_replay_rate_hz: float = 100.0       # NX motor loop rate
    # GearSonic balance-correction stage (CONV-007 — spec deferred).
    gearsonic_enabled: bool = True           # identity passthrough until wired
    gearsonic_device: Literal["cuda:0", "cuda:1", "cpu"] = "cuda:0"
    # Safety envelope (rejected before leaving the provider). Applied
    # AFTER GearSonic per the pipeline-ordering lock above.
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
    Provider, requests an ``action_horizon``-step action chunk from the
    model server (default 16 per KIST fine-tune; NVIDIA example default
    8, source TBD), passes the chunk through the GearSonic
    balance-correction stage (CONV-007), then unpacks the chunk into
    step-level JointCmd messages at 100 Hz for the UnitreeG1 Provider
    publish path. CONV-006 is the canonical wording for these rates.
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        """
        Parameters
        ----------
        config : VLAConfig, optional
            Runtime configuration. Defaults to GR00T N1.7,
            ``action_horizon=16`` (KIST config; NVIDIA example default
            is 8 — source TBD), assumed ~15 Hz chunk emission
            (KIST L40 ~63.9 ms / chunk, RTX 4090 unmeasured — TBD),
            100 Hz step replay, GearSonic enabled. See CONV-006.
        """
        self._config = config or VLAConfig()
        self._running = False
        # CONV-010: UnitreeG1Provider is @singleton — fetched here.
        # run.py MUST construct UnitreeG1 before VLAProvider so this
        # returns the configured instance, not a default-config singleton.
        self._unitree_g1 = UnitreeG1Provider()
        self._current_chunk_id = 0
        # Set by _on_estop callback; infer() gates on it (E-STOP active
        # blocks all new dispatch until cleared + TaskSrvProvider decides
        # what to do next).
        self._estop_active = False
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
        """Connect to the KIST Model Server, subscribe E-STOP, spawn replay worker."""
        # TODO(REQ-39) [TASK-40]: open gRPC/HTTP client to model server
        # TODO(REQ-43) [TASK-40]: load GearSonic model (if local) or open
        #                         RPC to the remote device (if separate)
        # TODO(REQ-39) [TASK-40]: self._unitree_g1.register_estop_callback(self._on_estop)
        # TODO(REQ-31) [TASK-40]: spawn step-replay worker thread @ 100 Hz
        #     (see policy TODOs below for empty-queue / crossfade / preemption /
        #     arm-low split)
        raise NotImplementedError("VLAProvider.start: TBD [TASK-40]")

    def stop(self) -> None:
        """Cancel in-flight inference, drain replay queue, close clients."""
        # TODO(REQ-39) [TASK-40]: unregister_estop_callback, cancel request,
        #                         join worker, close client
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
            ``chunk_id`` assigned to the resulting chunk. The chunk
            contains ``self._config.action_horizon`` steps (currently
            16 per KIST config — see CONV-006; NVIDIA example default
            is 8, source TBD); ``step_index`` runs
            ``0..action_horizon-1`` inside the chunk. Step-level
            JointCmd messages will carry ``(chunk_id, step_index)``.
        """
        # TODO(REQ-31) [TASK-40]: snapshot obs from UnitreeG1 Provider
        # TODO(REQ-39) [TASK-40]: send (prompt, obs) to KIST Model Server
        #                         (with request_timeout_s + retry policy)
        # TODO(REQ-43) [TASK-40]: pipe chunk through GearSonic stage
        # TODO(REQ-39) [TASK-40]: safety clip (joint_delta_clip_rad) — AFTER
        #                         GearSonic per pipeline order lock
        # TODO(REQ-31) [TASK-40]: enqueue action_horizon steps (currently 16
        #                         per CONV-006) with new chunk_id; the replay
        #                         loop publishes them at 100 Hz
        raise NotImplementedError("VLAProvider.infer: TBD [TASK-40]")

    def cancel_chunk(
        self, reason: Literal["prompt change", "estop"] = "prompt change"
    ) -> None:
        """
        Drop the currently replaying chunk.

        Two semantics depending on ``reason``:

        - ``"prompt change"`` (default — TaskSrvProvider switches sub-task):
          flush the replay queue and **hold the last published JointCmd**
          (re-publish at 100 Hz so NX keeps the current pose). The next
          ``infer()`` will overwrite this with fresh steps.

        - ``"estop"`` (E-STOP callback): flush the replay queue and
          **publish a ``Damp`` LocoCommand** via UnitreeG1 — Unitree's
          deterministic damp behaviour is the safe rest state. Do NOT
          rely on hold-last-pose for E-STOP.

        The two paths exist to avoid a footgun: holding the last
        VLA-produced pose during E-STOP can sustain an unsafe joint
        configuration (e.g. arm mid-reach); Damp puts the robot in a
        known-safe state.
        """
        # TODO(REQ-31) [TASK-40]: flush replay queue
        # TODO(REQ-31) [TASK-40]: if reason == "prompt change": switch worker
        #                         to "republish last step" mode
        # TODO(REQ-31) [TASK-40]: if reason == "estop":
        #                         self._unitree_g1.publish_loco_cmd({"name":"Damp"})
        raise NotImplementedError("VLAProvider.cancel_chunk: TBD [TASK-40]")

    # ------------------------------------------------------------------
    # Read-only state (polled by GUI BG)
    # ------------------------------------------------------------------
    @property
    def current_chunk_id(self) -> int:
        """Most recently dispatched chunk_id; 0 before any infer()."""
        return self._current_chunk_id

    # ------------------------------------------------------------------
    # E-STOP push callback (registered with UnitreeG1Provider in start())
    # ------------------------------------------------------------------
    def _on_estop(self, active: bool, ts: float) -> None:
        """E-STOP push callback. Cancel current chunk; gate future infer()."""
        # TODO(REQ-39) [TASK-40]: self._estop_active = active
        # TODO(REQ-39) [TASK-40]: if active: self.cancel_chunk(reason="estop")
        # TODO(REQ-39) [TASK-40]: infer() must early-return when
        #                         self._estop_active is True.
        raise NotImplementedError("VLAProvider._on_estop: TBD [TASK-40]")

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

        When placement moves off-PC (separate Jetson / NX), this method's
        *intent* stays the same (post-VLA balance correction); the
        argument list grows to carry the IMU snapshot over the wire
        (the remote process cannot see ``self._unitree_g1``).
        """
        # TODO(REQ-43) [TASK-40]: when spec lands, run model here:
        #     corrected = self._gearsonic(
        #         chunk, imu_base, imu_ankle_l, imu_ankle_r, joint_state,
        #     )
        # TODO(REQ-43) [TASK-40]: if placement moves off-PC, change call site
        #     to RPC; signature grows by the IMU + joint_state args above.
        return chunk  # identity passthrough placeholder

    # ------------------------------------------------------------------
    # Internals — chunk → step unpack (CONV-006 canonical)
    #
    # Replay-worker policy decisions (LOCKED for the demo — implementer to
    # follow):
    #   - Empty queue (underflow): re-publish the last step at 100 Hz to
    #     hold the pose; do NOT silently stop publishing.
    #   - Chunk N → N+1 crossfade: 2-step linear blend between the last
    #     step of N and the first step of N+1 (≈20 ms blend at 100 Hz).
    #     PC-side per CONV-006; NX queue_aggregate stays OFF.
    #   - Mid-chunk preemption: new infer() arriving while N is mid-replay
    #     drops the rest of N immediately and the crossfade above blends
    #     from the last-published step into N+1's first step (so the
    #     crossfade rule applies whether the previous chunk was fully or
    #     partially consumed).
    #   - arm vs low split: each step's 29-DoF joint set is partitioned
    #     into the IF-6 arm subset (publish_joint_cmd_arm) and the
    #     2026-05-22 lower-body subset (publish_joint_cmd_low) using the
    #     _ARM_JOINTS / _LOW_JOINTS frozensets to be defined per the ICD.
    # ------------------------------------------------------------------
    def _enqueue_chunk_steps(self, chunk: Any, chunk_id: int) -> None:
        """
        Unpack an ``action_horizon``-step chunk (currently 16 per
        CONV-006) into the 100 Hz replay queue.

        Each enqueued step carries ``(chunk_id, step_index)`` so the NX
        ``motor_controller`` ring-buffer can detect chunk boundaries for
        the optional crossfade fallback (default OFF per CONV-006).
        """
        # TODO(REQ-31) [TASK-40]: implement per the policy block above
        #     (underflow / crossfade / preemption / arm-low split).
        raise NotImplementedError("VLAProvider._enqueue_chunk_steps: TBD [TASK-40]")
