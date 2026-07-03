import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from src.pipeline.gearsonic.planner.input_builder import PlannerInputBuilder
from src.pipeline.gearsonic.planner.streamer import PlannerStreamer
from src.providers.singleton import singleton

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent / "models/onnx/planner_sonic.onnx"


@singleton
class PlannerOnnxRuntime:

    depends_on = [PlannerStreamer]

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_trajectory: Optional[np.ndarray] = None  # (N, 36) float32

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._init_lock = threading.Lock()
        self._init_joint_pos: Optional[np.ndarray] = None  # (29,) float32 mujoco order

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="planner_onnx_runtime",
            daemon=True,
        )
        self._thread.start()
        logger.info("PlannerOnnxRuntime started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("PlannerOnnxRuntime: thread did not stop within 5s")
        self._thread = None
        logger.info("PlannerOnnxRuntime stopped")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, joint_pos_mujoco: np.ndarray) -> None:
        """Supply robot joint positions (mujoco order, 29 joints) for context init."""
        with self._init_lock:
            self._init_joint_pos = np.asarray(joint_pos_mujoco, dtype=np.float32)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def latest_trajectory(self) -> Optional[np.ndarray]:
        """Latest planner trajectory (N, 36) float32 at 30 Hz. None until first inference."""
        with self._lock:
            return self._latest_trajectory

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        streamer = PlannerStreamer()
        sess    = ort.InferenceSession(str(_MODEL_PATH))
        builder = PlannerInputBuilder()

        initialized = False

        while not self._stop_event.is_set():
            cmd = streamer.command
            if cmd is None:
                time.sleep(0.01)
                continue

            if not initialized:
                with self._init_lock:
                    joint_pos = self._init_joint_pos
                if joint_pos is None:
                    time.sleep(0.01)
                    continue
                builder.initialize(joint_pos)
                initialized = True

            builder.update_command(
                mode=cmd.mode,
                target_vel=cmd.target_vel,
                movement_direction=cmd.movement_direction,
                facing_direction=cmd.facing_direction,
                random_seed=cmd.random_seed,
            )

            inputs = builder.build()
            mujoco_qpos, num_pred_frames = sess.run(None, inputs)

            n    = int(num_pred_frames[0])
            traj = mujoco_qpos[0, :n].copy()  # (N, 36) float32

            builder.set_trajectory(traj)

            with self._lock:
                self._latest_trajectory = traj

            logger.debug("PlannerOnnxRuntime: %d frames", n)
