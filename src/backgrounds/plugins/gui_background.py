"""
GUI Background [TASK-47, REQ-41]

Display **data publisher** — does NOT composite/render. Polls the camera frame
and task status and pushes them over a WebSocket; a separate browser renderer
(kist-drl-g1-gui) draws the overlay.

Per frame tick, every connected client receives two messages:
  - text (JSON)  : status snapshot (schema below)
  - binary       : latest camera JPEG bytes (omitted when no frame yet)

Status JSON:
  { "scenario": str|null,
    "subtask": { "name": str, "i": int, "n": int } | null,
    "state": "idle|active|success|failed" }

Polled (CONV-010/011, no IOProvider):
  UnitreeG1Provider.color.value (JPEG)
  TaskSrvProvider.state / .active_scenario_name / .active_sub_task(.name)
                 / .active_sub_task_index / .active_sub_task_total

Transport = WebSocket (`websockets`; ROS-free renderer). Latency budget ≤ 200 ms.
Backpressure: at most one in-flight broadcast — if the previous hasn't finished,
the new frame is dropped (latest-wins); slow clients are dropped on send timeout.
"""

import asyncio
import json
import logging
from typing import Any, Optional, Set, Tuple

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.task_srv_provider import TaskSrvProvider
from providers.unitree_g1_provider import UnitreeG1Provider


class GUIBackgroundConfig(BackgroundConfig):
    """Configuration for the GUI WebSocket publisher."""

    fps: float = Field(default=15.0, gt=0, description="Snapshot publish rate (Hz).")
    ws_host: str = Field(default="0.0.0.0", description="WebSocket bind host.")
    ws_port: int = Field(default=8081, description="WebSocket bind port.")
    send_timeout_s: float = Field(
        default=1.0,
        gt=0,
        description="Per-message send timeout; a client slower than this is dropped.",
    )
    heartbeat_every_frames: int = Field(
        default=75,                          # 5 s at 15 fps
        ge=0,
        description="INFO heartbeat every N frames (CONV-009 — makes the loop observable). 0 disables.",
    )


class GUIBackground(Background[GUIBackgroundConfig]):
    """Polls camera + task status and broadcasts (frame + status) to WS clients."""

    def __init__(self, config: GUIBackgroundConfig):
        super().__init__(config)
        # CONV-010: providers fetched directly (no IOProvider). run.py builds them first.
        self._unitree_g1 = UnitreeG1Provider()
        self._task_srv = TaskSrvProvider()
        self._clients: Set[Any] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        logging.info(
            "GUIBackground: initialized (fps=%.1f, ws=%s:%d)",
            config.fps, config.ws_host, config.ws_port,
        )

    def run(self) -> None:
        """Own an asyncio loop: serve WebSocket + drift-free snapshot/broadcast until stop."""
        import websockets  # local import keeps the module importable without the dep present

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        period = 1.0 / float(self.config.fps)
        hb = self.config.heartbeat_every_frames
        sched = {"next": loop.time(), "n": 0}

        async def _handler(ws: Any) -> None:
            # Receive-only renderer: keep the handler alive by draining inbound
            # (there usually is none) until the client disconnects.
            self._clients.add(ws)
            logging.info("GUIBackground: client connected (%d total)", len(self._clients))
            try:
                async for _ in ws:
                    pass
            finally:
                self._clients.discard(ws)

        def _pump() -> None:
            if self.should_stop():
                loop.stop()
                return
            try:
                jpeg, status = self._snapshot()
                # Backpressure: only one broadcast in flight — else drop this frame.
                if self._broadcast_task is None or self._broadcast_task.done():
                    self._broadcast_task = loop.create_task(self._broadcast(jpeg, status))
                sched["n"] += 1
                if hb and sched["n"] % hb == 0:
                    logging.info(
                        "GUIBackground: heartbeat frame=%d clients=%d state=%s",
                        sched["n"], len(self._clients), status.get("state"),
                    )
            except Exception:
                logging.exception("GUIBackground: pump tick raised; continuing")
            finally:
                # Reschedule unconditionally so a transient error never freezes the stream.
                sched["next"] += period
                if sched["next"] - loop.time() < 0:
                    logging.warning("GUIBackground: frame overran period; resetting baseline")
                    sched["next"] = loop.time()
                loop.call_at(sched["next"], _pump)

        logging.info("GUIBackground: stream loop entering (period=%.3fs)", period)
        server = None
        try:
            server = loop.run_until_complete(
                websockets.serve(_handler, self.config.ws_host, self.config.ws_port)
            )
            loop.call_soon(_pump)
            loop.run_forever()
        finally:
            if server is not None:
                server.close()
                loop.run_until_complete(server.wait_closed())
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            logging.info("GUIBackground: stream loop exited (cleanup done)")

    # ------------------------------------------------------------------
    def _snapshot(self) -> Tuple[Optional[bytes], dict]:
        """Poll camera + task status into (jpeg_bytes | None, status_dict).

        Direct access (CONV-010 guarantees the providers are constructed before
        this background). Any surprise is contained by the pump's try/finally.
        """
        jpeg = _jpeg_bytes(self._unitree_g1.color.value)
        st = self._task_srv
        sub = st.active_sub_task
        status = {
            "scenario": st.active_scenario_name,
            "subtask": (
                {"name": sub.name, "i": st.active_sub_task_index, "n": st.active_sub_task_total}
                if sub is not None else None
            ),
            "state": st.state.value,
        }
        return jpeg, status

    async def _broadcast(self, jpeg: Optional[bytes], status: dict) -> None:
        """Send status (text) then frame (binary) to each client. One broadcast
        runs at a time (pump guard), so the text/binary pair never interleaves.
        A client slower than send_timeout_s is dropped."""
        if not self._clients:
            return
        text = json.dumps(status, ensure_ascii=False, default=str)
        timeout = self.config.send_timeout_s
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.send(text), timeout)
                if jpeg is not None:
                    await asyncio.wait_for(ws.send(jpeg), timeout)
            except Exception:
                self._clients.discard(ws)
                try:
                    await ws.close()
                except Exception:
                    pass

    def stop(self) -> None:
        """Base sets stop_event; the pump sees should_stop() and stops the loop."""
        return None


# ---------------------------------------------------------------------------


def _jpeg_bytes(frame: Any) -> Optional[bytes]:
    """Extract JPEG bytes from a CompressedImage-like / bytes / None frame value."""
    if frame is None:
        return None
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    data = getattr(frame, "data", None)   # sensor_msgs/CompressedImage.data
    return bytes(data) if data is not None else None
