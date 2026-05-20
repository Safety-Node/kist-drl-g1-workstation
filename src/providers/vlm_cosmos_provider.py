"""
VLM Provider (NVIDIA COSMOS) -- KIST DRL G1 Workstation
=======================================================

drawio C4 Container:
    Name        : VLM Provider
    Technology  : COSMOS
    Description : Describes scene state as text.

Edges:
    UnitreeG1 Provider -> VLM Provider : Camera image / Buf State [Image / text]
    VLM Provider -> Vision Sensor      : Scene Description [text]

TBD:
    - Decide deployment: local GPU (NIM) vs cloud Cosmos endpoint
    - Frame sampling rate (FPS) + max image resolution
    - Prompt templates for object-list / spatial-relations / pose description
    - Output schema (free text vs JSON with bounding info)
    - Cache + dedup on near-identical frames
    - Safety: visual content filter + hallucination flag in metadata
"""

import logging
from typing import Optional

from .singleton import singleton


@singleton
class VLMCosmosProvider:
    """
    Workstation-side NVIDIA COSMOS VLM wrapper.

    Consumes camera images (and optional buffer state text) from the
    UnitreeG1 Provider and emits a structured scene description to the
    Vision Sensor.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "cosmos-reason1",
        target_fps: float = 2.0,
    ):
        """
        Parameters
        ----------
        endpoint : str, optional
            NIM URL or hosted Cosmos URL. None = use local.
        api_key : str, optional
            API key when using hosted endpoint.
        model_name : str
            Selected Cosmos variant.
        target_fps : float
            Maximum frames-per-second pushed through the model.
        """
        # TODO: HTTP/gRPC client init; warm-up call
        self._endpoint = endpoint
        self._api_key = api_key
        self._model_name = model_name
        self._target_fps = target_fps
        logging.info("VLMCosmosProvider: skeleton initialized (model=%s)", model_name)

    async def describe(self, image_bytes: bytes, buf_state: Optional[str] = None) -> str:
        """
        Run a single VLM forward pass on the supplied frame.

        Parameters
        ----------
        image_bytes : bytes
            Encoded image (JPEG/PNG) or raw RGB array buffer.
        buf_state : str, optional
            Additional buffer/joint state context appended to the prompt.

        Returns
        -------
        str
            Scene description text.
        """
        # TODO: prompt assembly with buf_state context
        # TODO: throttle to target_fps
        # TODO: post-process / safety filter
        raise NotImplementedError("VLMCosmosProvider.describe: TBD")
