# SPDX-License-Identifier: Apache-2.0
"""Bootstrap for the HunyuanImage-3 DiT executor.

Builds a `SimpleScheduler` that drives `HunyuanImage3DiTPipeline` on
incoming AR→DiT projected payloads. Pure setup code — denoising loop math
lives in `dit_pipeline.py`; the transformer body lives in
`dit_transformer.py` (to be added in M2).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def create_hunyuan_image3_dit_scheduler(
    *,
    model_path: str,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    vae_scale_factor: int = 16,
    backend: Optional[Any] = None,
):
    """Construct a SimpleScheduler running the HunyuanImage-3 DiT pipeline.

    Args:
      model_path: Same checkpoint dir as the AR stage (the DiT backbone
        shares the 80B weights with AR).
      gpu_id / tp_rank / tp_size: GPU placement.
      vae_scale_factor: Pixel-to-latent compression ratio. HunyuanImage-3
        uses 16 (DC-AE).
      backend: Optional explicit `DiTBackend` instance. When None, this
        function lazily constructs the production backend (added in M2).
        Tests can pass a stub backend directly.

    Returns:
      A `SimpleScheduler` wrapping the pipeline's async dispatch handler.
    """
    from sglang_omni.models.hunyuan_image3.dit_pipeline import HunyuanImage3DiTPipeline
    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    if backend is None:
        backend = _build_default_backend(
            model_path=model_path,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    pipeline = HunyuanImage3DiTPipeline(
        backend=backend,
        vae_scale_factor=vae_scale_factor,
    )

    async def _dispatch(payload: StagePayload) -> StagePayload:
        data = dict(payload.data) if payload.data else {}
        req = payload.request
        params = (req.params or {}) if req is not None else {}

        prompt = str(data.get("prompt") or "")
        height = int(data.get("height") or 1024)
        width = int(data.get("width") or 1024)
        ar_kv_layers = data.get("ar_kv_layers") or {}
        ar_kv_meta = data.get("ar_kv_meta")
        num_inference_steps = int(params.get("num_inference_steps", 8))
        guidance_scale = float(params.get("guidance_scale", 1.0))
        negative_prompt = params.get("negative_prompt")
        seed = params.get("seed")

        state = pipeline.prepare_request(
            request_id=payload.request_id,
            prompt=prompt,
            height=height,
            width=width,
            ar_kv_layers=ar_kv_layers,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=int(seed) if seed is not None else None,
            ar_kv_meta=ar_kv_meta,
        )
        pipeline.denoise(state)
        pipeline.decode(state)

        out = dict(data)
        out["image_data"] = [state.image_bytes] if state.image_bytes else []
        out["image_format"] = state.image_format
        out["image_size"] = f"{width}x{height}"
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=out,
        )

    return SimpleScheduler(_dispatch)


def _build_default_backend(
    *,
    model_path: str,
    gpu_id: int,
    tp_rank: int,
    tp_size: int,
):
    """Construct the production DiT backend.

    Implemented in M2 alongside `dit_transformer.py`. Until then, raises
    a clear error so misconfigured deployments fail loudly rather than
    silently producing garbage.
    """
    raise NotImplementedError(
        "HunyuanImage-3 DiT backend is not yet implemented. The DiT transformer "
        "port lands in milestone M2. For correctness testing, inject a stub "
        "backend via create_hunyuan_image3_dit_scheduler(backend=...)."
    )
