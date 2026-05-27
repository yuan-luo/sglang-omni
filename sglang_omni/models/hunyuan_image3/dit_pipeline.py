# SPDX-License-Identifier: Apache-2.0
"""DiT flow-matching pipeline for HunyuanImage-3.

Consumes the AR→DiT projected payload (prompt text, target height/width, AR
per-layer K/V) and produces a finished image. The transformer body is a
placeholder — `_run_transformer_step` raises NotImplementedError until the
full HunyuanImage-3 DiT block port lands.

Lifecycle of one request inside this pipeline:

  1. `prepare_request(payload)` — text-encode the prompt, build the initial
     latent (Gaussian noise of shape `[B, C, H', W']`), install the AR
     prefix into `ImageKVCacheManager` (one entry per layer), and — when
     CFG is on — build the negative branch's K/V via `build_neg_ar_kv`.

  2. `denoise(state)` — `num_inference_steps` flow-matching updates. At each
     step the transformer is called once per CFG branch. Per-layer
     attention uses `kv_manager.inject_into_layer(...)` to prepend the AR
     prefix.

  3. `decode(state)` — VAE-decode the final latent to a PIL Image / raw
     bytes for the OpenAI response.

The pipeline is intentionally framework-agnostic: it depends on a
`DiTBackend` protocol (text encoder + transformer + VAE) injected at
construction. The concrete backend will be wired in M2 alongside the DiT
weight port.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch

from sglang_omni.models.hunyuan_image3.image_kv_cache_manager import ImageKVCacheManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol.
# ---------------------------------------------------------------------------
class DiTBackend(Protocol):
    """Minimum interface the DiT pipeline expects from a model backend.

    Implemented by `dit_transformer.py` in M2 once the HunyuanImage-3 DiT
    transformer port lands. Kept narrow so the pipeline can be exercised
    with a stub backend in unit tests.
    """

    num_layers: int
    in_channels: int
    patch_size: int
    device: torch.device
    dtype: torch.dtype

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Return text embeddings, shape `[B, T, D]`."""

    def step(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor,
        *,
        kv_manager: ImageKVCacheManager,
        branch: str,
    ) -> torch.Tensor:
        """One denoising step. Returns the predicted velocity.

        The backend is responsible for routing per-layer K/V through
        `kv_manager.inject_into_layer(layer_id, k_step, v_step, branch=branch)`
        inside its own attention modules.
        """

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE-decode `latent` → image tensor in `[B, C, H, W]`, range [0, 1]."""


# ---------------------------------------------------------------------------
# Pipeline state.
# ---------------------------------------------------------------------------
@dataclass
class DiTRequestState:
    """Per-request transient state held during a flow-matching run."""

    request_id: str
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: Optional[int]

    # Populated by `prepare_request`.
    latent: Optional[torch.Tensor] = None
    text_emb: Optional[torch.Tensor] = None
    neg_text_emb: Optional[torch.Tensor] = None
    kv_manager: Optional[ImageKVCacheManager] = None

    # Diffusion progress.
    step_index: int = 0
    finished: bool = False

    # Output (set after `decode`).
    image_bytes: Optional[bytes] = None
    image_format: str = "png"

    # Free-form metadata propagated to the response.
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline.
# ---------------------------------------------------------------------------
class HunyuanImage3DiTPipeline:
    """Flow-matching loop driver.

    Args:
      backend: Concrete DiT backend (text encoder + transformer + VAE).
      patch_size: Spatial patch size used to convert (H, W) pixels into a
        latent grid. Read off `backend.patch_size` by default; surfaced as
        an explicit arg for tests using stub backends.
      vae_scale_factor: Ratio between pixel and latent resolution
        (HunyuanImage-3 uses 16 — DC-AE compresses 16× per spatial axis).
    """

    def __init__(
        self,
        backend: DiTBackend,
        *,
        vae_scale_factor: int = 16,
    ):
        self.backend = backend
        self.vae_scale_factor = vae_scale_factor

    # ---------------- 1. Prepare ----------------
    def prepare_request(
        self,
        request_id: str,
        *,
        prompt: str,
        height: int,
        width: int,
        ar_kv_layers: dict[int, tuple[torch.Tensor, torch.Tensor]],
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        ar_kv_meta: Optional[dict[str, Any]] = None,
    ) -> DiTRequestState:
        """Initialize per-request state.

        Args:
          ar_kv_layers: One entry per backbone layer, mapping
            `layer_id → (K, V)` with the AR-side shapes documented in
            `ImageKVCacheManager`. Typically produced by `KVExporter` and
            transported via the SHM relay.
          ar_kv_meta: Optional metadata describing the AR prefix layout
            (used to build the negative-branch K/V when `guidance_scale > 1`).
            Expected keys (all int):
              - `pos_reuse_len`: total AR prefix length.
              - `shared_prefix_len`: leading shared (system-prompt) length.
              - `neg_reuse_len`: target negative prefix length.
            Plus optional `neg_only_kv: dict[layer_id, (K, V)]` for the
            negative-only suffix slice when `neg_reuse_len > shared_prefix_len`.
        """
        state = DiTRequestState(
            request_id=request_id,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # ---- Initial noise latent ----
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        if latent_h <= 0 or latent_w <= 0:
            raise ValueError(
                f"Target resolution {height}x{width} too small for vae_scale_factor="
                f"{self.vae_scale_factor}"
            )
        generator = (
            torch.Generator(device=self.backend.device).manual_seed(seed)
            if seed is not None
            else None
        )
        state.latent = torch.randn(
            (1, self.backend.in_channels, latent_h, latent_w),
            device=self.backend.device,
            dtype=self.backend.dtype,
            generator=generator,
        )

        # ---- Text embeddings ----
        state.text_emb = self.backend.encode_text([prompt])
        if guidance_scale > 1.0:
            state.neg_text_emb = self.backend.encode_text([negative_prompt or ""])

        # ---- AR K/V install ----
        kv_manager = ImageKVCacheManager(
            num_layers=self.backend.num_layers, device=self.backend.device
        )
        kv_manager.set_prompt_kv_bulk(ar_kv_layers)

        if guidance_scale > 1.0 and ar_kv_meta is not None:
            self._build_negative_branch(kv_manager, ar_kv_meta)

        state.kv_manager = kv_manager
        return state

    # ---------------- 2. Denoise ----------------
    @torch.no_grad()
    def denoise(self, state: DiTRequestState) -> DiTRequestState:
        """Run the full flow-matching loop end-to-end."""
        timesteps = self._build_timesteps(state.num_inference_steps)
        for i, t in enumerate(timesteps):
            self._step(state, t)
            state.step_index = i + 1
        state.finished = True
        return state

    @torch.no_grad()
    def step(self, state: DiTRequestState) -> bool:
        """Advance the denoising loop by one step. Returns True when done."""
        timesteps = self._build_timesteps(state.num_inference_steps)
        if state.step_index >= len(timesteps):
            state.finished = True
            return True
        self._step(state, timesteps[state.step_index])
        state.step_index += 1
        if state.step_index >= len(timesteps):
            state.finished = True
        return state.finished

    # ---------------- 3. Decode ----------------
    @torch.no_grad()
    def decode(self, state: DiTRequestState) -> DiTRequestState:
        """VAE-decode the final latent into encoded image bytes."""
        if state.latent is None:
            raise RuntimeError("decode() called before prepare_request()")
        if not state.finished:
            raise RuntimeError("decode() called before denoise() finished")
        image = self.backend.decode_latent(state.latent)
        state.image_bytes = self._encode_image(image, fmt=state.image_format)
        return state

    # ---------------- Internal: one denoising step (with CFG) ----------------
    def _step(self, state: DiTRequestState, timestep: torch.Tensor) -> None:
        if state.latent is None or state.text_emb is None or state.kv_manager is None:
            raise RuntimeError("DiT state not prepared")

        v_pos = self.backend.step(
            latent=state.latent,
            timestep=timestep,
            text_emb=state.text_emb,
            kv_manager=state.kv_manager,
            branch="positive",
        )
        if state.guidance_scale > 1.0 and state.neg_text_emb is not None:
            v_neg = self.backend.step(
                latent=state.latent,
                timestep=timestep,
                text_emb=state.neg_text_emb,
                kv_manager=state.kv_manager,
                branch="negative",
            )
            v = v_neg + state.guidance_scale * (v_pos - v_neg)
        else:
            v = v_pos
        state.latent = self._flow_update(state.latent, v, timestep)

    # ---------------- Internal: numerics ----------------
    def _build_timesteps(self, n_steps: int) -> torch.Tensor:
        """Uniform timestep schedule on `[1, 0]` (flow-matching convention).

        HunyuanImage-3-Distil uses 8 steps with linear t-spacing. The exact
        schedule (sigmoid / shifted) is part of the M2 port; this default
        keeps the pipeline runnable end-to-end with stub backends.
        """
        return torch.linspace(1.0, 0.0, steps=n_steps + 1, device=self.backend.device)[
            :-1
        ]

    def _flow_update(
        self,
        latent: torch.Tensor,
        velocity: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Euler step: `x_{t+dt} = x_t - dt * v(x_t, t)`.

        Placeholder for the production sampler (shifted-flow / log-linear /
        adaptive). Sufficient for shape-correct end-to-end smoke tests.
        """
        dt = 1.0 / float(timestep.numel() or 1)
        return latent - dt * velocity

    def _build_negative_branch(
        self,
        kv_manager: ImageKVCacheManager,
        meta: dict[str, Any],
    ) -> None:
        """Materialize the negative-CFG branch's per-layer K/V."""
        pos_reuse_len = int(meta["pos_reuse_len"])
        shared_prefix_len = int(meta["shared_prefix_len"])
        neg_reuse_len = int(meta["neg_reuse_len"])
        neg_only_kv = meta.get("neg_only_kv") or {}
        for layer_id in range(self.backend.num_layers):
            neg_only_k, neg_only_v = neg_only_kv.get(layer_id, (None, None))
            kv_manager.build_neg_ar_kv(
                layer_id,
                pos_reuse_len=pos_reuse_len,
                neg_reuse_len=neg_reuse_len,
                shared_prefix_len=shared_prefix_len,
                neg_only_k=neg_only_k,
                neg_only_v=neg_only_v,
            )

    # ---------------- Internal: image encoding ----------------
    def _encode_image(self, image: torch.Tensor, *, fmt: str = "png") -> bytes:
        """Convert `[B, C, H, W]` float tensor in [0, 1] to encoded bytes."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                f"Expected single-sample image [1, C, H, W], got {tuple(image.shape)}"
            )
        arr = image[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        arr = arr.permute(1, 2, 0).cpu().numpy()
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        pil = Image.fromarray(np.asarray(arr))
        buf = BytesIO()
        pil.save(buf, format=fmt.upper())
        return buf.getvalue()
