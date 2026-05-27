# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3 DiT backbone — Protocol-conforming backend skeleton.

This module is intentionally a skeleton. The flow-matching pipeline
(`dit_pipeline.py`) drives the model via the `DiTBackend` Protocol; this
file gives the production backend a concrete class to fill in during M2.
All forward-path math is left as `NotImplementedError` so a partial port
fails loudly rather than silently producing garbage.

What needs to land in M2:
  1. Patchify / unpatchify between [B, C, H, W] latents and [B, T, D]
     token sequences.
  2. Time embedding (sinusoidal-style; see `_TimestepEmbedder`).
  3. The 80B transformer stack — same architecture as the AR backbone but
     called with the diffusion-side input modulation. Each block routes
     per-layer K/V through `kv_manager.inject_into_layer(...)`.
  4. Output projection to predict velocity `v`.
  5. VAE decode delegation to `autoencoder.py`.

Text conditioning note: for HunyuanImage-3 there is no separate text
encoder. The text conditioning IS the AR K/V cache, prepended layer-by-
layer at attention time. `encode_text` therefore returns an empty
placeholder tensor; the pipeline carries it through `step()` but the
backend ignores it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from sglang_omni.models.hunyuan_image3.image_kv_cache_manager import (
    ImageKVCacheManager,
)


class HunyuanImage3DiTBackend(nn.Module):
    """Concrete `DiTBackend` for HunyuanImage-3.

    Wraps:
      - patchify / unpatchify utility ops
      - the 80B transformer body (re-using AR weights)
      - DC-AE VAE (decode-only path)
    """

    # Public attributes consumed by the pipeline.
    num_layers: int
    in_channels: int
    patch_size: int
    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        *,
        num_layers: int = 32,
        in_channels: int = 64,
        patch_size: int = 1,
        hidden_size: int = 6144,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda")
        self.dtype = dtype

        # Subcomponents — instantiated empty; weights loaded by `load_weights`
        # via the same remap that AR uses (the 80B body is shared).
        self.t_embedder = _TimestepEmbedder(hidden_size)
        self.x_embedder: Optional[nn.Module] = None  # patchify: [B,C,H,W]→[B,T,D]
        self.blocks: Optional[nn.ModuleList] = None  # 32 transformer blocks
        self.final_layer: Optional[nn.Module] = None  # head: D → patch_size² * C
        self.vae: Optional[nn.Module] = None  # autoencoder (decode-only here)

    # ---------------- DiTBackend protocol ----------------
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """No-op for HunyuanImage-3.

        Text conditioning lives in the AR K/V cache. Returns a zero-size
        placeholder so the pipeline's `step()` signature stays uniform.
        """
        return torch.empty(len(prompts), 0, 0, device=self.device, dtype=self.dtype)

    def step(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_emb: torch.Tensor,
        *,
        kv_manager: ImageKVCacheManager,
        branch: str,
    ) -> torch.Tensor:
        """Predict the flow-matching velocity at one timestep.

        Reference forward graph (to be implemented):
            tokens = self._patchify(latent)               # [B, T, D]
            t_cond = self.t_embedder(timestep)            # [B, D]
            for layer_id, block in enumerate(self.blocks):
                tokens = block(
                    tokens,
                    cond=t_cond,
                    kv_manager=kv_manager,
                    layer_id=layer_id,
                    branch=branch,
                )
            v_tokens = self.final_layer(tokens)           # [B, T, p²·C]
            return self._unpatchify(v_tokens, latent.shape)
        """
        raise NotImplementedError(
            "HunyuanImage3DiTBackend.step is not yet implemented. The DiT "
            "transformer body lands in milestone M2."
        )

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE-decode the final latent into pixels in [0, 1]."""
        if self.vae is None:
            raise NotImplementedError(
                "VAE not loaded. `decode_latent` is implemented but requires "
                "the autoencoder weights to be wired in load_weights."
            )
        return self.vae.decode(latent).clamp(0.0, 1.0)

    # ---------------- Patchify helpers ----------------
    def _patchify(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, (H/p)(W/p), C·p²]. Implemented in M2."""
        raise NotImplementedError

    def _unpatchify(
        self, tokens: torch.Tensor, latent_shape: torch.Size
    ) -> torch.Tensor:
        """[B, T, C·p²] -> [B, C, H, W]. Implemented in M2."""
        raise NotImplementedError

    # ---------------- Weight loading ----------------
    def load_weights(self, weights) -> None:
        """Load DiT weights from the checkpoint.

        Mirrors `ar_model.HunyuanImage3ForConditionalGeneration.load_weights`
        for the shared transformer body, plus the DiT-specific heads:
            - `x_embedder.weight` (patchify conv/linear)
            - `t_embedder.*`      (time embedding MLP)
            - `final_layer.*`     (velocity head)
            - `vae.*`             (DC-AE autoencoder)
        """
        raise NotImplementedError(
            "load_weights pending M2 — depends on the finalized weight-name "
            "remap for DiT-side parameters."
        )


# ---------------------------------------------------------------------------
# Subcomponents
# ---------------------------------------------------------------------------
class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → 2-layer MLP.

    Standard DiT-style time conditioning. Output dim is `hidden_size`. The
    sinusoidal frequency bank is built once at construction; the MLP is
    learned and loaded from the checkpoint.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def _sinusoidal_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Standard DiT sinusoidal time embedding."""
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device)
            * (torch.log(torch.tensor(max_period, dtype=torch.float32)) / half)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self._sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))
