# SPDX-License-Identifier: Apache-2.0
"""Per-layer AR→DiT K/V cache manager for HunyuanImage-3.

Sits on the DiT side. After AR finishes its forward, the bridge produces a
`dict[layer_id, PerLayerKV]` (see `sglang_omni.model_runner.kv_exporter`).
DiT then runs flow-matching: at step 0 the AR K/V is prepended to DiT's
per-layer K/V via `cache_prompt_kv`; subsequent steps reuse the prefix via
`reuse_prompt_kv`. For CFG (`guidance_scale > 1`), `build_neg_ar_kv` derives
the negative branch's K/V from the shared system-prompt prefix without
re-running AR.

The class is deliberately decoupled from any specific DiT transformer
implementation — it just stores tensors and exposes `inject_into_layer(...)`
that returns the prepended K, V given the current DiT step's K, V.

Tensor conventions (all on the same CUDA device):
    AR K  : [ar_seq_len, num_kv_heads, head_dim]
    AR V  : [ar_seq_len, num_kv_heads, v_head_dim]
    DiT K : [batch, dit_seq_len, num_kv_heads, head_dim]
    DiT V : [batch, dit_seq_len, num_kv_heads, v_head_dim]

`num_kv_heads`, `head_dim`, and `v_head_dim` must match across the two
stages because AR and DiT share the same 80B backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PromptKVSlice:
    """A precomputed slice of the AR prefix, ready to be prepended to DiT K/V.

    Stored once per request after AR finishes; reused unchanged across all
    diffusion steps.
    """

    k: torch.Tensor  # [ar_seq_len, num_kv_heads, head_dim]
    v: torch.Tensor  # [ar_seq_len, num_kv_heads, v_head_dim]
    seq_len: int = 0

    def __post_init__(self) -> None:
        if self.seq_len == 0:
            self.seq_len = int(self.k.shape[0])
        if self.k.shape[0] != self.v.shape[0]:
            raise ValueError(
                f"AR K/V seq_len mismatch: k={self.k.shape}, v={self.v.shape}"
            )


@dataclass
class _LayerEntry:
    """Per-layer state held by the manager."""

    prompt: PromptKVSlice
    # Negative-CFG prefix — only populated when `build_neg_ar_kv` was called
    # for this layer. Same shape as `prompt` but truncated/altered to share
    # the system-prompt prefix while diverging on the negative-only suffix.
    neg_prompt: Optional[PromptKVSlice] = None


class ImageKVCacheManager:
    """Holds AR-exported KV for one DiT request and injects per-layer.

    Lifecycle per request:
        1. AR finishes -> KVExporter -> manager.set_prompt_kv(layer_id, k, v) ∀ layer
        2. (optional CFG) manager.build_neg_ar_kv(layer_id, pos_reuse_len,
           neg_reuse_len, shared_prefix_len) ∀ layer
        3. For each diffusion step s:
             k_full, v_full = manager.inject_into_layer(layer_id, k_step, v_step,
                                                       branch={"positive","negative"})
             # k_full/v_full are passed into the layer's attention
        4. After last step: drop manager (refcount → 0 frees AR-KV memory)
    """

    def __init__(self, num_layers: int, *, device: Optional[torch.device] = None):
        self.num_layers = num_layers
        self.device = device
        self._layers: dict[int, _LayerEntry] = {}

    # ---------------- Population (called by the AR→DiT bridge) ----------------
    def set_prompt_kv(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Install the AR K/V prefix for one layer.

        Called once per layer after AR finishes. Tensors are stored by
        reference (the bridge's caller is responsible for keeping them alive
        for the lifetime of the diffusion run — typically via the request
        object).
        """
        self._layers[layer_id] = _LayerEntry(prompt=PromptKVSlice(k=k, v=v))

    def set_prompt_kv_bulk(
        self, layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Bulk-install all layers' AR K/V at once."""
        for layer_id, (k, v) in layer_kv.items():
            self.set_prompt_kv(layer_id, k, v)

    # ---------------- Step-0 / step-N injection ----------------
    def inject_into_layer(
        self,
        layer_id: int,
        k_step: torch.Tensor,
        v_step: torch.Tensor,
        *,
        branch: str = "positive",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_full, v_full) = concat(AR_prefix, k_step) along the seq axis.

        Args:
          layer_id: Absolute layer index; must have been populated via
            `set_prompt_kv` (or, for `branch="negative"`, via
            `build_neg_ar_kv`).
          k_step: This step's per-layer K. Shape [batch, dit_seq_len,
            num_kv_heads, head_dim]. (Batch axis is allowed but the AR prefix
            is broadcast across the batch — see broadcasting note below.)
          v_step: Same shape as k_step but with v_head_dim.
          branch: "positive" reuses the full AR prefix; "negative" reuses the
            shared system-prompt slice built via `build_neg_ar_kv`.

        Returns:
          Tuple (k_full, v_full) ready to be passed into the layer's
          attention. Shape [batch, ar_seq_len + dit_seq_len, ...].

        Notes:
          - The AR prefix is per-request (not per-batch). When `k_step` has
            an explicit batch axis, the AR prefix is expanded to match.
          - Tensors are concatenated along the second-to-last dimension when
            k_step has a batch axis (rank 4), or the first dimension when
            k_step has no batch (rank 3).
        """
        entry = self._require_layer(layer_id)
        slice_ = entry.neg_prompt if branch == "negative" else entry.prompt
        if slice_ is None:
            raise RuntimeError(
                f"Layer {layer_id}: branch={branch!r} prefix not populated. "
                f"Did you forget to call build_neg_ar_kv?"
            )
        return _prepend(slice_.k, slice_.v, k_step, v_step)

    # ---------------- Negative-CFG (build the negative branch's prefix) ----------------
    def build_neg_ar_kv(
        self,
        layer_id: int,
        *,
        pos_reuse_len: int,
        neg_reuse_len: int,
        shared_prefix_len: int,
        neg_only_k: Optional[torch.Tensor] = None,
        neg_only_v: Optional[torch.Tensor] = None,
    ) -> None:
        """Derive the negative branch's AR K/V from the positive branch.

        The positive branch's AR prefix is [0..pos_reuse_len). The negative
        branch reuses the shared system-prompt prefix [0..shared_prefix_len)
        and diverges on [shared_prefix_len..neg_reuse_len). The caller
        supplies the neg-only tail K/V (the result of a small AR prefill
        over just that slice); we concatenate them with the shared prefix
        to produce the negative branch's installed prefix.

        Args:
          layer_id: Layer to update.
          pos_reuse_len: Length of the positive AR prefix already installed
            via `set_prompt_kv` for this layer.
          neg_reuse_len: Target length of the negative prefix; must be
            ≥ shared_prefix_len.
          shared_prefix_len: Number of leading tokens shared between the
            positive and negative system prompts (typically the system
            prompt's length).
          neg_only_k, neg_only_v: K/V for the slice
            [shared_prefix_len..neg_reuse_len) computed by a small AR
            prefill on the negative-only tokens. Required when
            neg_reuse_len > shared_prefix_len; may be omitted (None) when
            the two branches share the full prefix.
        """
        if not (0 < shared_prefix_len <= pos_reuse_len):
            raise ValueError(
                f"shared_prefix_len={shared_prefix_len} must be in (0, "
                f"pos_reuse_len={pos_reuse_len}]"
            )
        if neg_reuse_len < shared_prefix_len:
            raise ValueError(
                f"neg_reuse_len={neg_reuse_len} must be >= "
                f"shared_prefix_len={shared_prefix_len}"
            )

        pos_slice = self._require_layer(layer_id).prompt
        shared_k = pos_slice.k[:shared_prefix_len]
        shared_v = pos_slice.v[:shared_prefix_len]

        tail_len = neg_reuse_len - shared_prefix_len
        if tail_len == 0:
            neg_k, neg_v = shared_k, shared_v
        else:
            if neg_only_k is None or neg_only_v is None:
                raise ValueError(
                    f"neg_reuse_len={neg_reuse_len} > shared_prefix_len="
                    f"{shared_prefix_len} requires neg_only_k/neg_only_v"
                )
            if neg_only_k.shape[0] != tail_len or neg_only_v.shape[0] != tail_len:
                raise ValueError(
                    f"neg_only_k/neg_only_v tail length must be {tail_len}; "
                    f"got k={neg_only_k.shape[0]}, v={neg_only_v.shape[0]}"
                )
            neg_k = torch.cat([shared_k, neg_only_k], dim=0)
            neg_v = torch.cat([shared_v, neg_only_v], dim=0)

        self._layers[layer_id].neg_prompt = PromptKVSlice(
            k=neg_k, v=neg_v, seq_len=neg_reuse_len
        )

    # ---------------- Introspection ----------------
    def positive_prefix_len(self, layer_id: int) -> int:
        return self._require_layer(layer_id).prompt.seq_len

    def negative_prefix_len(self, layer_id: int) -> Optional[int]:
        entry = self._require_layer(layer_id)
        return entry.neg_prompt.seq_len if entry.neg_prompt is not None else None

    def has_negative_branch(self, layer_id: int) -> bool:
        return self._layers.get(layer_id, _LayerEntry(prompt=None)).neg_prompt is not None  # type: ignore[arg-type]

    def clear(self) -> None:
        """Drop all AR K/V — call after the diffusion run completes."""
        self._layers.clear()

    # ---------------- Internal ----------------
    def _require_layer(self, layer_id: int) -> _LayerEntry:
        entry = self._layers.get(layer_id)
        if entry is None:
            raise KeyError(
                f"Layer {layer_id} has no installed AR prefix. Call "
                f"set_prompt_kv({layer_id}, ...) before diffusion step 0."
            )
        return entry


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _prepend(
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    step_k: torch.Tensor,
    step_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concat prefix and step K/V along the seq axis, broadcasting batch.

    prefix shape: [ar_seq_len, num_kv_heads, head_dim]
    step shape:   [batch, dit_seq_len, num_kv_heads, head_dim]
                  OR
                  [dit_seq_len, num_kv_heads, head_dim]  (no batch axis)
    """
    if step_k.dim() == 4:
        batch = step_k.shape[0]
        prefix_k_b = prefix_k.unsqueeze(0).expand(batch, *prefix_k.shape)
        prefix_v_b = prefix_v.unsqueeze(0).expand(batch, *prefix_v.shape)
        return (
            torch.cat([prefix_k_b, step_k], dim=1),
            torch.cat([prefix_v_b, step_v], dim=1),
        )
    if step_k.dim() == 3:
        return (
            torch.cat([prefix_k, step_k], dim=0),
            torch.cat([prefix_v, step_v], dim=0),
        )
    raise ValueError(
        f"step_k must be rank 3 or 4, got rank {step_k.dim()} (shape {step_k.shape})"
    )
