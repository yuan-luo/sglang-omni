# SPDX-License-Identifier: Apache-2.0
"""Per-layer K/V readback from `token_to_kv_pool`.

Used to bridge the autoregressive (AR) stage of a multi-stage diffusion
pipeline to a downstream diffusion (DiT) stage: after AR finishes generating
its full prefix + control tokens, the bridge gathers each layer's K/V from
the paged KV pool and hands them to DiT as a frozen prefix that DiT's
attention layers can prepend to their per-step K/V.

The implementation mirrors how SGLang's attention backends read the pool —
via the public `get_key_buffer(layer_id)` / `get_value_buffer(layer_id)`
accessors (which synchronize against `layer_transfer_counter`), indexed by
the request's `req_to_token` slot map.

Pool layout (verified against `sglang.srt.mem_cache.memory_pool`):
    k_buffer[layer_id] : Tensor[size + page_size, head_num, head_dim] (store_dtype)
    v_buffer[layer_id] : Tensor[size + page_size, head_num, v_head_dim] (store_dtype)

For our default config (bf16 model, no kv-quantization), `store_dtype == dtype`
and the buffers are directly consumable. If kv-cache quantization is enabled,
the public `get_key_buffer` does the `.view(dtype)` reinterpretation for us.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PerLayerKV:
    """Per-layer key/value tensors for a single request's full prefix.

    Shapes:
      k: [seq_len, num_kv_heads, head_dim]
      v: [seq_len, num_kv_heads, v_head_dim]

    Both tensors live on the same CUDA device as the pool. The caller owns
    these tensors (they are gathered copies, not aliases into the pool).
    """

    k: torch.Tensor
    v: torch.Tensor


class KVExporter:
    """Extracts per-layer K/V from `token_to_kv_pool` for a finished request.

    Usage:
        exporter = KVExporter(tp_worker)
        per_layer = exporter.gather_request(rid, seq_len)
        # per_layer is dict[int, PerLayerKV] keyed by absolute layer index

    The returned tensors are **detached, contiguous copies** — safe to ship
    over inter-stage relays (NCCL, SHM) without worrying about pool reuse.
    """

    def __init__(self, tp_worker: Any):
        """Args:
        tp_worker: SGLang TP worker that owns the model_runner + caches.
        """
        self._tp_worker = tp_worker
        runner = tp_worker.model_runner
        self._token_pool = runner.token_to_kv_pool
        self._req_pool = runner.req_to_token_pool

    @property
    def num_layers(self) -> int:
        """Total layer count exposed by the underlying pool."""
        # SGLang's KVCache stores `layer_num` set at construction time.
        return getattr(self._token_pool, "layer_num", None) or len(
            getattr(self._token_pool, "k_buffer", [])
        )

    @property
    def start_layer(self) -> int:
        """First layer index this rank owns (matches RadixAttention's layer_id)."""
        return getattr(self._token_pool, "start_layer", 0) or 0

    @property
    def end_layer(self) -> int:
        """One past the last layer index this rank owns."""
        end = getattr(self._token_pool, "end_layer", None)
        if end is not None:
            return end
        return self.start_layer + self.num_layers

    def gather_request(
        self,
        rid: int,
        seq_len: int,
        *,
        clone: bool = True,
    ) -> dict[int, PerLayerKV]:
        """Pull all layers' K/V for one request from the pool.

        Args:
          rid: Request id within the req_to_token_pool (the integer slot
            assigned by the scheduler — NOT the user-facing request UUID).
          seq_len: How many leading tokens of the request's K/V to export.
            Must be ≤ the request's actual sequence length. The bridge
            typically passes the full prefix length (input + generated).
          clone: When True (default), the gathered tensors are contiguous
            copies safe to outlive the request's lifetime. Set False only
            for diagnostic reads that happen before the request's pages
            are released.

        Returns:
          dict mapping absolute layer index → PerLayerKV.

        Raises:
          ValueError: if seq_len is non-positive or exceeds the request's
            allocated slot count.
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        # req_to_token[rid] is a [max_seq_len] int64 vector of pool slot
        # indices. The first `seq_len` entries are this request's tokens
        # in order. Later entries are unused/garbage and must be ignored.
        slot_indices = self._req_pool.req_to_token[rid][:seq_len]
        if slot_indices.numel() != seq_len:
            raise ValueError(
                f"req {rid}: requested seq_len={seq_len} but only "
                f"{slot_indices.numel()} slot indices available"
            )

        out: dict[int, PerLayerKV] = {}
        for layer_id in range(self.start_layer, self.end_layer):
            # The public accessors handle:
            #   - `layer_transfer_counter.wait_until(layer_id)` synchronization
            #     so in-flight writes to this layer's buffer are visible
            #   - `.view(dtype)` reinterpretation when store_dtype != dtype
            #     (kv-cache quantization). For bf16 / non-quantized configs
            #     this is a no-op.
            k_buf = self._token_pool.get_key_buffer(layer_id)
            v_buf = self._token_pool.get_value_buffer(layer_id)
            # Fancy-index gather produces a contiguous copy.
            k = k_buf[slot_indices]
            v = v_buf[slot_indices]
            if clone:
                k = k.contiguous().clone()
                v = v.contiguous().clone()
            out[layer_id] = PerLayerKV(k=k, v=v)
        return out

    def gather_request_layer_range(
        self,
        rid: int,
        seq_len: int,
        layer_start: int,
        layer_stop: int,
        *,
        clone: bool = True,
    ) -> dict[int, PerLayerKV]:
        """Gather a subset of layers (e.g. for pipeline-parallel slicing).

        layer_start / layer_stop are absolute layer indices. They must be
        within [self.start_layer, self.end_layer).
        """
        if not (
            self.start_layer <= layer_start < layer_stop <= self.end_layer
        ):
            raise ValueError(
                f"layer range [{layer_start}, {layer_stop}) not within owned "
                f"layers [{self.start_layer}, {self.end_layer})"
            )
        slot_indices = self._req_pool.req_to_token[rid][:seq_len]
        out: dict[int, PerLayerKV] = {}
        for layer_id in range(layer_start, layer_stop):
            k = self._token_pool.get_key_buffer(layer_id)[slot_indices]
            v = self._token_pool.get_value_buffer(layer_id)[slot_indices]
            if clone:
                k = k.contiguous().clone()
                v = v.contiguous().clone()
            out[layer_id] = PerLayerKV(k=k, v=v)
        return out


def pack_per_layer_kv_for_relay(
    per_layer: dict[int, PerLayerKV],
) -> dict[str, torch.Tensor]:
    """Flatten a per-layer KV map into a relay-friendly dict.

    Each layer's K/V become two entries:
        "L{layer_id}.k": [seq_len, num_kv_heads, head_dim]
        "L{layer_id}.v": [seq_len, num_kv_heads, v_head_dim]

    Helper for sending over the SHM/NCCL/NIXL relays which expect a flat
    dict of named tensors.
    """
    out: dict[str, torch.Tensor] = {}
    for layer_id, kv in per_layer.items():
        out[f"L{layer_id}.k"] = kv.k
        out[f"L{layer_id}.v"] = kv.v
    return out


def unpack_per_layer_kv_from_relay(
    flat: dict[str, torch.Tensor],
) -> dict[int, PerLayerKV]:
    """Reverse of `pack_per_layer_kv_for_relay`."""
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in flat.items():
        if not key.startswith("L"):
            continue
        try:
            layer_str, side = key[1:].split(".")
            layer_id = int(layer_str)
        except ValueError:
            continue
        if side not in ("k", "v"):
            continue
        by_layer.setdefault(layer_id, {})[side] = tensor
    return {
        layer_id: PerLayerKV(k=parts["k"], v=parts["v"])
        for layer_id, parts in by_layer.items()
        if "k" in parts and "v" in parts
    }
