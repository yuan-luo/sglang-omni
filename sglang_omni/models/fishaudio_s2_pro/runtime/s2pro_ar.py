# SPDX-License-Identifier: Apache-2.0
"""Shared S2-Pro runtime components (step output + sampling helpers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class S2ProStepOutput:
    """Per-step output containing multi-codebook tokens."""

    codes: torch.Tensor  # [num_codebooks+1, 1]


# ---------------------------------------------------------------------------
# Sampling helpers
#
# All operations below are CUDA-graph-safe: no Python-level data-dependent
# branching, no scatter (uses argsort+gather instead), and top_k is a tensor
# rather than a Python int so the graph shape stays fixed.
# ---------------------------------------------------------------------------


def _multinomial_no_sync(probs: Tensor) -> Tensor:
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _logits_to_probs(
    logits: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    top_k: Tensor,
) -> Tensor:
    """Convert logits to probabilities with top-p and top-k filtering.

    Uses argsort+gather (not scatter) for CUDA graph capture compatibility.
    ``top_k`` must be a tensor ``(batch, 1)`` so the graph shape is fixed.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > top_p

    indices = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    sorted_indices_to_remove = sorted_indices_to_remove | (indices >= top_k)
    sorted_indices_to_remove[..., 0] = False

    sorted_logits = sorted_logits / torch.clip(temperature, min=1e-5)
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, -float("Inf"))
    probs_sort = F.softmax(sorted_logits, dim=-1)

    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    return torch.gather(probs_sort, dim=-1, index=inverse_indices)


def _sample_with_topk(
    logits: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    top_k: Union[int, Tensor] = 30,
    repetition_penalty: Tensor | None = None,
    previous_tokens: Tensor | None = None,
) -> Tensor:
    if previous_tokens is not None and repetition_penalty is not None:
        prev = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=prev)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits = logits.clone()
        logits.scatter_(dim=-1, index=prev, src=score.to(logits.dtype))

    if isinstance(top_k, int):
        top_k = torch.tensor([[top_k]], device=logits.device, dtype=torch.int64).expand(
            logits.shape[0], 1
        )

    probs = _logits_to_probs(logits, temperature, top_p, top_k)
    return _multinomial_no_sync(probs)
