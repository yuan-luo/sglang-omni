# SPDX-License-Identifier: Apache-2.0
"""Talker-specific SGLang engine components for codec generation.

The talker stage receives thinker hidden states and generates multi-layer
RVQ codec codes in a single forward pass (no incremental decoding at the
backbone level).  The code predictor then autoregressively generates
residual codes for layers 1-15.

Architecture:
  TalkerARRequestData  -- per-request state (thinker outputs, codec codes)
  TalkerBatchPlanner   -- single-request batch planner (no continuous batching)
  TalkerResourceManager -- pass-through (KV cache managed by model worker)
  TalkerOutputProcessor -- extracts codec codes from model output dict
  TalkerIterationController -- single-pass: always finished after one step
  TalkerModelRunner    -- orchestrates full codec generation pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from .interfaces import ResourceManager

if TYPE_CHECKING:
    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker
    from sglang_omni.models.qwen3_omni.talker import Qwen3OmniTalker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CODE_GROUPS = 16  # Total codec code groups (1 backbone + 15 predictor)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class TalkerARRequestData:
    """Per-request state for the talker codec generation engine.

    Stores thinker outputs (embeddings, hidden states, multimodal mask) and
    accumulates generated codec codes across the generation pipeline.

    Fields:
        thinker_embed: Projected token embeddings from the thinker.
        thinker_hidden: Hidden states from the thinker's accept layer.
        is_multimodal_mask: Boolean mask indicating multimodal vs text positions.
        max_new_tokens: Unused for MVP (single-pass), kept for interface compat.
        request_id: Unique identifier echoed back in outputs.
        codec_codes: Generated codes [num_code_groups, seq_len] stored as tensor.
        summed_codec_embeds: Sum of all codec layer embeddings for Code2Wav.
        is_finished: Whether generation is complete.
    """

    # Thinker outputs (inputs to talker)
    thinker_embed: torch.Tensor  # [seq_len, thinker_hidden_size]
    thinker_hidden: torch.Tensor  # [seq_len, thinker_hidden_size]
    is_multimodal_mask: torch.Tensor | None = None  # [seq_len] bool
    max_new_tokens: int = 0
    request_id: str = ""

    # Generated state
    codec_codes: torch.Tensor | None = None  # [num_code_groups, seq_len]
    summed_codec_embeds: torch.Tensor | None = None  # [seq_len, hidden_size]
    is_finished: bool = False


@dataclass
class TalkerBatchData:
    """Batch data for a single talker forward pass.

    For MVP, this wraps exactly one request's data. The model runner
    reads thinker outputs directly from the request data.
    """

    request_data: TalkerARRequestData


# ---------------------------------------------------------------------------
# BatchPlanner
# ---------------------------------------------------------------------------


class TalkerBatchPlanner:
    """Single-request batch planner for talker codec generation.

    No continuous batching for MVP -- processes one request at a time.
    Running requests take priority over waiting requests.
    """

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        if running:
            return [running[0]]

        if not waiting:
            return []

        request = waiting[0]
        if not resource_manager.can_allocate(request):
            return []

        resource_manager.allocate(request)
        return [request]

    def build_batch(self, requests: list[SchedulerRequest]) -> TalkerBatchData:
        """Build batch data from the single selected request."""
        request_data: TalkerARRequestData = requests[0].data
        return TalkerBatchData(request_data=request_data)


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


class TalkerResourceManager:
    """Pass-through resource manager for the talker.

    The talker backbone's KV cache is managed by the SGLang model worker
    internally.  This manager tracks only the count of active requests
    (capped at 1 for MVP single-request processing).
    """

    MAX_CONCURRENT_REQUESTS = 1

    def __init__(self) -> None:
        self._active_count: int = 0

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return self._active_count < self.MAX_CONCURRENT_REQUESTS

    def allocate(self, request: SchedulerRequest) -> None:
        self._active_count += 1

    def free(self, request: SchedulerRequest) -> None:
        self._active_count = max(0, self._active_count - 1)


# ---------------------------------------------------------------------------
# OutputProcessor
# ---------------------------------------------------------------------------


class TalkerOutputProcessor:
    """Converts raw model output dict to per-request RequestOutputs.

    The model runner returns a dict with ``codec_codes`` and
    ``summed_codec_embeds`` tensors.  This processor wraps them into
    the generic ``RequestOutput`` envelope.
    """

    def process(
        self,
        model_output: dict[str, torch.Tensor],
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        outputs: dict[str, RequestOutput] = {}
        for request in scheduler_output.requests:
            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=model_output,
                finished=True,
                finish_reason="stop",
            )
        return outputs


# ---------------------------------------------------------------------------
# IterationController
# ---------------------------------------------------------------------------


class TalkerIterationController:
    """Single-pass iteration controller for the talker.

    The talker processes all positions in one forward pass, so every
    request is finished after a single iteration.
    """

    def update_request(
        self, request: SchedulerRequest, output: RequestOutput
    ) -> None:
        """Store generated codec codes and embeddings on the request data."""
        talker_data: TalkerARRequestData = request.data
        if not isinstance(output.data, dict):
            return

        codec_codes = output.data.get("codec_codes")
        if isinstance(codec_codes, torch.Tensor):
            talker_data.codec_codes = codec_codes

        summed_embeds = output.data.get("summed_codec_embeds")
        if isinstance(summed_embeds, torch.Tensor):
            talker_data.summed_codec_embeds = summed_embeds

        talker_data.is_finished = True

    def is_finished(
        self, request: SchedulerRequest, output: RequestOutput
    ) -> bool:
        return True


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------


class TalkerModelRunner:
    """Model runner that orchestrates the full talker codec generation.

    Pipeline per request:
      1. Project thinker outputs to talker hidden dim (prepare_input_embeds)
      2. Forward through talker MoE backbone (full sequence, single pass)
      3. Compute layer-0 codec logits and sample (codec_head)
      4. Generate layers 1-15 via code predictor (autoregressive per position)
      5. Return all codec codes + summed embeddings

    The talker backbone uses RadixAttention and requires a proper
    ForwardBatch with paged KV cache.  A ``model_worker`` (from Task 5)
    provides this infrastructure.  For MVP, we also support a raw
    ``talker_model`` path that constructs a minimal ForwardBatch.

    Args:
        talker_model: The Qwen3OmniTalker model instance.
        model_worker: Optional SGLang ModelWorker for ForwardBatch infra.
        device: Target device for tensor operations.
    """

    def __init__(
        self,
        talker_model: Qwen3OmniTalker,
        model_worker: ModelWorker | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.talker_model = talker_model
        self.model_worker = model_worker
        self.device = device or next(talker_model.parameters()).device

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute the full talker codec generation pipeline.

        Returns a ModelRunnerOutput containing per-request codec codes
        and summed codec embeddings.
        """
        if not scheduler_output.requests:
            return _empty_model_runner_output()

        batch_data: TalkerBatchData = scheduler_output.batch_data
        request_data = batch_data.request_data
        request_id = scheduler_output.requests[0].request_id

        codec_output = self._generate_codec_codes(request_data)

        outputs = {
            request_id: RequestOutput(
                request_id=request_id,
                data=codec_output,
                finished=True,
                finish_reason="stop",
            ),
        }
        req_ids = [request_id]
        req_id_to_index = {request_id: 0}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    @torch.inference_mode()
    def _generate_codec_codes(
        self, request_data: TalkerARRequestData
    ) -> dict[str, torch.Tensor]:
        """Run the full codec generation pipeline for a single request.

        Steps:
          1. prepare_input_embeds: project thinker -> talker hidden dim
          2. backbone forward: full sequence through talker MoE layers
          3. codec_head: compute layer-0 logits, argmax to get codes
          4. code_predictor_forward: generate layers 1-15

        Returns:
            Dict with ``codec_codes`` [num_code_groups, seq_len] and
            ``summed_codec_embeds`` [seq_len, talker_hidden_size].
        """
        input_embeds = self._prepare_input_embeds(request_data)
        talker_hidden = self._run_backbone_forward(input_embeds)
        layer0_codes = self._sample_layer0_codes(talker_hidden)
        codec_codes, summed_embeds = self._run_code_predictor(
            layer0_codes, talker_hidden
        )

        return {
            "codec_codes": codec_codes.squeeze(0),
            "summed_codec_embeds": summed_embeds.squeeze(0),
        }

    def _prepare_input_embeds(
        self, request_data: TalkerARRequestData
    ) -> torch.Tensor:
        """Project thinker outputs to talker hidden dimension.

        Adds batch dimension [1, seq_len, hidden] for single-request
        processing.

        Returns:
            input_embeds: [1, seq_len, talker_hidden_size]
        """
        thinker_embed = request_data.thinker_embed.to(self.device)
        thinker_hidden = request_data.thinker_hidden.to(self.device)

        is_multimodal_mask = None
        if request_data.is_multimodal_mask is not None:
            is_multimodal_mask = request_data.is_multimodal_mask.to(self.device)

        input_embeds = self.talker_model.prepare_input_embeds(
            thinker_embeds=thinker_embed,
            thinker_hidden_states=thinker_hidden,
            is_multimodal_mask=is_multimodal_mask,
        )

        # Add batch dimension if needed: [seq_len, hidden] -> [1, seq_len, hidden]
        if input_embeds.dim() == 2:
            input_embeds = input_embeds.unsqueeze(0)

        return input_embeds

    def _run_backbone_forward(
        self, input_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Run the talker MoE backbone on the full input sequence.

        Uses the model worker's ForwardBatch infrastructure when available.
        Falls back to constructing a minimal ForwardBatch for standalone use.

        Args:
            input_embeds: [1, seq_len, talker_hidden_size]

        Returns:
            talker_hidden: [1, seq_len, talker_hidden_size]
        """
        seq_len = input_embeds.shape[1]

        if self.model_worker is not None:
            return self._backbone_via_model_worker(input_embeds, seq_len)

        return self._backbone_via_direct_forward(input_embeds, seq_len)

    def _backbone_via_model_worker(
        self,
        input_embeds: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Execute backbone forward using SGLang model worker infrastructure.

        Constructs a ScheduleBatch -> ForwardBatch to satisfy RadixAttention's
        paged KV cache requirements.  KV cache is released after the forward
        pass to avoid leaks.

        Args:
            input_embeds: [1, seq_len, talker_hidden_size]
            seq_len: Sequence length.

        Returns:
            talker_hidden: [1, seq_len, talker_hidden_size]
        """
        from sglang.srt.mem_cache.common import release_kv_cache
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        from sglang_omni.vendor.sglang.core import Req, ScheduleBatch

        req = Req(
            rid=f"talker_{id(input_embeds)}",
            origin_input_text="",
            origin_input_ids=[0] * seq_len,
        )
        req.fill_ids = list(range(seq_len))
        req.extend_input_len = seq_len

        schedule_batch = ScheduleBatch(reqs=[req], batch_is_full=False)
        schedule_batch.prepare_for_extend()

        model_worker_batch = schedule_batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )

        flat_embeds = input_embeds.squeeze(0)
        positions = torch.arange(seq_len, device=self.device)

        try:
            hidden_states = self.talker_model.forward(
                input_ids=torch.zeros(
                    seq_len, dtype=torch.long, device=self.device
                ),
                positions=positions,
                forward_batch=forward_batch,
                inputs_embeds=flat_embeds,
            )
        finally:
            release_kv_cache(req, tree_cache=None)

        return hidden_states.unsqueeze(0)

    def _backbone_via_direct_forward(
        self,
        input_embeds: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Execute backbone forward without model worker (testing path).

        Constructs a minimal mock ForwardBatch. This path is only usable
        when the backbone layers do not strictly require paged KV cache
        (e.g., in unit tests with patched attention).

        Args:
            input_embeds: [1, seq_len, talker_hidden_size]
            seq_len: Sequence length.

        Returns:
            talker_hidden: [1, seq_len, talker_hidden_size]
        """
        positions = torch.arange(seq_len, device=self.device)
        flat_embeds = input_embeds.squeeze(0)  # [seq_len, hidden]

        forward_batch = _build_mock_forward_batch(seq_len, self.device)

        hidden_states = self.talker_model.forward(
            input_ids=torch.zeros(seq_len, dtype=torch.long, device=self.device),
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=flat_embeds,
        )

        return hidden_states.unsqueeze(0)

    def _sample_layer0_codes(
        self, talker_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Compute layer-0 codec logits and sample codes via argmax.

        Args:
            talker_hidden: [1, seq_len, talker_hidden_size]

        Returns:
            layer0_codes: [1, seq_len] int64
        """
        # codec_head expects [tokens, hidden], flatten batch dim
        flat_hidden = talker_hidden.squeeze(0)  # [seq_len, hidden]
        logits = self.talker_model.compute_logits(flat_hidden)  # [seq_len, vocab]
        layer0_codes = logits.argmax(dim=-1)  # [seq_len]
        return layer0_codes.unsqueeze(0)  # [1, seq_len]

    def _run_code_predictor(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate all codec layers via the code predictor.

        Args:
            layer0_codes: [1, seq_len] layer-0 codes from argmax.
            talker_hidden: [1, seq_len, talker_hidden_size]

        Returns:
            codec_codes: [1, num_code_groups, seq_len] all layer codes.
            summed_embeds: [1, seq_len, talker_hidden_size] summed embeddings.
        """
        return self.talker_model.code_predictor_forward(
            layer0_codes=layer0_codes,
            talker_hidden=talker_hidden,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_model_runner_output() -> ModelRunnerOutput:
    """Create an empty ModelRunnerOutput for no-op steps."""
    return ModelRunnerOutput(
        outputs={},
        req_ids=[],
        req_id_to_index={},
    )


class _MockForwardBatch:
    """Minimal ForwardBatch substitute for testing.

    Only suitable for unit tests where attention is patched or the model
    uses standard SDPA fallback.
    """

    def __init__(self, seq_len: int, device: torch.device) -> None:
        self.seq_lens = torch.tensor([seq_len], device=device)
        self.req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        self.seq_lens_sum = seq_len
        self.out_cache_loc = torch.arange(seq_len, device=device)
        self.forward_mode = None
        self.total_num_tokens = seq_len


def _build_mock_forward_batch(seq_len: int, device: torch.device) -> _MockForwardBatch:
    """Build a minimal mock ForwardBatch for testing without model worker."""
    return _MockForwardBatch(seq_len, device)
