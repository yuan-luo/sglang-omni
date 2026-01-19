# SPDX-License-Identifier: Apache-2.0
"""AR (Autoregressive) model support - Simple version with HF KV cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..types import Request, RequestOutput, SchedulerOutput


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class ARRequestData:
    """AR-specific request data (stored in Request.data)."""

    input_ids: torch.Tensor
    output_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    # For simple HF-style KV cache
    past_key_values: tuple | None = None


@dataclass
class ARBatchData:
    """AR-specific batch data (SchedulerOutput.batch_data).

    Simple version: single request only (no batching yet).
    """

    input_ids: torch.Tensor  # [num_tokens]
    is_prefill: bool
    past_key_values: tuple | None = None


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------


class SimpleARPolicy:
    """Simple AR policy using HF-style KV cache.

    Characteristics:
    - Single request at a time (no batching)
    - Uses HF's native past_key_values
    - Iterative generation until EOS or max length

    For initial development. Can upgrade to batched/paged version later.
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
        max_new_tokens: int = 256,
        eos_token_id: int | list[int] = 2,
    ):
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.eos_token_ids = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )
        self._has_request = False

    def can_schedule(self, request: Request) -> bool:
        # Simple: only one request at a time
        return not self._has_request

    def on_schedule(self, request: Request) -> None:
        self._has_request = True

    def on_finish(self, request: Request) -> None:
        self._has_request = False
        # Clear KV cache to free memory
        request.data.past_key_values = None

    def build_batch(self, requests: list[Request]) -> ARBatchData:
        # Simple version: only one request
        assert len(requests) == 1, "SimpleARPolicy only supports single request"
        request = requests[0]
        data: ARRequestData = request.data

        is_prefill = data.num_computed_tokens == 0

        if is_prefill:
            # Prefill: all input tokens
            input_ids = data.input_ids
        else:
            # Decode: last generated token
            last_token = data.output_ids[-1]
            input_ids = torch.tensor([last_token], dtype=torch.long)

        return ARBatchData(
            input_ids=input_ids,
            is_prefill=is_prefill,
            past_key_values=data.past_key_values,
        )

    def update_request(self, request: Request, output: RequestOutput) -> None:
        data: ARRequestData = request.data

        # output.data = (sampled_token, new_past_key_values)
        token, past_kv = output.data

        data.output_ids.append(token)
        data.past_key_values = past_kv

        if data.num_computed_tokens == 0:
            data.num_computed_tokens = len(data.input_ids)
        else:
            data.num_computed_tokens += 1

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        data: ARRequestData = request.data
        token, _ = output.data

        # Check EOS
        if token in self.eos_token_ids:
            return True

        # Check max new tokens
        if len(data.output_ids) >= self.max_new_tokens:
            return True

        # Check total length
        if data.num_computed_tokens >= self.max_seq_len:
            return True

        return False


# -----------------------------------------------------------------------------
# InputPreparer
# -----------------------------------------------------------------------------


class SimpleARInputPreparer:
    """Simple AR input preparer for HF models (single request)."""

    def prepare(
        self, batch_data: ARBatchData, device: torch.device
    ) -> dict[str, Any]:
        input_ids = batch_data.input_ids.unsqueeze(0).to(device)  # [1, seq_len]

        result = {
            "input_ids": input_ids,
            "use_cache": True,
        }

        if batch_data.past_key_values is not None:
            result["past_key_values"] = batch_data.past_key_values

        return result


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class SimpleAROutputProcessor:
    """Simple AR output processor with greedy sampling."""

    def __init__(self, temperature: float = 0.0):
        """Initialize output processor.

        Args:
            temperature: Sampling temperature. 0.0 = greedy.
        """
        self.temperature = temperature

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        # Get logits and past_key_values
        if hasattr(model_output, "logits"):
            logits = model_output.logits  # [batch, seq, vocab]
            past_key_values = model_output.past_key_values
        else:
            raise ValueError(f"Unexpected model output type: {type(model_output)}")

        # Sample from last position
        last_logits = logits[:, -1, :]  # [batch, vocab]

        if self.temperature == 0.0:
            # Greedy
            next_token = last_logits.argmax(dim=-1).item()
        else:
            # Temperature sampling
            probs = torch.softmax(last_logits / self.temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        # Single request
        request = scheduler_output.requests[0]

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=(next_token, past_key_values),
                finished=False,  # Policy decides this
            )
        }
