# SPDX-License-Identifier: Apache-2.0
"""Encoder model support - Policy, InputPreparer, OutputProcessor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..types import Request, RequestOutput, SchedulerOutput


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class EncoderRequestData:
    """Encoder-specific request data (stored in Request.data)."""

    input_ids: torch.Tensor
    embeddings: torch.Tensor | None = None  # Filled after execution


@dataclass
class EncoderBatchData:
    """Encoder-specific batch data (SchedulerOutput.batch_data)."""

    input_ids_list: list[torch.Tensor]
    seq_lens: list[int]


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------


class EncoderPolicy:
    """Scheduling policy for encoder models.

    Characteristics:
    - Single forward pass (no iteration)
    - No KV cache
    - Simple resource tracking (just count)
    """

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self._count = 0

    def can_schedule(self, request: Request) -> bool:
        return self._count < self.max_batch_size

    def on_schedule(self, request: Request) -> None:
        self._count += 1

    def on_finish(self, request: Request) -> None:
        self._count = max(0, self._count - 1)

    def build_batch(self, requests: list[Request]) -> EncoderBatchData:
        return EncoderBatchData(
            input_ids_list=[r.data.input_ids for r in requests],
            seq_lens=[len(r.data.input_ids) for r in requests],
        )

    def update_request(self, request: Request, output: RequestOutput) -> None:
        request.data.embeddings = output.data

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        return True  # Encoder always done in one pass


# -----------------------------------------------------------------------------
# InputPreparer
# -----------------------------------------------------------------------------


class EncoderInputPreparer:
    """Converts EncoderBatchData to model inputs."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def prepare(
        self, batch_data: EncoderBatchData, device: torch.device
    ) -> dict[str, Any]:
        max_len = max(batch_data.seq_lens)
        batch_size = len(batch_data.input_ids_list)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        for i, ids in enumerate(batch_data.input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = ids.to(device)
            attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------------------------------------------------------
# OutputProcessor
# -----------------------------------------------------------------------------


class EncoderOutputProcessor:
    """Extracts embeddings from encoder output."""

    def __init__(self, pooling: str = "last"):
        """Initialize output processor.

        Args:
            pooling: Pooling strategy - "last", "mean", or "cls"
        """
        self.pooling = pooling

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        # Handle different output formats
        if hasattr(model_output, "last_hidden_state"):
            hidden_states = model_output.last_hidden_state  # [batch, seq, hidden]
        elif isinstance(model_output, torch.Tensor):
            hidden_states = model_output
        else:
            raise ValueError(f"Unexpected model output type: {type(model_output)}")

        batch_data: EncoderBatchData = scheduler_output.batch_data

        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            seq_len = batch_data.seq_lens[i]

            if self.pooling == "last":
                emb = hidden_states[i, seq_len - 1]
            elif self.pooling == "mean":
                emb = hidden_states[i, :seq_len].mean(dim=0)
            elif self.pooling == "cls":
                emb = hidden_states[i, 0]
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=emb,
                finished=True,
                finish_reason="stop",
            )

        return outputs
