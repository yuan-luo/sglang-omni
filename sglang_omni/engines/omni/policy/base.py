# SPDX-License-Identifier: Apache-2.0
"""Base protocols for model-type-specific components."""

from typing import Any, Protocol

import torch

from ..types import Request, RequestOutput, SchedulerOutput


class SchedulingPolicy(Protocol):
    """Model-specific scheduling logic.

    The Policy is responsible for:
    1. Resource management (can we schedule this request?)
    2. Batch building (how to batch requests together?)
    3. Completion detection (is this request done?)

    It is NOT responsible for:
    - Request lifecycle (Scheduler does this)
    - Input/output transformation (InputPreparer/OutputProcessor do this)
    """

    # -------------------------------------------------------------------------
    # Resource Management
    # -------------------------------------------------------------------------

    def can_schedule(self, request: Request) -> bool:
        """Can this request be scheduled? (resources available?)"""
        ...

    def on_schedule(self, request: Request) -> None:
        """Called when request moves WAITING -> RUNNING. Allocate resources."""
        ...

    def on_finish(self, request: Request) -> None:
        """Called when request finishes. Free resources."""
        ...

    # -------------------------------------------------------------------------
    # Batch Building
    # -------------------------------------------------------------------------

    def build_batch(self, requests: list[Request]) -> Any:
        """Build model-specific batch data from requests.

        Returns opaque batch_data that will be passed to InputPreparer.
        """
        ...

    # -------------------------------------------------------------------------
    # State Update
    # -------------------------------------------------------------------------

    def update_request(self, request: Request, output: RequestOutput) -> None:
        """Update request state after model execution."""
        ...

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        """Check if request is finished."""
        ...


class InputPreparer(Protocol):
    """Converts SchedulerOutput.batch_data to model inputs."""

    def prepare(self, batch_data: Any, device: torch.device) -> dict[str, Any]:
        """Convert opaque batch_data to model input dict.

        For Encoder: padded input_ids + attention_mask
        For AR: input_ids + positions + past_key_values + ...
        For DiT: latents + timesteps + ...
        """
        ...


class OutputProcessor(Protocol):
    """Converts model outputs to RequestOutputs."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Convert model output to per-request outputs.

        For Encoder: extract embeddings per request
        For AR: sample tokens per request
        For DiT: extract denoised latents per request
        """
        ...
