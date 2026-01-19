# SPDX-License-Identifier: Apache-2.0
"""Generic Scheduler - model-agnostic request lifecycle management."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from .types import (
    ModelRunnerOutput,
    Request,
    RequestStatus,
    SchedulerOutput,
)

if TYPE_CHECKING:
    from .policy.base import SchedulingPolicy


class Scheduler:
    """Generic request scheduler.

    Responsibilities:
    - Manage request lifecycle (WAITING -> RUNNING -> FINISHED)
    - Delegate scheduling decisions to Policy
    - Produce SchedulerOutput for ModelRunner

    Does NOT know about:
    - Input formats (tokens, latents, etc.)
    - Model-specific batching logic
    - Resource details (KV cache, etc.)
    """

    def __init__(self, policy: SchedulingPolicy, max_running: int = 256):
        self.policy = policy
        self.max_running = max_running

        # Request state
        self.requests: dict[str, Request] = {}
        self.waiting: deque[str] = deque()
        self.running: list[str] = []

        # Result futures (created lazily in get_result)
        self._futures: dict[str, asyncio.Future[Request]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add_request(self, request_id: str, data: Any) -> None:
        """Add a new request with model-specific data."""
        request = Request(
            request_id=request_id,
            data=data,
            arrival_time=time.time(),
        )
        self.requests[request_id] = request
        self.waiting.append(request_id)
        # Note: Future created lazily in get_result() to avoid event loop issues

    def abort_request(self, request_id: str) -> None:
        """Abort a request."""
        if request_id not in self.requests:
            return

        request = self.requests[request_id]
        request.status = RequestStatus.ABORTED
        self._finish_request(request)

    def has_requests(self) -> bool:
        """Check if there are any requests to process."""
        return len(self.waiting) > 0 or len(self.running) > 0

    async def get_result(self, request_id: str) -> Request:
        """Wait for a request to complete."""
        if request_id not in self.requests:
            raise KeyError(f"Unknown request: {request_id}")

        # Create future lazily (requires running event loop)
        if request_id not in self._futures:
            self._futures[request_id] = asyncio.get_running_loop().create_future()

        # If already finished, resolve immediately
        request = self.requests[request_id]
        if request.status in (RequestStatus.FINISHED, RequestStatus.ABORTED):
            return request

        return await self._futures[request_id]

    # -------------------------------------------------------------------------
    # Core Scheduling
    # -------------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput | None:
        """Schedule next batch. Returns None if no work."""
        if not self.waiting and not self.running:
            return None

        to_schedule: list[Request] = []

        # 1. Continue running requests
        for req_id in self.running:
            to_schedule.append(self.requests[req_id])

        # 2. Add waiting requests (if resources available)
        to_move: list[str] = []
        for req_id in self.waiting:
            if len(to_schedule) >= self.max_running:
                break

            request = self.requests[req_id]
            if self.policy.can_schedule(request):
                self.policy.on_schedule(request)
                request.status = RequestStatus.RUNNING
                to_schedule.append(request)
                to_move.append(req_id)

        # Move from waiting to running
        for req_id in to_move:
            self.waiting.remove(req_id)
            self.running.append(req_id)

        if not to_schedule:
            return None

        # Build batch using policy (model-specific)
        batch_data = self.policy.build_batch(to_schedule)

        return SchedulerOutput(requests=to_schedule, batch_data=batch_data)

    def update(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> list[Request]:
        """Update state from model output.

        Returns list of finished requests.
        """
        finished: list[Request] = []

        for request in scheduler_output.requests:
            output = model_output.outputs.get(request.request_id)
            if output is None:
                continue

            # Update via policy (model-specific)
            self.policy.update_request(request, output)

            # Check completion via policy
            if self.policy.is_finished(request, output):
                self._finish_request(request)
                finished.append(request)

        return finished

    def _finish_request(self, request: Request) -> None:
        """Clean up finished request."""
        if request.status not in (RequestStatus.FINISHED, RequestStatus.ABORTED):
            request.status = RequestStatus.FINISHED
        request.finish_time = time.time()

        # Free resources via policy
        self.policy.on_finish(request)

        # Remove from running
        if request.request_id in self.running:
            self.running.remove(request.request_id)

        # Resolve future if someone is waiting
        if request.request_id in self._futures:
            future = self._futures[request.request_id]
            if not future.done():
                future.set_result(request)
