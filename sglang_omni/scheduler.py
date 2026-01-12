# SPDX-License-Identifier: Apache-2.0
"""Scheduler and Worker abstractions for stages."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class WorkItem:
    """A unit of work in the scheduler queue."""

    request_id: str
    data: Any
    priority: int = 0  # Lower = higher priority (for future use)


class Worker(ABC, Generic[InputT, OutputT]):
    """Abstract base class for stage workers.

    Workers process batches of inputs and produce outputs.
    Subclass this to implement specific processing logic.
    """

    @abstractmethod
    async def execute(
        self, batch: list[tuple[str, InputT]]
    ) -> list[tuple[str, OutputT]]:
        """Execute processing on a batch of inputs.

        Args:
            batch: List of (request_id, input_data) tuples

        Returns:
            List of (request_id, output_data) tuples
        """

    async def setup(self) -> None:
        """Optional setup hook called before processing starts."""

    async def teardown(self) -> None:
        """Optional teardown hook called when worker stops."""


class EchoWorker(Worker[Any, Any]):
    """Simple worker that echoes input with optional transformation.

    Useful for testing the pipeline without real processing.
    """

    def __init__(self, transform: Any = None, delay: float = 0.0):
        """Initialize echo worker.

        Args:
            transform: Optional function to apply to each input
            delay: Optional delay in seconds to simulate processing
        """
        self.transform = transform
        self.delay = delay

    async def execute(self, batch: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        results = []
        for request_id, data in batch:
            if self.transform:
                output = self.transform(data)
            else:
                output = data
            results.append((request_id, output))

        return results


class FIFOScheduler:
    """Simple FIFO scheduler with single worker.

    For Phase 1, we keep it simple:
    - FIFO queue
    - Single worker
    - Process one request at a time (batch_size=1)
    """

    def __init__(self, worker: Worker, batch_size: int = 1):
        """Initialize scheduler.

        Args:
            worker: The worker to execute tasks
            batch_size: Number of items to batch together (default: 1)
        """
        self.worker = worker
        self.batch_size = batch_size
        self._queue: deque[WorkItem] = deque()
        self._pending: dict[str, WorkItem] = {}  # request_id -> item (for abort)
        self._running = False
        self._metrics = {
            "enqueued": 0,
            "processed": 0,
            "aborted": 0,
        }

    def enqueue(self, request_id: str, data: Any) -> None:
        """Add work to the queue.

        Args:
            request_id: Request identifier
            data: Input data for processing
        """
        item = WorkItem(request_id=request_id, data=data)
        self._queue.append(item)
        self._pending[request_id] = item
        self._metrics["enqueued"] += 1
        logger.debug(
            "Scheduler enqueued req=%s, queue_size=%d", request_id, len(self._queue)
        )

    def remove_pending(self, request_id: str) -> bool:
        """Remove a pending request from the queue (for abort).

        Args:
            request_id: Request to remove

        Returns:
            True if found and removed, False otherwise
        """
        if request_id not in self._pending:
            return False

        # Remove from pending dict
        del self._pending[request_id]

        # Remove from queue (O(n) but acceptable for now)
        self._queue = deque(
            item for item in self._queue if item.request_id != request_id
        )

        self._metrics["aborted"] += 1
        logger.debug("Scheduler removed pending req=%s", request_id)
        return True

    async def process_one(self) -> list[tuple[str, Any]] | None:
        """Process up to batch_size items from the queue.

        Returns:
            List of (request_id, output) tuples, or None if queue is empty
        """
        if not self._queue:
            return None

        # Collect batch
        batch_items: list[WorkItem] = []
        while self._queue and len(batch_items) < self.batch_size:
            item = self._queue.popleft()
            # Skip if already removed (aborted)
            if item.request_id in self._pending:
                batch_items.append(item)
                del self._pending[item.request_id]

        if not batch_items:
            return None

        # Execute
        batch_input = [(item.request_id, item.data) for item in batch_items]
        results = await self.worker.execute(batch_input)

        self._metrics["processed"] += len(results)
        logger.debug("Scheduler processed %d items", len(results))

        return results

    def queue_size(self) -> int:
        """Return current queue size."""
        return len(self._queue)

    def has_pending(self, request_id: str) -> bool:
        """Check if a request is pending."""
        return request_id in self._pending

    def health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        return {
            "status": "healthy",
            "queue_size": len(self._queue),
            "pending_count": len(self._pending),
            **self._metrics,
        }
