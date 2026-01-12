# SPDX-License-Identifier: Apache-2.0
"""Stage abstraction for pipeline processing."""

import asyncio
import logging
from typing import Any, Callable

from sglang_omni.control_plane import StageControlPlane
from sglang_omni.data_plane import SHMDataPlane
from sglang_omni.scheduler import FIFOScheduler, Worker
from sglang_omni.types import (
    CompleteMessage,
    DataReadyMessage,
    ShutdownMessage,
    StageInfo,
    SubmitMessage,
)

logger = logging.getLogger(__name__)


# Type alias for get_next function
# Returns: (next_stage_name, next_stage_endpoint) or None for END
GetNextFn = Callable[[str, Any], tuple[str, str] | None]


class Stage:
    """A processing stage in the pipeline.

    Each stage:
    - Receives work via control plane (ZMQ)
    - Reads input data from data plane (SHM)
    - Processes via scheduler/worker
    - Writes output to data plane (SHM)
    - Routes to next stage via get_next()
    - Handles abort signals
    """

    def __init__(
        self,
        name: str,
        worker: Worker,
        get_next: GetNextFn,
        recv_endpoint: str,
        coordinator_endpoint: str,
        abort_endpoint: str,
        batch_size: int = 1,
    ):
        """Initialize a stage.

        Args:
            name: Stage name (unique identifier)
            worker: Worker instance for processing
            get_next: Function to determine next stage
                      (request_id, output) -> (stage_name, endpoint) or None
            recv_endpoint: ZMQ endpoint to receive work
            coordinator_endpoint: ZMQ endpoint to send completions
            abort_endpoint: ZMQ endpoint for abort broadcasts
            batch_size: Scheduler batch size
        """
        self.name = name
        self.get_next = get_next

        # Components
        self.scheduler = FIFOScheduler(worker, batch_size=batch_size)
        self.data_plane = SHMDataPlane()
        self.control_plane = StageControlPlane(
            stage_name=name,
            recv_endpoint=recv_endpoint,
            coordinator_endpoint=coordinator_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # State
        self._running = False
        self._aborted_requests: set[str] = set()

    async def start(self) -> None:
        """Start the stage."""
        await self.control_plane.start()
        await self.scheduler.worker.setup()
        self._running = True
        logger.info("Stage %s started", self.name)

    async def stop(self) -> None:
        """Stop the stage."""
        self._running = False
        await self.scheduler.worker.teardown()
        self.control_plane.close()
        logger.info("Stage %s stopped", self.name)

    async def run(self) -> None:
        """Main loop: receive work, process, route output."""
        await self.start()

        # Start abort listener as background task
        abort_task = asyncio.create_task(self._abort_listener())

        try:
            while self._running:
                # Receive work (blocking)
                msg = await self.control_plane.recv()

                # Check for shutdown
                if isinstance(msg, ShutdownMessage):
                    logger.info("Stage %s received shutdown", self.name)
                    break

                await self._handle_message(msg)

        except asyncio.CancelledError:
            logger.info("Stage %s cancelled", self.name)
        except Exception as e:
            logger.error("Stage %s error: %s", self.name, e)
            raise
        finally:
            # Cancel abort listener
            abort_task.cancel()
            try:
                await abort_task
            except asyncio.CancelledError:
                pass
            await self.stop()

    async def _abort_listener(self) -> None:
        """Background task to listen for abort broadcasts."""
        try:
            while self._running:
                abort_msg = await self.control_plane.recv_abort()
                self.on_abort(abort_msg.request_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Stage %s abort listener error: %s", self.name, e)

    async def _handle_message(self, msg: DataReadyMessage | SubmitMessage) -> None:
        """Handle an incoming message."""
        if isinstance(msg, SubmitMessage):
            # Initial submission from coordinator
            await self._process_submit(msg)
        elif isinstance(msg, DataReadyMessage):
            # Data from previous stage
            await self._process_data_ready(msg)
        else:
            logger.warning(
                "Stage %s received unexpected message: %s", self.name, type(msg)
            )

    async def _process_submit(self, msg: SubmitMessage) -> None:
        """Process initial submission."""
        request_id = msg.request_id
        logger.debug("Stage %s received submit: req=%s", self.name, request_id)

        # Check if aborted
        if request_id in self._aborted_requests:
            logger.debug("Stage %s skipping aborted req=%s", self.name, request_id)
            return

        # Enqueue and process
        self.scheduler.enqueue(request_id, msg.data)
        await self._process_queue()

    async def _process_data_ready(self, msg: DataReadyMessage) -> None:
        """Process data ready notification."""
        request_id = msg.request_id
        logger.debug(
            "Stage %s received data_ready: req=%s from %s",
            self.name,
            request_id,
            msg.from_stage,
        )

        # Check if aborted
        if request_id in self._aborted_requests:
            logger.debug("Stage %s skipping aborted req=%s", self.name, request_id)
            self.data_plane.cleanup(request_id)
            return

        # Read data from SHM
        result = self.data_plane.get(
            request_id=request_id,
            metadata=msg.shm_metadata,
            from_stage=msg.from_stage,
            to_stage=self.name,
        )
        if result is None:
            logger.error(
                "Stage %s failed to get data for req=%s", self.name, request_id
            )
            await self._send_failure(request_id, "Failed to read from SHM")
            return

        data, _ = result

        # Enqueue and process
        self.scheduler.enqueue(request_id, data)
        await self._process_queue()

    async def _process_queue(self) -> None:
        """Process items from the queue."""
        while self.scheduler.queue_size() > 0:
            results = await self.scheduler.process_one()
            if results is None:
                break

            for request_id, output in results:
                await self._route_output(request_id, output)

    async def _route_output(self, request_id: str, output: Any) -> None:
        """Route output to next stage or complete."""
        # Determine next stage
        next_info = self.get_next(request_id, output)

        if next_info is None:
            # END - send completion to coordinator
            logger.debug("Stage %s: req=%s completed (END)", self.name, request_id)
            await self.control_plane.send_complete(
                CompleteMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    success=True,
                    result=output,
                )
            )
        else:
            # Route to next stage
            next_stage, next_endpoint = next_info
            logger.debug(
                "Stage %s: routing req=%s to %s", self.name, request_id, next_stage
            )

            # Write output to SHM
            success, metadata = self.data_plane.put(
                request_id=request_id,
                data=output,
                from_stage=self.name,
                to_stage=next_stage,
            )
            if not success or metadata is None:
                await self._send_failure(request_id, "Failed to write to SHM")
                return

            # Send data ready notification
            await self.control_plane.send_to_stage(
                next_stage,
                next_endpoint,
                DataReadyMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    to_stage=next_stage,
                    shm_metadata=metadata,
                ),
            )

    async def _send_failure(self, request_id: str, error: str) -> None:
        """Send failure notification to coordinator."""
        await self.control_plane.send_complete(
            CompleteMessage(
                request_id=request_id,
                from_stage=self.name,
                success=False,
                error=error,
            )
        )

    def on_abort(self, request_id: str) -> None:
        """Handle abort for a request."""
        logger.debug("Stage %s: aborting req=%s", self.name, request_id)
        self._aborted_requests.add(request_id)
        self.scheduler.remove_pending(request_id)
        self.data_plane.cleanup(request_id)

    def info(self) -> StageInfo:
        """Return stage info."""
        return StageInfo(
            name=self.name,
            control_endpoint=self.control_plane.recv_endpoint,
        )

    def health(self) -> dict[str, Any]:
        """Return health status."""
        return {
            "name": self.name,
            "running": self._running,
            "scheduler": self.scheduler.health(),
            "data_plane": self.data_plane.health(),
        }


def run_stage_process(
    name: str,
    worker: Worker,
    get_next: GetNextFn,
    recv_endpoint: str,
    coordinator_endpoint: str,
    abort_endpoint: str,
    batch_size: int = 1,
) -> None:
    """Run a stage in its own process.

    This is the entry point for multiprocessing.Process.
    """
    stage = Stage(
        name=name,
        worker=worker,
        get_next=get_next,
        recv_endpoint=recv_endpoint,
        coordinator_endpoint=coordinator_endpoint,
        abort_endpoint=abort_endpoint,
        batch_size=batch_size,
    )

    asyncio.run(stage.run())
