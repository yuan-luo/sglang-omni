# Omni Engine Design

A unified, composable engine architecture for multi-modal model serving (Encoder, AR, Decoder, DiT).

## Design Principles

### Core Principle: Clear Separation > Leaky Abstractions

1. **Generic components know nothing about model specifics**
   - Scheduler doesn't know about tokens, positions, KV cache
   - ModelRunner doesn't know what `batch_data` contains

2. **Model-specific logic lives in dedicated components**
   - Policy: scheduling decisions, resource management
   - InputPreparer: batch_data → model inputs
   - OutputProcessor: model outputs → RequestOutput

3. **Opaque data passing**
   - `Request.data`: model-specific, opaque to Scheduler
   - `SchedulerOutput.batch_data`: built by Policy, consumed by InputPreparer
   - `RequestOutput.data`: model-specific output

### Learned from vLLM v2
1. **Separation of Concerns**: Scheduler, Executor, ModelRunner as distinct components
2. **Contract-based Communication**: SchedulerOutput as the contract between scheduler and runner
3. **Async-first**: Non-blocking execution with futures

### Learned from MiniSGL
1. **Scheduler Owns State**: Single source of truth, runner is stateless
2. **Simple Batch Lifecycle**: Rebuild per step (optimize later if needed)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OmniEngine                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Scheduler (Generic)                            │   │
│  │                                                                      │   │
│  │  - requests: dict[str, Request]                                     │   │
│  │  - waiting / running queues                                         │   │
│  │  - delegates to Policy for model-specific logic                     │   │
│  │                                                                      │   │
│  │  schedule() → SchedulerOutput                                       │   │
│  │  update(SchedulerOutput, ModelRunnerOutput)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                           │                                 │
│                                           │ SchedulerOutput                 │
│                                           │   - requests: list[Request]     │
│                                           │   - batch_data: Any (opaque)    │
│                                           ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ModelRunner (Generic)                          │   │
│  │                                                                      │   │
│  │  execute(SchedulerOutput) → ModelRunnerOutput                       │   │
│  │      ├── InputPreparer.prepare(batch_data)   ← model-specific       │   │
│  │      ├── model.forward()                                            │   │
│  │      └── OutputProcessor.process()           ← model-specific       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Model-Specific Components:
├── Policy (can_schedule, build_batch, is_finished, ...)
├── InputPreparer (batch_data → model inputs)
├── OutputProcessor (model outputs → RequestOutput)
└── RequestData (EncoderRequestData, ARRequestData, DiTRequestData)
```

---

## Component Responsibilities

| Component | Knows About | Doesn't Know About |
|-----------|-------------|-------------------|
| Scheduler | Request lifecycle, queues | Tokens, tensors, KV cache |
| Policy | Model-specific batching, resources | Request lifecycle management |
| ModelRunner | How to call model | What batch_data contains |
| InputPreparer | How to convert batch_data → tensors | Scheduling decisions |
| OutputProcessor | How to extract per-request output | Request state |

---

## Request Lifecycle

```
                    add_request()
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        WAITING                                   │
│  - Request received, queued for scheduling                      │
│  - Policy.can_schedule() checks resource availability           │
└─────────────────────────┬───────────────────────────────────────┘
                          │ schedule() + Policy.on_schedule()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RUNNING                                   │
│  - Actively being processed by ModelRunner                      │
│  - For AR: multiple iterations until EOS                        │
│  - For Encoder: single iteration                                │
│  - For DiT: fixed N iterations                                  │
└───────────────┬─────────────────────────────────────────────────┘
                │ Policy.is_finished() returns True
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINISHED                                  │
│  - Policy.on_finish() called (free resources)                   │
│  - Future resolved with result                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Engine ABC

```python
# ═══════════════════════════════════════════════════════════════════════════
# engines/base.py - Engine abstract base class
# ═══════════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from typing import Any

class Engine(ABC):
    """Abstract base class for all engines."""

    @abstractmethod
    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        ...

    @abstractmethod
    async def get_result(self, request_id: str) -> Any:
        """Get result for a request (blocks until ready)."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start the engine processing loop."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine processing loop."""
        ...
```

---

## Core Types (Generic - Model Agnostic)

```python
# ═══════════════════════════════════════════════════════════════════════════
# types.py - Generic types only (no model-specific fields)
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()

@dataclass
class Request:
    """
    Generic request container.

    The Scheduler only cares about:
    - request_id: identity
    - status: lifecycle state

    Everything else is stored in `data` (opaque to Scheduler).
    """
    request_id: str
    status: RequestStatus = RequestStatus.WAITING
    data: Any = None  # Model-specific, opaque to Scheduler

    # Timestamps (generic)
    arrival_time: float = 0.0
    finish_time: float | None = None

@dataclass
class SchedulerOutput:
    """
    Generic contract between Scheduler and ModelRunner.

    - requests: which requests to process
    - batch_data: opaque, built by Policy, consumed by InputPreparer
    """
    requests: list[Request]
    batch_data: Any  # Opaque - built by Policy, consumed by InputPreparer

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def request_ids(self) -> list[str]:
        return [r.request_id for r in self.requests]

@dataclass
class RequestOutput:
    """
    Generic output for a single request.

    The `data` field contains model-specific output
    (tokens, embeddings, latents, etc.)
    """
    request_id: str
    data: Any = None  # Model-specific output
    finished: bool = False
    finish_reason: str | None = None  # "stop", "length", "abort"

@dataclass
class ModelRunnerOutput:
    """Generic output from ModelRunner."""
    outputs: dict[str, RequestOutput]  # request_id → output
```

---

## Scheduler (Generic)

```python
# ═══════════════════════════════════════════════════════════════════════════
# scheduler.py - Generic scheduler (knows nothing about model specifics)
# ═══════════════════════════════════════════════════════════════════════════

from collections import deque
import asyncio
import time

class Scheduler:
    """
    Generic request scheduler.

    Responsibilities:
    - Manage request lifecycle (WAITING → RUNNING → FINISHED)
    - Delegate scheduling decisions to Policy
    - Produce SchedulerOutput for ModelRunner

    Does NOT know about:
    - Input formats (tokens, latents, etc.)
    - Model-specific batching logic
    - Resource details (KV cache, etc.)
    """

    def __init__(self, policy: "SchedulingPolicy", max_running: int = 256):
        self.policy = policy
        self.max_running = max_running

        # Request state
        self.requests: dict[str, Request] = {}
        self.waiting: deque[str] = deque()
        self.running: list[str] = []

        # Result futures (created lazily in get_result)
        self._futures: dict[str, asyncio.Future] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

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
        if request_id in self.requests:
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
        if request.status == RequestStatus.FINISHED:
            return request

        return await self._futures[request_id]

    # ─────────────────────────────────────────────────────────────────────────
    # Core Scheduling
    # ─────────────────────────────────────────────────────────────────────────

    def schedule(self) -> SchedulerOutput | None:
        """Schedule next batch. Returns None if no work."""
        if not self.waiting and not self.running:
            return None

        to_schedule: list[Request] = []

        # 1. Continue running requests
        for req_id in self.running:
            to_schedule.append(self.requests[req_id])

        # 2. Add waiting requests (if resources available)
        to_move = []
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
        model_output: ModelRunnerOutput
    ) -> list[Request]:
        """
        Update state from model output.
        Returns list of finished requests.
        """
        finished = []

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
```

---

## Policy Protocol (Model-Specific Interface)

```python
# ═══════════════════════════════════════════════════════════════════════════
# policy/base.py - Policy protocol
# ═══════════════════════════════════════════════════════════════════════════

from typing import Protocol

class SchedulingPolicy(Protocol):
    """
    Model-specific scheduling logic.

    The Policy is responsible for:
    1. Resource management (can we schedule this request?)
    2. Batch building (how to batch requests together?)
    3. Completion detection (is this request done?)

    It is NOT responsible for:
    - Request lifecycle (Scheduler does this)
    - Input/output transformation (InputPreparer/OutputProcessor do this)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Resource Management
    # ─────────────────────────────────────────────────────────────────────────

    def can_schedule(self, request: Request) -> bool:
        """Can this request be scheduled? (resources available?)"""
        ...

    def on_schedule(self, request: Request) -> None:
        """Called when request moves WAITING → RUNNING. Allocate resources."""
        ...

    def on_finish(self, request: Request) -> None:
        """Called when request finishes. Free resources."""
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # Batch Building
    # ─────────────────────────────────────────────────────────────────────────

    def build_batch(self, requests: list[Request]) -> Any:
        """
        Build model-specific batch data from requests.

        Returns opaque batch_data that will be passed to InputPreparer.

        For Encoder: might return EncoderBatchData
        For AR: might return ARBatchData with positions, block_table, etc.
        For DiT: might return DiTBatchData with latents, timesteps, etc.
        """
        ...

    # ─────────────────────────────────────────────────────────────────────────
    # State Update
    # ─────────────────────────────────────────────────────────────────────────

    def update_request(self, request: Request, output: RequestOutput) -> None:
        """Update request state after model execution."""
        ...

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        """Check if request is finished."""
        ...
```

---

## ModelRunner (Generic)

```python
# ═══════════════════════════════════════════════════════════════════════════
# model_runner.py - Generic model runner
# ═══════════════════════════════════════════════════════════════════════════

import torch
from typing import Protocol, Any

class InputPreparer(Protocol):
    """Converts SchedulerOutput.batch_data to model inputs."""

    def prepare(self, batch_data: Any, device: torch.device) -> dict[str, Any]:
        """
        Convert opaque batch_data to model input dict.

        For Encoder: padded input_ids + attention_mask
        For AR: input_ids + positions + block_table + ...
        For DiT: latents + timesteps + ...
        """
        ...

class OutputProcessor(Protocol):
    """Converts model outputs to RequestOutputs."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        """
        Convert model output to per-request outputs.

        For Encoder: extract embeddings per request
        For AR: sample tokens per request
        For DiT: extract denoised latents per request
        """
        ...

class ModelRunner:
    """
    Generic model executor.

    Responsibilities:
    - Convert SchedulerOutput to model inputs (via InputPreparer)
    - Execute model forward pass
    - Convert model outputs to RequestOutputs (via OutputProcessor)

    Completely stateless. All state lives in Scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_preparer: InputPreparer,
        output_processor: OutputProcessor,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model.to(device).eval()
        self.input_preparer = input_preparer
        self.output_processor = output_processor
        self.device = device

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute model on batch."""
        # 1. Prepare inputs (model-specific)
        model_inputs = self.input_preparer.prepare(
            scheduler_output.batch_data,
            self.device
        )

        # 2. Forward pass
        with torch.inference_mode():
            model_output = self.model(**model_inputs)

        # 3. Process outputs (model-specific)
        request_outputs = self.output_processor.process(
            model_output,
            scheduler_output
        )

        return ModelRunnerOutput(outputs=request_outputs)
```

---

## OmniEngine

```python
# ═══════════════════════════════════════════════════════════════════════════
# engine.py - OmniEngine combining scheduler and runner
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
from typing import Any
from ..base import Engine

class OmniEngine(Engine):
    """
    Unified engine for all model types.

    Execution model:
    - Busy loop: schedule() → execute() → update()
    - Async-friendly: add_request() and get_result() are async
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: ModelRunner,
    ):
        self.scheduler = scheduler
        self.model_runner = model_runner

        self._running = False
        self._loop_task: asyncio.Task | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        self.scheduler.add_request(request_id, data)

    async def get_result(self, request_id: str) -> Any:
        """Get result for a request (blocks until ready)."""
        request = await self.scheduler.get_result(request_id)
        return request.data

    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        self.scheduler.abort_request(request_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the engine processing loop."""
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the engine processing loop."""
        self._running = False
        if self._loop_task:
            await self._loop_task
            self._loop_task = None

    # ─────────────────────────────────────────────────────────────────────────
    # Processing Loop
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            await self._step()
            await asyncio.sleep(0)  # Yield to other coroutines

    async def _step(self) -> bool:
        """Execute one step. Returns True if work was done."""
        # 1. Schedule
        scheduler_output = self.scheduler.schedule()

        if scheduler_output is None:
            await asyncio.sleep(0.001)  # Brief sleep when idle
            return False

        # 2. Execute (run in executor to not block event loop)
        loop = asyncio.get_event_loop()
        model_output = await loop.run_in_executor(
            None,
            self.model_runner.execute,
            scheduler_output
        )

        # 3. Update state
        self.scheduler.update(scheduler_output, model_output)

        return True
```

---

## Model-Specific Implementations

### Encoder

```python
# ═══════════════════════════════════════════════════════════════════════════
# policy/encoder.py - Encoder-specific implementation
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Encoder-specific data structures
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class EncoderPolicy:
    """
    Scheduling policy for encoder models.

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

# ─────────────────────────────────────────────────────────────────────────────
# InputPreparer
# ─────────────────────────────────────────────────────────────────────────────

class EncoderInputPreparer:
    """Converts EncoderBatchData to model inputs."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def prepare(self, batch_data: EncoderBatchData, device: torch.device) -> dict:
        max_len = max(batch_data.seq_lens)
        batch_size = len(batch_data.input_ids_list)

        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.float,
            device=device
        )

        for i, ids in enumerate(batch_data.input_ids_list):
            seq_len = len(ids)
            input_ids[i, :seq_len] = ids.to(device)
            attention_mask[i, :seq_len] = 1.0

        return {"input_ids": input_ids, "attention_mask": attention_mask}

# ─────────────────────────────────────────────────────────────────────────────
# OutputProcessor
# ─────────────────────────────────────────────────────────────────────────────

class EncoderOutputProcessor:
    """Extracts embeddings from encoder output."""

    def __init__(self, pooling: str = "last"):
        self.pooling = pooling  # "last", "mean", "cls"

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        hidden_states = model_output.last_hidden_state  # [batch, seq, hidden]
        batch_data: EncoderBatchData = scheduler_output.batch_data

        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            seq_len = batch_data.seq_lens[i]

            if self.pooling == "last":
                emb = hidden_states[i, seq_len - 1]
            elif self.pooling == "mean":
                emb = hidden_states[i, :seq_len].mean(dim=0)
            else:  # cls
                emb = hidden_states[i, 0]

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=emb,
                finished=True,
            )

        return outputs
```

### AR (Autoregressive)

```python
# ═══════════════════════════════════════════════════════════════════════════
# policy/ar.py - AR-specific implementation
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
import torch

# ─────────────────────────────────────────────────────────────────────────────
# AR-specific data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ARRequestData:
    """AR-specific request data (stored in Request.data)."""
    input_ids: torch.Tensor
    output_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    # For paged attention (optional, can start with simple KV cache)
    block_ids: list[int] = field(default_factory=list)

    # For simple HF-style KV cache
    past_key_values: tuple | None = None

@dataclass
class ARBatchData:
    """AR-specific batch data (SchedulerOutput.batch_data)."""
    input_ids: torch.Tensor          # [num_tokens]
    positions: torch.Tensor          # [num_tokens]
    seq_lens: list[int]              # Total length per sequence
    query_lens: list[int]            # New tokens this step

    # For paged attention
    block_table: list[list[int]] | None = None
    context_lens: list[int] | None = None

    # For simple HF-style KV cache
    past_key_values_list: list[tuple] | None = None

# ─────────────────────────────────────────────────────────────────────────────
# Policy (Simple version - HF KV cache)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleARPolicy:
    """
    Simple AR policy using HF-style KV cache.

    For initial development. Can upgrade to paged attention later.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        eos_token_id: int = 2,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self._count = 0

    def can_schedule(self, request: Request) -> bool:
        return self._count < self.max_batch_size

    def on_schedule(self, request: Request) -> None:
        self._count += 1

    def on_finish(self, request: Request) -> None:
        self._count = max(0, self._count - 1)
        # Clear KV cache
        request.data.past_key_values = None

    def build_batch(self, requests: list[Request]) -> ARBatchData:
        all_input_ids = []
        all_positions = []
        seq_lens = []
        query_lens = []
        past_key_values_list = []

        for request in requests:
            data: ARRequestData = request.data
            is_prefill = data.num_computed_tokens == 0

            if is_prefill:
                # Prefill: all input tokens
                ids = data.input_ids
                num_new = len(ids)
                pos = torch.arange(num_new)
            else:
                # Decode: last generated token
                last_token = data.output_ids[-1]
                ids = torch.tensor([last_token])
                num_new = 1
                pos = torch.tensor([data.num_computed_tokens])

            all_input_ids.append(ids)
            all_positions.append(pos)
            seq_lens.append(data.num_computed_tokens + num_new)
            query_lens.append(num_new)
            past_key_values_list.append(data.past_key_values)

        return ARBatchData(
            input_ids=torch.cat(all_input_ids),
            positions=torch.cat(all_positions),
            seq_lens=seq_lens,
            query_lens=query_lens,
            past_key_values_list=past_key_values_list,
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

        if token == self.eos_token_id:
            return True
        if data.num_computed_tokens >= self.max_seq_len:
            return True
        return False

# ─────────────────────────────────────────────────────────────────────────────
# InputPreparer (Simple - single request for now)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleARInputPreparer:
    """Simple AR input preparer for HF models (single request)."""

    def prepare(self, batch_data: ARBatchData, device: torch.device) -> dict:
        # For simplicity, assume single request
        # TODO: Handle batching with attention masks

        input_ids = batch_data.input_ids.unsqueeze(0).to(device)

        past_kv = None
        if batch_data.past_key_values_list and batch_data.past_key_values_list[0] is not None:
            past_kv = batch_data.past_key_values_list[0]

        return {
            "input_ids": input_ids,
            "past_key_values": past_kv,
            "use_cache": True,
        }

# ─────────────────────────────────────────────────────────────────────────────
# OutputProcessor
# ─────────────────────────────────────────────────────────────────────────────

class SimpleAROutputProcessor:
    """Simple AR output processor with greedy sampling."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        logits = model_output.logits  # [batch, seq, vocab]
        past_key_values = model_output.past_key_values

        # Greedy sample from last position
        next_token = logits[:, -1, :].argmax(dim=-1).item()

        # Single request for now
        request = scheduler_output.requests[0]

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=(next_token, past_key_values),
                finished=False,  # Policy decides this
            )
        }
```

### DiT (Diffusion Transformer)

```python
# ═══════════════════════════════════════════════════════════════════════════
# policy/dit.py - DiT-specific implementation
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
import torch

# ─────────────────────────────────────────────────────────────────────────────
# DiT-specific data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiTRequestData:
    """DiT-specific request data (stored in Request.data)."""
    latents: torch.Tensor            # Current latents
    condition: torch.Tensor | None   # Text/image condition
    current_step: int = 0

@dataclass
class DiTBatchData:
    """DiT-specific batch data (SchedulerOutput.batch_data)."""
    latents: torch.Tensor            # [batch, C, H, W]
    timesteps: torch.Tensor          # [batch]
    conditions: torch.Tensor | None  # [batch, seq, hidden]

# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class DiTPolicy:
    """
    Scheduling policy for Diffusion Transformer models.

    Characteristics:
    - Fixed number of denoising steps
    - No KV cache
    - Can batch by step number
    """

    def __init__(self, num_steps: int = 50, max_batch_size: int = 16):
        self.num_steps = num_steps
        self.max_batch_size = max_batch_size
        self._count = 0

    def can_schedule(self, request: Request) -> bool:
        return self._count < self.max_batch_size

    def on_schedule(self, request: Request) -> None:
        self._count += 1

    def on_finish(self, request: Request) -> None:
        self._count = max(0, self._count - 1)

    def build_batch(self, requests: list[Request]) -> DiTBatchData:
        latents = torch.stack([r.data.latents for r in requests])
        timesteps = torch.tensor([r.data.current_step for r in requests])

        conditions = None
        if requests[0].data.condition is not None:
            conditions = torch.stack([r.data.condition for r in requests])

        return DiTBatchData(
            latents=latents,
            timesteps=timesteps,
            conditions=conditions,
        )

    def update_request(self, request: Request, output: RequestOutput) -> None:
        request.data.latents = output.data  # Updated latents
        request.data.current_step += 1

    def is_finished(self, request: Request, output: RequestOutput) -> bool:
        return request.data.current_step >= self.num_steps

# ─────────────────────────────────────────────────────────────────────────────
# InputPreparer
# ─────────────────────────────────────────────────────────────────────────────

class DiTInputPreparer:
    """Converts DiTBatchData to model inputs."""

    def __init__(self, scheduler: Any):  # Diffusion scheduler
        self.scheduler = scheduler

    def prepare(self, batch_data: DiTBatchData, device: torch.device) -> dict:
        # Convert step index to actual timestep value
        timesteps = self.scheduler.timesteps[batch_data.timesteps]

        return {
            "hidden_states": batch_data.latents.to(device),
            "timestep": timesteps.to(device),
            "encoder_hidden_states": batch_data.conditions.to(device) if batch_data.conditions is not None else None,
        }

# ─────────────────────────────────────────────────────────────────────────────
# OutputProcessor
# ─────────────────────────────────────────────────────────────────────────────

class DiTOutputProcessor:
    """Processes DiT output (denoising step)."""

    def __init__(self, scheduler: Any):  # Diffusion scheduler
        self.scheduler = scheduler

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput
    ) -> dict[str, RequestOutput]:
        # model_output is noise prediction
        noise_pred = model_output.sample
        batch_data: DiTBatchData = scheduler_output.batch_data

        outputs = {}
        for i, request in enumerate(scheduler_output.requests):
            # Apply scheduler step to get denoised latents
            step_output = self.scheduler.step(
                noise_pred[i],
                batch_data.timesteps[i],
                batch_data.latents[i],
            )

            outputs[request.request_id] = RequestOutput(
                request_id=request.request_id,
                data=step_output.prev_sample,
                finished=False,
            )

        return outputs
```

---

## Factory Functions

```python
# ═══════════════════════════════════════════════════════════════════════════
# factory.py - Factory functions for creating engines
# ═══════════════════════════════════════════════════════════════════════════

def create_encoder_engine(
    model: torch.nn.Module,
    tokenizer: Any,
    max_batch_size: int = 32,
    device: str = "cuda",
) -> OmniEngine:
    """
    Create an encoder engine.

    Example:
        engine = create_encoder_engine(bert_model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Hello world", return_tensors="pt")
        data = EncoderRequestData(input_ids=input_ids[0])

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")
    """
    policy = EncoderPolicy(max_batch_size=max_batch_size)
    scheduler = Scheduler(policy=policy, max_running=max_batch_size)

    model_runner = ModelRunner(
        model=model,
        input_preparer=EncoderInputPreparer(
            pad_token_id=tokenizer.pad_token_id or 0
        ),
        output_processor=EncoderOutputProcessor(pooling="last"),
        device=torch.device(device),
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)


def create_simple_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> OmniEngine:
    """
    Create a simple AR engine (single request, HF KV cache).

    For initial development. Upgrade to batched/paged version later.

    Example:
        engine = create_simple_ar_engine(llama_model, tokenizer)
        await engine.start()

        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
        data = ARRequestData(input_ids=input_ids[0])

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # ARRequestData with output_ids
    """
    policy = SimpleARPolicy(
        max_batch_size=1,  # Single request for now
        max_seq_len=max_seq_len,
        eos_token_id=tokenizer.eos_token_id or 2,
    )
    scheduler = Scheduler(policy=policy, max_running=1)

    model_runner = ModelRunner(
        model=model,
        input_preparer=SimpleARInputPreparer(),
        output_processor=SimpleAROutputProcessor(),
        device=torch.device(device),
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)
```

---

## File Structure

```
sglang_omni/
├── engines/
│   ├── base.py                           # Engine ABC
│   │
│   └── omni/
│       ├── __init__.py                   # Public exports
│       │
│       ├── types.py                      # Generic types only
│       │   - Request
│       │   - RequestStatus
│       │   - SchedulerOutput
│       │   - RequestOutput
│       │   - ModelRunnerOutput
│       │
│       ├── scheduler.py                  # Generic Scheduler
│       │
│       ├── model_runner.py               # Generic ModelRunner
│       │   - InputPreparer protocol
│       │   - OutputProcessor protocol
│       │
│       ├── engine.py                     # OmniEngine
│       │
│       ├── policy/                       # Model-type-specific support
│       │   ├── __init__.py
│       │   ├── base.py                   # SchedulingPolicy protocol
│       │   ├── encoder.py                # Encoder support (see below)
│       │   ├── ar.py                     # AR support
│       │   └── dit.py                    # DiT support
│       │
│       └── factory.py                    # create_*_engine functions
```

### What Each Policy File Contains

```python
# policy/encoder.py - Everything needed to support Encoder models

# 1. Data structures (what Request.data and batch_data contain)
@dataclass
class EncoderRequestData: ...

@dataclass
class EncoderBatchData: ...

# 2. Scheduling logic (when/how to batch)
class EncoderPolicy: ...

# 3. Input/Output transformation (batch_data ↔ tensors)
class EncoderInputPreparer: ...
class EncoderOutputProcessor: ...
```

Note: The actual nn.Module (BERT, LLaMA, DiT) comes from **outside** — passed into the factory:

```python
# User provides the actual model
from transformers import BertModel
bert = BertModel.from_pretrained("bert-base")

# Factory wires it up with Encoder-specific support
engine = create_encoder_engine(
    model=bert,           # ← Actual nn.Module from user
    tokenizer=tokenizer,
)
```

So `policy/encoder.py` doesn't contain BERT — it contains the logic to **schedule and batch requests** for any encoder model.

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OmniEngine._run_loop()                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SCHEDULE                                                                  │
│                                                                              │
│    scheduler.schedule()                                                      │
│    ├── Check waiting requests                                               │
│    ├── policy.can_schedule() → resource check                               │
│    ├── policy.on_schedule() → allocate resources                            │
│    └── policy.build_batch() → model-specific batch_data                     │
│                                                                              │
│    Output: SchedulerOutput                                                  │
│    ├── requests: [Request, ...]                                             │
│    └── batch_data: Any (opaque, model-specific)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. EXECUTE                                                                   │
│                                                                              │
│    model_runner.execute(scheduler_output)                                   │
│    ├── input_preparer.prepare(batch_data) → model inputs                    │
│    ├── model(**inputs) → model outputs                                      │
│    └── output_processor.process() → RequestOutputs                          │
│                                                                              │
│    Output: ModelRunnerOutput                                                │
│    └── outputs: {req_id: RequestOutput(data=...)}                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. UPDATE                                                                    │
│                                                                              │
│    scheduler.update(scheduler_output, model_output)                         │
│    ├── For each request:                                                    │
│    │   ├── policy.update_request() → update Request.data                    │
│    │   └── if policy.is_finished(): _finish_request()                       │
│    │       └── policy.on_finish() → free resources                          │
│    └── Resolve futures for finished requests                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              [Loop continues]
```

---

## Implementation Order

1. **Phase 1**: `types.py` - Generic types
2. **Phase 2**: `scheduler.py` - Generic Scheduler
3. **Phase 3**: `model_runner.py` - Generic ModelRunner
4. **Phase 4**: `engine.py` - OmniEngine
5. **Phase 5**: `policy/encoder.py` + preparers/processors - Test with BERT
6. **Phase 6**: `policy/ar.py` (simple version) - Test with LLaMA
7. **Phase 7**: Upgrade AR to batched/paged attention
8. **Phase 8**: `policy/dit.py` - Test with DiT

---

## Notes on Preprocess

Preprocess (image resize, mel spectrum, tokenization, etc.) is intentionally **not** part of this design. It can be:

1. **Done by caller before `add_request()`** - Simplest approach
2. **A separate PreprocessRunner** - For async CPU preprocessing
3. **A separate stage in a multi-stage pipeline** - PreprocessStage → EncoderStage → LLMStage

This keeps the core engine focused on GPU execution.
