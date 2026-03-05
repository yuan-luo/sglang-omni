# SPDX-License-Identifier: Apache-2.0
"""SGLang AR backend integration into OmniEngine's Protocol system.

Provides BatchPlanner, ResourceManager, OutputProcessor, IterationController,
and ModelRunner implementations that delegate to SGLang's PrefillManager,
DecodeManager, and ModelWorker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.mem_cache.common import release_kv_cache

from ..types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)
from .ar import ARRequestData

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager

logger = logging.getLogger(__name__)


@dataclass
class SGLangARRequestData(ARRequestData):
    """SGLang specific AR request data."""

    req: Any = None
    synced: bool = False
    hidden_states_buffer: dict[int, list[torch.Tensor]] = field(default_factory=dict)


class SGLangBatchPlanner:
    """Batch planner that delegates to SGLang's PrefillManager and DecodeManager."""

    def __init__(
        self,
        prefill_manager: "PrefillManager",
        decode_manager: "DecodeManager",
        server_args: "ServerArgs",
    ):
        self.prefill_manager = prefill_manager
        self.decode_manager = decode_manager
        self.server_args = server_args
        self.last_batch: Any | None = None
        self.forward_ct: int = 0
        self.req_id_map: dict[str, SchedulerRequest] = {}
        self._cached_schedule_batch: Any | None = None

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: Any,
    ) -> list[SchedulerRequest]:
        self._post_step_operations()
        active_request_ids = {req.request_id for req in waiting}
        active_request_ids.update(req.request_id for req in running)
        self._prune_inactive_state(active_request_ids)

        for sched_req in waiting:
            data: SGLangARRequestData = sched_req.data
            if not data.synced:
                self.prefill_manager.add_one_request(data.req)
                data.synced = True
                self.req_id_map[data.req.rid] = sched_req

        running_batch = self.decode_manager.running_batch
        running_bs = running_batch.batch_size()
        num_allocatable_reqs = max(
            self.server_args.max_running_requests - running_bs, 0
        )

        running_batch_for_prefill = self.decode_manager.running_batch
        if (
            running_batch_for_prefill is not None
            and running_batch_for_prefill.is_empty()
        ):
            running_batch_for_prefill = None

        schedule_batch = self.prefill_manager.schedule_next_batch(
            running_batch_for_prefill,
            num_allocatable_reqs,
            new_token_ratio=self.decode_manager.new_token_ratio,
        )

        if schedule_batch is None and self.decode_manager.runnable:
            schedule_batch = self.decode_manager.schedule_next_batch(self.forward_ct)

        if schedule_batch is None:
            self._cached_schedule_batch = None
            return []

        self._cached_schedule_batch = schedule_batch
        self.forward_ct += 1

        selected: list[SchedulerRequest] = []
        keep_indices: list[int] = []
        for i, req in enumerate(schedule_batch.reqs):
            sched_req = self.req_id_map.get(req.rid)
            if sched_req is not None and sched_req.status not in (
                SchedulerStatus.FINISHED,
                SchedulerStatus.ABORTED,
            ):
                keep_indices.append(i)
                selected.append(sched_req)
            elif sched_req is None:
                logger.warning("SGLang req %s not found in req_id_map", req.rid)

        if len(keep_indices) != len(schedule_batch.reqs):
            if keep_indices:
                schedule_batch.filter_batch(keep_indices=keep_indices)
                self._cached_schedule_batch = schedule_batch
            else:
                self._cached_schedule_batch = None
                return []

        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return self._cached_schedule_batch

    def _post_step_operations(self):
        """Handle post-step merging from last prefill batch into decode batch."""
        chunked_req_to_exclude = set()
        active_chunked_req = self.prefill_manager.chunked_req
        if active_chunked_req is not None:
            chunked_req_to_exclude.add(active_chunked_req)
            self.prefill_manager.tree_cache.cache_unfinished_req(
                active_chunked_req, chunked=True
            )
            if active_chunked_req.req_pool_idx is not None:
                self.prefill_manager.req_to_token_pool.free(
                    active_chunked_req.req_pool_idx
                )

        if self.last_batch is None:
            return

        if self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req is not None:
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.decode_manager.running_batch.batch_is_full = False

            if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
                if self.decode_manager.running_batch.is_empty():
                    self.decode_manager.running_batch = self.last_batch
                else:
                    self.decode_manager.running_batch.merge_batch(self.last_batch)

        if not self.decode_manager.running_batch.is_empty():
            finished_indices = []
            for i, req in enumerate(self.decode_manager.running_batch.reqs):
                sched_req = self.req_id_map.get(req.rid)
                if req.finished() or (
                    sched_req is not None
                    and sched_req.status
                    in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED)
                ):
                    finished_indices.append(i)

            if finished_indices:
                keep = [
                    i
                    for i in range(len(self.decode_manager.running_batch.reqs))
                    if i not in finished_indices
                ]
                if keep:
                    self.decode_manager.running_batch.filter_batch(keep_indices=keep)
                else:
                    from sglang.srt.managers.schedule_batch import ScheduleBatch

                    self.decode_manager.running_batch = ScheduleBatch(
                        reqs=[], batch_is_full=False
                    )

        self.last_batch = None

    def record_last_batch(self, schedule_batch: Any):
        """Called after model execution to record the last batch for post-step."""
        self.last_batch = schedule_batch

    def _prune_inactive_state(self, active_request_ids: set[str]) -> None:
        """Prune stale request mappings and remove inactive reqs from SGLang queues."""
        inactive_rids: set[str] = set()
        for rid, sched_req in list(self.req_id_map.items()):
            if sched_req.request_id not in active_request_ids or sched_req.status in (
                SchedulerStatus.FINISHED,
                SchedulerStatus.ABORTED,
            ):
                inactive_rids.add(rid)
                del self.req_id_map[rid]

        if not inactive_rids:
            return

        if self.prefill_manager.waiting_queue:
            self.prefill_manager.waiting_queue = [
                req
                for req in self.prefill_manager.waiting_queue
                if req.rid not in inactive_rids
            ]

        running_batch = self.decode_manager.running_batch
        if running_batch is None or running_batch.is_empty():
            return
        keep_indices = [
            i
            for i, req in enumerate(running_batch.reqs)
            if req.rid not in inactive_rids
        ]
        if len(keep_indices) == len(running_batch.reqs):
            return
        if keep_indices:
            running_batch.filter_batch(keep_indices=keep_indices)
        else:
            from sglang.srt.managers.schedule_batch import ScheduleBatch

            self.decode_manager.running_batch = ScheduleBatch(
                reqs=[], batch_is_full=False
            )


class SGLangResourceManager:
    """Resource manager that delegates KV cache management to SGLang internals."""

    def __init__(self, token_to_kv_pool_allocator, req_to_token_pool, tree_cache):
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return True

    def allocate(self, request: SchedulerRequest) -> None:
        pass

    def free(self, request: SchedulerRequest) -> None:
        data: SGLangARRequestData = request.data
        if data.req is not None:
            release_kv_cache(data.req, self.tree_cache)


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        token_list = (
            model_output.next_token_ids.tolist()
            if model_output.next_token_ids is not None
            else []
        )

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
            )
        return outputs


HIDDEN_LAYER_KEY_MAP: dict[int, str] = {
    -1: "thinker_embed",
    24: "thinker_hidden",
}


class SGLangIterationController:
    """Handles per-request state updates with chunked prefill semantics."""

    def __init__(self, tree_cache: Any) -> None:
        self.tree_cache = tree_cache

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: SGLangARRequestData = request.data
        req = data.req

        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return

        token_id = output.data
        if token_id is not None:
            req.output_ids.append(token_id)
            req.check_finished()
            if not req.finished() and req.decode_batch_idx == 0:
                self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        # TODO (chenyang): This looks a bit weird, how to refactor
        is_done = request.data.req.finished()
        if is_done:
            self._finalize_hidden_states(request)
        return is_done

    def _finalize_hidden_states(self, request: SchedulerRequest) -> None:
        """Concatenate accumulated hidden state chunks into final tensors.

        thinker_embed and thinker_hidden are stored in
        request.data.extra_model_outputs for the talker.
        """
        data: SGLangARRequestData = request.data
        if not data.hidden_states_buffer:
            return
        for layer_idx, chunks in data.hidden_states_buffer.items():
            key = HIDDEN_LAYER_KEY_MAP.get(layer_idx, f"hidden_layer_{layer_idx}")
            if chunks:
                data.extra_model_outputs[key] = torch.cat(chunks, dim=0)
        data.hidden_states_buffer.clear()


class HiddenStateCaptureHook:
    """Captures intermediate hidden states from specific decoder layers."""

    def __init__(self) -> None:
        self._captured: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def register(
        self,
        layers: torch.nn.ModuleList,
        embed_tokens: torch.nn.Module,
        layer_indices: list[int],
    ) -> None:
        """Register forward hooks on the given decoder layers.

        Args:
            layers: The decoder layer list (e.g. ``model.thinker.model.layers``).
            embed_tokens: The embedding module (e.g. ``model.thinker.model.embed_tokens``).
            layer_indices: Layer indices to capture. ``-1`` means embed_tokens.
        """
        for idx in layer_indices:
            if idx == -1:
                handle = embed_tokens.register_forward_hook(self._make_hook(idx))
            else:
                assert idx < len(
                    layers
                ), f"Layer index {idx} out of range (model has {len(layers)} layers)"
                handle = layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        """Create a forward hook that captures the hidden state for layer_idx.

        Note that decoder layers may return (hidden, ...) tuples or a bare
        tensor depending on the model implementation.
        """

        def _hook(module: Any, input_args: Any, output: Any) -> None:
            if isinstance(output, tuple) and len(output) >= 1:
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                return
            self._captured[layer_idx] = hidden.detach()

        return _hook

    def pop_captured(self) -> dict[int, torch.Tensor]:
        """Return and clear captured hidden states from the last forward pass."""
        captured = self._captured
        self._captured = {}
        return captured

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class SGLangModelRunner:
    """Model runner that uses SGLang's ModelWorker for execution.

    Note that SGLangModelRunner optionally captures intermediate
    hidden states via forward hooks for thinker-to-talker architecture.
    """

    def __init__(
        self,
        model_worker: "ModelWorker",
        output_processor: SGLangOutputProcessor,
        batch_planner: SGLangBatchPlanner | None = None,
        capture_hidden_layers: list[int] | None = None,
        decoder_layers: torch.nn.ModuleList | None = None,
        embed_tokens: torch.nn.Module | None = None,
    ):
        self.model_worker = model_worker
        self.output_processor = output_processor
        self.batch_planner = batch_planner
        self._hidden_hook: HiddenStateCaptureHook | None = None

        if capture_hidden_layers:
            assert (
                decoder_layers is not None
            ), "decoder_layers must be provided when capture_hidden_layers is set"
            assert (
                embed_tokens is not None
            ), "embed_tokens must be provided when capture_hidden_layers is set"
            self._hidden_hook = HiddenStateCaptureHook()
            self._hidden_hook.register(
                decoder_layers, embed_tokens, capture_hidden_layers
            )

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        schedule_batch = scheduler_output.batch_data

        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_worker_batch = schedule_batch.get_model_worker_batch()

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )

        batch_result = self.model_worker.forward_batch_generation(forward_batch)

        if self._hidden_hook is not None:
            captured = self._hidden_hook.pop_captured()
            if captured:
                self._store_captured_hidden(captured, scheduler_output, schedule_batch)

        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(model_worker_batch.seq_lens),
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )
        else:
            batch_result.next_token_ids = self.model_worker.model_runner.sample(
                batch_result.logits_output, forward_batch
            )
        schedule_batch.output_ids = batch_result.next_token_ids

        if self.batch_planner is not None:
            self.batch_planner.record_last_batch(schedule_batch)

        outputs = self.output_processor.process(batch_result, scheduler_output)

        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    def _store_captured_hidden(
        self,
        captured: dict[int, torch.Tensor],
        scheduler_output: SchedulerOutput,
        schedule_batch: Any,
    ) -> None:
        """Accumulate captured hidden states onto per-request data buffers."""
        seq_lens = list(schedule_batch.seq_lens)
        offsets: list[int] = []
        running_offset = 0
        for length in seq_lens:
            offsets.append(running_offset)
            running_offset += int(length)

        for i, sched_req in enumerate(scheduler_output.requests):
            data: SGLangARRequestData = sched_req.data
            start = offsets[i]
            end = start + int(seq_lens[i])
            for layer_idx, tensor in captured.items():
                buf = data.hidden_states_buffer.setdefault(layer_idx, [])
                buf.append(tensor[start:end].clone())
