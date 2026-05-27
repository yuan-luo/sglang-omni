# SPDX-License-Identifier: Apache-2.0
"""Bootstrap for the HunyuanImage-3 AR scheduler.

Wires up the SGLang infrastructure (worker, caches, managers) + our
ModelRunner + request/result adapters into an OmniScheduler.

Pure setup code — no inference logic. Forward-path math lives in
`ar_model.py`. Per-request token assembly lives in `request_builders.py`.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_hunyuan_image3_ar_scheduler(
    server_args: Any,
    *,
    model_path: str,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Construct an OmniScheduler running the HunyuanImage-3 AR backbone.

    The returned scheduler accepts `StagePayload` items (driven by the pipeline
    coordinator) and produces AR token-id sequences.
    """
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if getattr(server_args, "tp_size", None) != tp_size:
        server_args.tp_size = tp_size

    from transformers import AutoTokenizer

    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.models.hunyuan_image3.request_builders import (
        make_ar_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="HunyuanImage3ForCausalMM",
    )

    # Pull vocab size from the HF config (fallback to tokenizer's view).
    hf_cfg = getattr(model_config, "hf_config", model_config)
    vocab_size = (
        getattr(hf_cfg, "vocab_size", None)
        or getattr(tokenizer, "vocab_size", None)
        or 133120
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    # Plain ModelRunner — no multimodal-prefill-injection hook for AR-only t2i.
    # When image inputs are wired (it2i / i2t modalities), a subclass will be
    # added that injects image embeddings at the AR-input positions, analogous
    # to MingThinkerModelRunner.
    model_runner = ModelRunner(model_worker, output_proc)

    request_builder, result_adapter = make_ar_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
    )
