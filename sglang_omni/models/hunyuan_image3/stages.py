# SPDX-License-Identifier: Apache-2.0
"""Stage factories for HunyuanImage-3.0.

Exposes:
  - `create_ar_executor`     — AR backbone (SGLang scheduler).
  - `create_dit_executor`    — DiT flow-matching pipeline (SimpleScheduler).

Heavy runtime imports are intentionally local to the factory body so that
importing `config.py` remains usable in lightweight environments without
sglang installed.
"""

from __future__ import annotations

from typing import Any


def create_ar_executor(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 22800,
    trust_remote_code: bool = True,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Build an SGLang scheduler that runs the HunyuanImage-3 AR backbone.

    Args:
      model_path: HF model dir or remote model id. Must point to a checkpoint
        containing the HunyuanImage3 backbone (Instruct or Distil variant).
      gpu_id: GPU index for this stage instance (per-worker).
      tp_rank / tp_size: Tensor parallel rank/size.
      nccl_port: Optional NCCL port (auto-assigned if None).
      max_seq_len: Model context length. Defaults to 22800 from the checkpoint
        config (`max_position_embeddings`).
      trust_remote_code: Forwarded to HF tokenizer/config loaders. Tencent's
        tokenizer (`tokenization_hunyuan_image_3.py`) is auto-loaded via
        `auto_map`, so this must be True.
      server_args_overrides: Extra kwargs forwarded to
        `build_sglang_server_args`. Use this to pin `attention_backend`,
        `disable_cuda_graph`, etc.

    Returns:
      A scheduler instance ready to be wired into the pipeline coordinator.
    """
    from sglang_omni.models.hunyuan_image3.bootstrap import (
        create_hunyuan_image3_ar_scheduler,
    )
    from sglang_omni.models.hunyuan_image3.registration import (
        register_hunyuan_image3_hf_config,
        register_hunyuan_image3_model_registry,
    )
    from sglang_omni.scheduling.sglang_backend.server_args_builder import (
        build_sglang_server_args,
    )

    register_hunyuan_image3_hf_config()
    register_hunyuan_image3_model_registry()

    overrides = dict(server_args_overrides or {})
    overrides.setdefault("trust_remote_code", trust_remote_code)
    overrides["tp_size"] = tp_size

    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        **overrides,
    )

    return create_hunyuan_image3_ar_scheduler(
        server_args,
        model_path=model_path,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
    )


def create_dit_executor(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    vae_scale_factor: int = 16,
):
    """Build the DiT flow-matching executor for HunyuanImage-3.

    Args:
      model_path: Checkpoint dir. The DiT shares the 80B backbone weights
        with the AR stage.
      gpu_id / tp_rank / tp_size: GPU placement.
      vae_scale_factor: Pixel-to-latent compression ratio (16 for DC-AE).

    Returns:
      A SimpleScheduler that consumes AR→DiT projected payloads and
      produces a payload with `image_data` / `image_format` / `image_size`.
    """
    from sglang_omni.models.hunyuan_image3.dit_bootstrap import (
        create_hunyuan_image3_dit_scheduler,
    )

    return create_hunyuan_image3_dit_scheduler(
        model_path=model_path,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        vae_scale_factor=vae_scale_factor,
    )
