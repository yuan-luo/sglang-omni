# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for HunyuanImage-3.0.

Variants:
  - `ar_only`  — single AR-backbone stage. AR-parity testing.
  - `dit_only` — single DiT pipeline stage. DiT correctness with synthetic
                 AR K/V (not yet implemented — needs synthetic K/V harness).
  - `default`  — full AR → DiT pipeline (production). Two stages fused onto
                 the same 8-GPU TP=8 plane so the 80B backbone weights are
                 loaded once and the AR K/V is transferred via the in-process
                 SHM relay.
"""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config.schema import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.hunyuan_image3"

# Stage name constants. DIT_STAGE will become a real stage when the DiT
# pipeline is wired up.
AR_STAGE = "ar"
DIT_STAGE = "dit"


def _ar_only_config(model_path: str, *, tp_size: int = 8) -> PipelineConfig:
    """Single-stage AR-only Variant — used for AR-parity testing."""
    devices = list(range(tp_size))
    ar_stage = StageConfig(
        name=AR_STAGE,
        factory=f"{_PKG}.stages.create_ar_executor",
        factory_args={
            "model_path": model_path,
            "trust_remote_code": True,  # Tencent tokenizer requires this
        },
        gpu=devices,
        tp_size=tp_size,
        terminal=True,
        # attention_backend: Triton by default; FLASHINFER is a potential
        # future optimization if compatible with HunyuanImage-3's mask shape.
    )
    return PipelineConfig(
        name="hunyuan_image3_ar_only",
        model_path=model_path,
        entry_stage=AR_STAGE,
        stages=[ar_stage],
    )


def _default_config(model_path: str, *, tp_size: int = 8) -> PipelineConfig:
    """Full AR → DiT pipeline.

    Both stages live on the same TP=8 plane and reuse the 80B backbone
    weights via the in-process relay. The AR stage projects its terminal
    payload into the DiT input shape (prompt + height/width + AR K/V) via
    `ar_to_dit.project_stage_payload_ar_to_dit`.
    """
    devices = list(range(tp_size))
    ar_stage = StageConfig(
        name=AR_STAGE,
        factory=f"{_PKG}.stages.create_ar_executor",
        factory_args={
            "model_path": model_path,
            "trust_remote_code": True,
        },
        gpu=devices,
        tp_size=tp_size,
        next=DIT_STAGE,
        project_payload={
            DIT_STAGE: f"{_PKG}.ar_to_dit.project_stage_payload_ar_to_dit",
        },
    )
    dit_stage = StageConfig(
        name=DIT_STAGE,
        factory=f"{_PKG}.stages.create_dit_executor",
        factory_args={
            "model_path": model_path,
        },
        gpu=devices,
        tp_size=tp_size,
        terminal=True,
    )
    # Both stages share the same TP=8 GPU set (`devices`); fusion proper
    # (single-process colocation) is gated on schema's `apply_fusion()` which
    # is not yet implemented — for now the SHM relay carries AR K/V between
    # stage workers that happen to live on the same GPUs.
    return PipelineConfig(
        name="hunyuan_image3",
        model_path=model_path,
        entry_stage=AR_STAGE,
        stages=[ar_stage, dit_stage],
    )


def _dit_only_config(model_path: str, *, tp_size: int = 8) -> PipelineConfig:
    """DiT-only Variant — DiT correctness validation."""
    raise NotImplementedError(
        "dit_only Variant requires the DiT pipeline port. "
        "Use Variants['ar_only'] in the meantime."
    )


class HunyuanImage3PipelineConfig:
    """Variants for HunyuanImage-3.0 pipeline configurations.

    Selected from sglang-omni's PIPELINE_CONFIG_REGISTRY via the model name.
    """

    EntryClass: ClassVar[str] = "HunyuanImage3PipelineConfig"

    Variants: ClassVar[dict[str, Any]] = {
        "ar_only": _ar_only_config,
        "default": _default_config,
        "dit_only": _dit_only_config,
    }
