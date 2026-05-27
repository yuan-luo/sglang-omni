# SPDX-License-Identifier: Apache-2.0
"""Lazy registration helpers for HunyuanImage-3.

Two registrations:
  1. HF AutoConfig — Tencent's config uses `model_type='hunyuan_image_3_moe'`;
     the HF config class is loaded via `trust_remote_code=True` from the
     model dir's `configuration_hunyuan_image_3.py`. No AutoConfig.register
     needed (HF's trust_remote_code does it automatically); the function
     below is kept as a no-op hook in case the config class is vendored later.
  2. SGLang ModelRegistry — alias both the canonical class name
     (`HunyuanImage3ForConditionalGeneration`) and the HF architectures-field
     name (`HunyuanImage3ForCausalMM`) to our class. Either name will load
     this model.
"""

from __future__ import annotations

_hunyuan_image3_registered = False


def register_hunyuan_image3_model_registry() -> None:
    """Register HunyuanImage-3 into sglang's ModelRegistry.

    Idempotent. Called from `_register_omni_model` in `SGLModelRunner`.
    """
    global _hunyuan_image3_registered
    if _hunyuan_image3_registered:
        return

    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.models.hunyuan_image3.ar_model import (
        HunyuanImage3ForConditionalGeneration,
    )

    # HF config.architectures = ["HunyuanImage3ForCausalMM"]. We also expose
    # `HunyuanImage3ForConditionalGeneration` as an alias for explicit loads.
    ModelRegistry.models["HunyuanImage3ForCausalMM"] = (
        HunyuanImage3ForConditionalGeneration
    )
    ModelRegistry.models["HunyuanImage3ForConditionalGeneration"] = (
        HunyuanImage3ForConditionalGeneration
    )

    _hunyuan_image3_registered = True


def register_hunyuan_image3_hf_config() -> None:
    """Register HF AutoConfig for HunyuanImage-3 if needed.

    HF's `trust_remote_code=True` flow already imports
    `configuration_hunyuan_image_3.HunyuanImage3Config` from the model dir's
    `auto_map`, so no explicit AutoConfig.register is required by default.
    This is kept as a no-op hook in case the config class is vendored later.
    """
    return
