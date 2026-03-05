# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro (FishQwen3OmniForCausalLM) model support for sglang-omni.

The S2-Pro model uses ``FishQwen3OmniForCausalLM`` (a HuggingFace-style
model loaded via ``AutoModel.from_pretrained``) with the ``qwen3.py``
inference module.  This is a fundamentally different architecture from
S1-Mini's ``DualARTransformer`` — S2-Pro has:

- Built-in audio decoder for codebook generation
- Repetition Aware Sampling (RAS)
- Constrained semantic decoding
- Qwen3 chat-format prompts via ``Conversation`` class
- HuggingFace ``PreTrainedTokenizerFast`` tokenizer
"""

from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
    Reference,
    S2ProTokenizerAdapter,
)

__all__ = [
    "create_tts_pipeline_config",
    "S2ProState",
    "S2ProTokenizerAdapter",
    "Reference",
]


def __getattr__(name: str):
    if name == "create_tts_pipeline_config":
        from .pipeline.config import create_tts_pipeline_config

        return create_tts_pipeline_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
