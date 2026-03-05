# SPDX-License-Identifier: Apache-2.0
"""SGLang model registry entry for the Talker model.

This module is discovered by SGLang's import_model_classes() via
SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_omni.models". It registers
Qwen3OmniTalkerForCausalLM so that ModelRunner can load it when the
architecture is set to "Qwen3OmniTalkerForCausalLM".
"""

from sglang_omni.models.qwen3_omni.talker import Qwen3OmniTalkerForCausalLM

EntryClass = [Qwen3OmniTalkerForCausalLM]
