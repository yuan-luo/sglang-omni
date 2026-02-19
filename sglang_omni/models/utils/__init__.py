# SPDX-License-Identifier: Apache-2.0
"""Shared model utilities."""

from sglang_omni.models.utils.hf import instantiate_module, load_hf_config
from sglang_omni.models.utils.common import get_layer_id, add_prefix

__all__ = [
    "instantiate_module",
    "load_hf_config",
    "get_layer_id",
    "add_prefix"
]
