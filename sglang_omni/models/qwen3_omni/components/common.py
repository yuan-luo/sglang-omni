# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni components."""

from __future__ import annotations

from typing import Any

from sglang_omni.utils import load_hf_config


def load_thinker_config(model_id: str) -> Any:
    cfg = load_hf_config(model_id, trust_remote_code=True, local_files_only=True)
    return getattr(cfg, "thinker_config", cfg)
