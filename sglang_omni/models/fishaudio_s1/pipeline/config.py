# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration factory for the FishAudio-S1 TTS pipeline."""

from __future__ import annotations

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.models.fishaudio_s1.pipeline.next_stage import (
    PREPROCESSING_STAGE,
    TTS_ENGINE_STAGE,
    VOCODER_STAGE,
)


def create_tts_pipeline_config(
    *,
    model_id: str = "fishaudio/openaudio-s1-mini",
    tts_device: str = "cuda:0",
    vocoder_device: str = "cuda:0",
    max_new_tokens: int = 2048,
    max_seq_len: int = 4096,
    use_compile: bool = True,
    use_radix_cache: bool = True,
    relay_type: str = "shm",
    fused_stages: list[list[str]] | None = None,
) -> PipelineConfig:
    """Create a 3-stage TTS pipeline config for FishAudio-S1.

    Stages: preprocessing (CPU) -> tts_engine (GPU) -> vocoder (GPU).
    """

    _pkg = "sglang_omni.models.fishaudio_s1.pipeline"

    def _relay(device: str) -> RelayConfig:
        return RelayConfig(type=relay_type, device=device)

    return PipelineConfig(
        name="fishaudio_s1_tts",
        model_path=model_id,
        entry_stage=PREPROCESSING_STAGE,
        fused_stages=fused_stages or [],
        stages=[
            StageConfig(
                name=PREPROCESSING_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_pkg}.stages.create_preprocessing_executor",
                    args={"model_id": model_id},
                ),
                get_next=f"{_pkg}.next_stage.preprocessing_next",
                relay=_relay("cpu"),
            ),
            StageConfig(
                name=TTS_ENGINE_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_pkg}.stages.create_tts_engine_executor",
                    args={
                        "model_id": model_id,
                        "device": tts_device,
                        "max_new_tokens": max_new_tokens,
                        "max_seq_len": max_seq_len,
                        "use_compile": use_compile,
                        "use_radix_cache": use_radix_cache,
                    },
                ),
                get_next=f"{_pkg}.next_stage.tts_engine_next",
                relay=_relay(tts_device),
            ),
            StageConfig(
                name=VOCODER_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_pkg}.stages.create_vocoder_executor",
                    args={
                        "model_id": model_id,
                        "device": vocoder_device,
                    },
                ),
                get_next=f"{_pkg}.next_stage.vocoder_next",
                relay=_relay(vocoder_device),
            ),
        ],
    )
