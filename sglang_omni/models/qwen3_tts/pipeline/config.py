# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration factory for Qwen3-TTS."""

from __future__ import annotations

from sglang_omni.config import (
    ExecutorConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.models.qwen3_tts.pipeline.next_stage import (
    CODEC_DECODER_STAGE,
    CODE_PREDICTOR_STAGE,
    FRONTEND_STAGE,
    TALKER_STAGE,
)

_STAGES = "sglang_omni.models.qwen3_tts.pipeline.stages"
_NEXT = "sglang_omni.models.qwen3_tts.pipeline.next_stage"


def create_tts_pipeline_config(
    *,
    model_id: str,
    name: str = "qwen3_tts",
    frontend_device: str = "cpu",
    talker_device: str = "cuda:0",
    code_predictor_device: str = "cuda:0",
    codec_decoder_device: str = "cuda:0",
    dtype: str | None = None,
    relay_type: str = "shm",
    num_code_predictor_workers: int = 1,
) -> PipelineConfig:
    """Build a :class:`PipelineConfig` for the Qwen3-TTS 4-stage pipeline.

    Topology::

        frontend → talker ⇄ code_predictor → codec_decoder → END
    """

    def _relay(device: str) -> RelayConfig:
        return RelayConfig(type=relay_type, device=device)

    return PipelineConfig(
        name=name,
        entry_stage=FRONTEND_STAGE,
        stages=[
            StageConfig(
                name=FRONTEND_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_STAGES}.create_frontend_executor",
                    args={"model_id": model_id},
                ),
                get_next=f"{_NEXT}.frontend_next",
                relay=_relay(frontend_device),
            ),
            StageConfig(
                name=TALKER_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_STAGES}.create_talker_executor",
                    args={
                        "model_id": model_id,
                        "device": talker_device,
                        "dtype": dtype,
                    },
                ),
                get_next=f"{_NEXT}.talker_next",
                relay=_relay(talker_device),
            ),
            StageConfig(
                name=CODE_PREDICTOR_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_STAGES}.create_code_predictor_executor",
                    args={
                        "model_id": model_id,
                        "device": code_predictor_device,
                        "dtype": dtype,
                    },
                ),
                get_next=f"{_NEXT}.code_predictor_next",
                relay=_relay(code_predictor_device),
                num_workers=num_code_predictor_workers,
            ),
            StageConfig(
                name=CODEC_DECODER_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_STAGES}.create_codec_decoder_executor",
                    args={
                        "model_id": model_id,
                        "device": codec_decoder_device,
                        "dtype": dtype,
                    },
                ),
                get_next=f"{_NEXT}.codec_decoder_next",
                relay=_relay(codec_decoder_device),
            ),
        ],
    )
