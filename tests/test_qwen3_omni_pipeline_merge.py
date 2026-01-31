# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Qwen3-Omni pipeline merge logic."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline.merge import merge_for_thinker
from sglang_omni.models.qwen3_omni.pipeline.next_stage import AUDIO_STAGE, IMAGE_STAGE
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(request_id: str, state: PipelineState) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}, params={}),
        data=state.to_dict(),
    )


def test_merge_for_thinker_prefers_encoder_outputs() -> None:
    request_id = "req-1"
    frontend_state = PipelineState(
        mm_inputs={
            "image": {"image_grid_thw": torch.tensor([[1, 4, 4]])},
            "audio": {"audio_feature_lengths": torch.tensor([128])},
        }
    )
    image_state = PipelineState(
        encoder_outs={
            IMAGE_STAGE: {
                "image_embeds": torch.zeros((4, 8)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        }
    )
    audio_state = PipelineState(
        encoder_outs={
            AUDIO_STAGE: {
                "audio_embeds": torch.zeros((3, 8)),
                "audio_feature_lengths": torch.tensor([64]),
            }
        }
    )

    payloads = {
        "frontend": _payload(request_id, frontend_state),
        IMAGE_STAGE: _payload(request_id, image_state),
        AUDIO_STAGE: _payload(request_id, audio_state),
    }

    merged = merge_for_thinker(payloads)
    state = PipelineState.from_dict(merged.data)
    thinker_inputs = state.thinker_inputs.get("model_inputs", {})

    assert torch.equal(thinker_inputs["image_grid_thw"], torch.tensor([[1, 2, 2]]))
    assert torch.equal(thinker_inputs["audio_feature_lengths"], torch.tensor([64]))
    assert IMAGE_STAGE in state.encoder_outs
    assert AUDIO_STAGE in state.encoder_outs

    # Frontend inputs should be pruned to lightweight fields.
    assert state.mm_inputs["image"].get("image_grid_thw") is not None
    assert state.mm_inputs["audio"].get("audio_feature_lengths") is not None
    assert len(state.mm_inputs["image"]) == 1
    assert len(state.mm_inputs["audio"]) == 1


def test_merge_for_thinker_falls_back_to_mm_inputs() -> None:
    request_id = "req-2"
    frontend_state = PipelineState(
        mm_inputs={
            "image": {"image_grid_thw": torch.tensor([[1, 3, 3]])},
            "audio": {
                "audio_feature_lengths": torch.tensor([96]),
                "feature_attention_mask": torch.ones((1, 10), dtype=torch.long),
            },
        }
    )
    audio_state = PipelineState(
        encoder_outs={
            AUDIO_STAGE: {
                "audio_embeds": torch.zeros((3, 8)),
            }
        }
    )

    payloads = {
        "frontend": _payload(request_id, frontend_state),
        AUDIO_STAGE: _payload(request_id, audio_state),
    }

    merged = merge_for_thinker(payloads)
    state = PipelineState.from_dict(merged.data)
    thinker_inputs = state.thinker_inputs.get("model_inputs", {})

    assert torch.equal(thinker_inputs["audio_feature_lengths"], torch.tensor([96]))
    assert torch.equal(
        thinker_inputs["feature_attention_mask"],
        torch.ones((1, 10), dtype=torch.long),
    )
