# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Qwen3-Omni stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.qwen3_omni.io import PipelineState, TalkerOutput, ThinkerOutput

if TYPE_CHECKING:
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData
    from sglang_omni.engines.omni.runtime.sglang_talker import TalkerARRequestData


def build_encoder_request(
    state: PipelineState, *, stage_name: str
) -> EncoderRequestData:
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(input_dict={"_skip": True, "_result": {}})
    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            input_dict=inputs,
            output_dict=skip_result if isinstance(skip_result, dict) else {},
        )
    cache_key = inputs.get("cache_key")
    return EncoderRequestData(
        input_dict=inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
    )


def apply_encoder_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    if isinstance(result, EncoderRequestData):
        if result.output_dict is not None:
            encoder_out = result.output_dict
        elif result.embeddings is not None:
            encoder_out = result.embeddings
        else:
            encoder_out = {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def build_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }

    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    return ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )


def build_sglang_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    request_id: str | None = None,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData from pipeline state.

    Constructs a SGLang Req with normalized SamplingParams, then wraps it
    in SGLangARRequestData (which inherits ARRequestData).
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    input_ids_list = input_ids.to(dtype=torch.long).tolist()

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }
    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    if "attention_mask" in model_inputs:
        model_inputs.pop("attention_mask", None)

    max_new_tokens = params.get("max_new_tokens", 2048)
    temperature = params.get("temperature", 0.0)

    # Build SGLang SamplingParams and normalize
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    # Build SGLang Req
    rid = request_id or "req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )

    # Build SGLangARRequestData — output_ids points to req.output_ids
    data = SGLangARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    return data


def apply_thinker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    if isinstance(result, ARRequestData):
        output_ids = list(result.output_ids)
        thinker_out: ThinkerOutput = {
            "output_ids": output_ids,
            "step": len(output_ids),
            "is_final": True,
            "extra_model_outputs": dict(result.extra_model_outputs),
        }
    else:
        thinker_out = {
            "output_ids": [],
            "step": 0,
            "is_final": True,
            "extra_model_outputs": {"result": result},
        }

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out


# ---------------------------------------------------------------------------
# Talker helpers
# ---------------------------------------------------------------------------


def build_sglang_talker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    request_id: str | None = None,
) -> "TalkerARRequestData":
    """Build TalkerARRequestData from pipeline state.

    Extracts thinker hidden states from talker_inputs and wraps them
    into a TalkerARRequestData for the talker engine.
    """
    from sglang_omni.engines.omni.runtime.sglang_talker import TalkerARRequestData

    talker_inputs = state.talker_inputs
    if not talker_inputs:
        raise ValueError("talker_inputs missing on PipelineState")

    thinker_embed = talker_inputs.get("thinker_embed")
    thinker_hidden = talker_inputs.get("thinker_hidden")
    is_multimodal_mask = talker_inputs.get("is_multimodal_mask")

    if not isinstance(thinker_embed, torch.Tensor):
        raise TypeError("talker_inputs.thinker_embed must be a torch.Tensor")
    if not isinstance(thinker_hidden, torch.Tensor):
        raise TypeError("talker_inputs.thinker_hidden must be a torch.Tensor")

    max_new_tokens = params.get("talker_max_new_tokens", thinker_embed.shape[-2])

    return TalkerARRequestData(
        thinker_embed=thinker_embed,
        thinker_hidden=thinker_hidden,
        is_multimodal_mask=is_multimodal_mask,
        max_new_tokens=max_new_tokens,
        request_id=request_id or "talker-req-0",
    )


def apply_talker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> TalkerOutput:
    """Store talker output (codec codes) on pipeline state."""
    from sglang_omni.engines.omni.runtime.sglang_talker import TalkerARRequestData

    if isinstance(result, TalkerARRequestData):
        codec_codes = result.codec_codes
        if hasattr(codec_codes, "tolist"):
            codec_codes = codec_codes.tolist()
        talker_out: TalkerOutput = {
            "codec_codes": codec_codes,
            "step": 1,
            "is_final": True,
        }
    elif isinstance(result, dict):
        talker_out = {
            "codec_codes": result.get("codec_codes", []),
            "step": result.get("step", 0),
            "is_final": True,
        }
    else:
        talker_out = {
            "codec_codes": [],
            "step": 0,
            "is_final": True,
        }

    state.talker_out = talker_out
    state.engine_outputs[stage_name] = talker_out
    return talker_out
