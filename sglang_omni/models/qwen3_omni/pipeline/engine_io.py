# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Qwen3-Omni stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.qwen3_omni.io import PipelineState, TalkerOutput, ThinkerOutput

if TYPE_CHECKING:
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData


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


@dataclass
class TalkerRequestData:
    """Per-request data for the talker codec generation stage."""

    thinker_embed: torch.Tensor
    thinker_hidden: torch.Tensor
    is_multimodal_mask: torch.Tensor | None = None
    output_dict: dict[str, Any] | None = None


def build_talker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
) -> TalkerRequestData:
    """Build TalkerRequestData from pipeline state."""

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

    return TalkerRequestData(
        thinker_embed=thinker_embed,
        thinker_hidden=thinker_hidden,
        is_multimodal_mask=is_multimodal_mask,
    )


def build_sglang_talker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    codec_vocab_size: int = 3072,
    codec_eos_token_id: int = 2150,
    request_id: str | None = None,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData for the Talker stage.

    Creates a SGLang Req with dummy input_ids (for KV cache allocation)
    and stores thinker embeddings in extra_model_outputs for prefill injection.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

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

    # Dummy input_ids for KV cache slot allocation during prefill.
    # The actual embeddings are injected by TalkerSGLangModelRunner.
    seq_len = thinker_embed.shape[0]
    input_ids_list = [0] * seq_len

    max_new_tokens = params.get("max_new_tokens", 4096)
    temperature = params.get("temperature", 0.9)
    top_k = params.get("top_k", 50)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_ids=[codec_eos_token_id],
    )
    sampling_params.normalize(tokenizer=None)
    sampling_params.verify(codec_vocab_size)

    rid = request_id or "talker-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=codec_vocab_size,
    )

    data = SGLangARRequestData(
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )

    # Store thinker embeddings for prefill injection by TalkerSGLangModelRunner
    data.extra_model_outputs["thinker_embed"] = thinker_embed
    data.extra_model_outputs["thinker_hidden"] = thinker_hidden
    if is_multimodal_mask is not None:
        data.extra_model_outputs["is_multimodal_mask"] = is_multimodal_mask

    return data


def apply_talker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> TalkerOutput:
    """Store talker output (codec codes) on pipeline state."""
    from sglang_omni.engines.omni.runtime.ar import ARRequestData

    if isinstance(result, TalkerRequestData):
        codec_dict = result.output_dict or {}
        codec_codes = codec_dict.get("codec_codes")
        if hasattr(codec_codes, "tolist"):
            codec_codes = codec_codes.tolist()
        talker_out: TalkerOutput = {
            "codec_codes": codec_codes,
            "step": 1,
            "is_final": True,
        }
    elif isinstance(result, ARRequestData):
        # SGLang AR talker: assemble codec codes from per-step code predictor output
        codec_codes = _assemble_codec_codes_from_ar_result(result)
        talker_out = {
            "codec_codes": codec_codes,
            "step": len(result.output_ids),
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


def _assemble_codec_codes_from_ar_result(result: Any) -> list:
    """Assemble 16-layer codec codes from SGLang AR talker output.

    The AR talker produces:
    - result.output_ids: list of layer-0 codec token IDs (one per decode step)
    - result.extra_model_outputs["codec_steps"]: list of per-step dicts with
      "codec_codes" tensor [num_code_groups, 1] from the code predictor

    Returns codec_codes as [num_code_groups][num_steps] nested list.
    """
    output_ids = list(result.output_ids)
    codec_steps = result.extra_model_outputs.get("codec_steps", [])
    num_steps = len(output_ids)

    if num_steps == 0:
        return []

    # Determine num_code_groups from first step's codec_codes
    if codec_steps:
        first_codes = codec_steps[0].get("codec_codes")
        if first_codes is not None and hasattr(first_codes, "shape"):
            num_groups = first_codes.shape[0]
        else:
            num_groups = 16
    else:
        num_groups = 16

    # Build [num_groups][num_steps] structure
    all_codes: list[list[int]] = [[] for _ in range(num_groups)]

    for step_idx in range(num_steps):
        # Layer 0: from output_ids
        all_codes[0].append(output_ids[step_idx])

        # Layers 1..N-1: from code predictor
        if step_idx < len(codec_steps):
            step_data = codec_steps[step_idx]
            step_codes = step_data.get("codec_codes")
            if step_codes is not None and hasattr(step_codes, "tolist"):
                codes_list = step_codes.tolist()
                # step_codes shape is [num_groups, 1] — each group has 1 code
                for g in range(1, num_groups):
                    if g < len(codes_list):
                        val = codes_list[g]
                        all_codes[g].append(val[0] if isinstance(val, list) else val)
                    else:
                        all_codes[g].append(0)
            else:
                for g in range(1, num_groups):
                    all_codes[g].append(0)
        else:
            for g in range(1, num_groups):
                all_codes[g].append(0)

    return all_codes
