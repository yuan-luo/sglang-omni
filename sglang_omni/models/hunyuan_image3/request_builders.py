# SPDX-License-Identifier: Apache-2.0
"""StagePayload ↔ SGLang Req adapters for the HunyuanImage-3 AR stage.

`request_builder`: takes a `StagePayload` carrying `(user_prompt, bot_task,
sys_type, sampling_params)` and produces an SGLang `Req` ready for the
scheduler. Uses `build_prompt_tokens()` to apply the chat-template + trigger
tag and `resolve_stop_token_ids()` for the t2i `<img_ratio_*>` stop set.

`result_adapter`: pulls the AR's generated token-id sequence + detokenized
text out of the scheduler output and stuffs it back into a `StagePayload`
under the well-known keys `ar_token_ids` and `ar_generated_text` (also
exposed as `cot_output` in the OpenAI response). The `<img_ratio_*>` tail
token is preserved on `ar_token_ids` so a downstream DiT stage can extract
the target image size; `ar_generated_text` is truncated at `</recaption>` /
`</think>` for clean text output.
"""

from __future__ import annotations

from typing import Any


def make_ar_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
):
    """Build the (request_builder, result_adapter) pair for the AR stage.

    Args:
      tokenizer: A loaded HF AutoTokenizer (trust_remote_code=True for
        HunyuanImage-3's custom tokenizer).
      vocab_size: AR vocabulary size; used to validate stop tokens.

    Returns:
      Tuple (request_builder, result_adapter) for OmniScheduler.
    """
    # Local import — heavy stuff is deferred so config import stays cheap.
    from sglang_omni.models.hunyuan_image3.prompt_utils import (
        build_prompt_tokens,
        resolve_stop_token_ids,
    )

    def request_builder(payload):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams as SGLSamplingParams

        from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

        # Pull request fields off the payload.
        req_data = payload.request
        params = (req_data.params or {}) if req_data is not None else {}
        prompt_text = _extract_prompt_text(payload)
        bot_task = params.get("bot_task")  # None / vanilla / recaption / think / think_recaption
        sys_type = params.get("sys_type")  # optional; defaults derived from bot_task
        custom_system_prompt = params.get("system_prompt")
        # HunyuanImage-3 AR is the t2i image-generation backbone. i2t/it2i
        # modalities (image input) will need a separate path that injects
        # image embeddings — see ModelRunner subclass plan.
        task = params.get("task", "t2i")
        num_images = int(params.get("num_images", 1))

        # Apply the chat template + trigger tag.
        prompt_result = build_prompt_tokens(
            prompt_text,
            tokenizer=tokenizer,
            task=task,
            bot_task=bot_task,
            sys_type=sys_type,
            custom_system_prompt=custom_system_prompt,
            num_images=num_images if task in ("i2t", "it2i") else 1,
        )
        input_ids = list(prompt_result.token_ids)

        # Stop tokens: t2i / it2i stop on any `<img_ratio_*>` ratio token;
        # text-output tasks stop on `<answer>`. resolve_stop_token_ids does
        # the right thing per task.
        stop_token_ids = resolve_stop_token_ids(
            task=task,
            bot_task=bot_task,
            tokenizer=tokenizer,
        )

        # Sampling. HunyuanImage-3's deterministic deploy uses temperature=0;
        # we forward whatever the caller passed and fall back to the recipe
        # defaults documented in the deploy configs.
        max_new_tokens = int(params.get("max_tokens", 8192))
        temperature = float(params.get("temperature", 0.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        seed = params.get("seed", None)

        sgl_sampling = SGLSamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,  # keep `<img_ratio_*>` visible to the bridge
        )
        if seed is not None:
            try:
                sgl_sampling.seed = int(seed)
            except (TypeError, ValueError):
                pass

        req = Req(
            rid=req_data.id if req_data is not None else None,
            origin_input_ids=input_ids,
            sampling_params=sgl_sampling,
        )

        return SGLangARRequestData(
            req=req,
            stage_payload=payload,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def result_adapter(req_data, finished_req):
        """Translate finished SGLang output back into a StagePayload.

        Output StagePayload data fields:
          - `ar_token_ids`: full generated token-id list (includes the
            terminating `<img_ratio_*>` token so the DiT bridge can extract
            target image size).
          - `ar_generated_text`: detokenized text, truncated at the first
            `</recaption>` (fallback `</think>`); empty for `bot_task=vanilla`.
          - `finish_reason`: scheduler's finish reason.
        """
        gen_token_ids: list[int] = list(getattr(finished_req, "output_ids", []) or [])
        gen_text = ""
        try:
            gen_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
        except Exception:  # pragma: no cover — defensive
            gen_text = ""

        truncated_text = _truncate_at_cot_end(gen_text)
        finish_reason = getattr(finished_req, "finished_reason", None)

        payload = req_data.stage_payload
        data = dict(payload.data) if payload is not None and payload.data else {}
        data["ar_token_ids"] = gen_token_ids
        data["ar_generated_text"] = truncated_text
        if finish_reason is not None:
            data["finish_reason"] = finish_reason

        # Return a StagePayload with the result. The caller (OmniScheduler)
        # routes this to the next stage (or terminal output).
        from sglang_omni.proto import StagePayload

        return StagePayload(
            request_id=payload.request_id if payload is not None else "",
            request=payload.request if payload is not None else None,
            data=data,
        )

    return request_builder, result_adapter


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _extract_prompt_text(payload) -> str:
    """Pull the user-visible prompt string out of the payload.

    Tries `payload.data["prompt"]` first, then `payload.request.prompt`.
    """
    if payload is None:
        return ""
    data = payload.data or {}
    if isinstance(data, dict):
        p = data.get("prompt")
        if isinstance(p, str):
            return p
        if isinstance(p, dict):
            inner = p.get("prompt") or p.get("text")
            if isinstance(inner, str):
                return inner
    req = payload.request
    if req is not None:
        rp = getattr(req, "prompt", None)
        if isinstance(rp, str):
            return rp
    return ""


def _truncate_at_cot_end(generated_text: str) -> str:
    """Truncate AR output at the first `</recaption>` (fallback `</think>`).

    Mirrors the upstream behavior: the trailing
    `<answer><boi><img_size_*><img_ratio_*>` is consumed by the bridge via
    token-id extraction; it must not leak into a text response or DiT
    prompt builder.
    """
    for marker in ("</recaption>", "</think>"):
        idx = generated_text.find(marker)
        if idx != -1:
            return generated_text[: idx + len(marker)]
    return generated_text
