# SPDX-License-Identifier: Apache-2.0
"""AR → DiT payload projector for HunyuanImage-3.

Translates the AR stage's output (generated token IDs + detokenized text +
the AR K/V exported via KVExporter) into a DiT stage input.

Three pieces of structured data are extracted from the AR output:

  1. `ratio_index` (int 0..36) — from the last `<img_ratio_*>` token in the
     generated stream. Maps to a (height, width) bucket via
     `build_ratio_size_table(base_size=1024)`.

  2. `cot_text` (str) — the AR-generated text truncated at the first
     `</recaption>` or `</think>` so the trailing
     `<answer><boi><img_size_*><img_ratio_*>` control sequence doesn't leak
     into DiT's text-conditioning embedding.

  3. Per-layer K/V (Dict[layer_id, PerLayerKV]) — installed into the DiT
     pipeline's `ImageKVCacheManager` at step 0.

The projector is the sglang-omni equivalent of vllm-omni's
`stage_input_processors/hunyuan_image3.py` ar2diffusion handler. It runs
on the AR stage's output side; the DiT stage's request_builder consumes
its output dict.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from sglang_omni.models.hunyuan_image3.prompt_utils import (
    HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS,
)

# Extra resolutions appended to the base resolution table when their aspect
# ratio is not already present. These are the canonical "named" buckets
# (e.g. 768×1024 for 3:4 portrait).
_EXTRA_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (1024, 768),
    (1280, 720),
    (768, 1024),
    (720, 1280),
)


def _truncate_at_cot_end(generated_text: str) -> str:
    """Truncate AR output at first `</recaption>` (fallback `</think>`).

    The trailing `<answer><boi><img_size_*><img_ratio_*>` is consumed via
    token-id extraction; it must not leak into DiT's text conditioning.
    """
    for marker in ("</recaption>", "</think>"):
        idx = generated_text.find(marker)
        if idx != -1:
            return generated_text[: idx + len(marker)]
    return generated_text


def _resolutions_by_step(base_size: int, align: int = 1) -> list[tuple[int, int]]:
    """Build the canonical resolution group for a given base_size.

    Mirrors HunyuanImage-3's ResolutionGroup: starts at the square
    (base, base), then walks outward in both directions (taller-narrower
    and wider-shorter) in steps of base/16 until hitting the (base/2, 2*base)
    extremes. Returned list is sorted by aspect ratio ascending.
    """
    step = base_size // 16
    min_height = base_size // 2
    min_width = base_size // 2
    max_height = base_size * 2
    max_width = base_size * 2

    out: list[tuple[int, int]] = [(base_size, base_size)]

    # Taller / narrower direction.
    h, w = base_size, base_size
    while True:
        if h >= max_height and w <= min_width:
            break
        h = min(h + step, max_height)
        w = max(w - step, min_width)
        out.append((h // align * align, w // align * align))

    # Shorter / wider direction.
    h, w = base_size, base_size
    while True:
        if h <= min_height and w >= max_width:
            break
        h = max(h - step, min_height)
        w = min(w + step, max_width)
        out.append((h // align * align, w // align * align))

    out.sort(key=lambda hw: hw[0] / hw[1])
    return out


@lru_cache(maxsize=4)
def build_ratio_size_table(base_size: int) -> list[tuple[int, int]]:
    """Return `[(height, width)]` indexed by ratio_index.

    The first 33 entries (indices 0..32) come from the canonical aspect-ratio
    sweep; the trailing 4 (indices 33..36) are named extras. Cached because
    the table is constant per base_size.
    """
    resolutions = _resolutions_by_step(base_size)
    existing_ratios = {h / w for (h, w) in resolutions}
    for h, w in _EXTRA_RESOLUTIONS:
        if h / w not in existing_ratios:
            resolutions.append((h, w))
            existing_ratios.add(h / w)
    return resolutions


@lru_cache(maxsize=1)
def _ratio_token_id_to_index() -> dict[int, int]:
    """Map `<img_ratio_X>` token IDs → ratio_index 0..36."""
    ratio_0 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
    ratio_32 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_32>"]
    ratio_33 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"]
    ratio_36 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"]

    table: dict[int, int] = {}
    for i in range(ratio_32 - ratio_0 + 1):
        table[ratio_0 + i] = i
    offset = ratio_32 - ratio_0 + 1
    for j in range(ratio_36 - ratio_33 + 1):
        table[ratio_33 + j] = offset + j
    return table


def extract_ratio_index(generated_token_ids) -> Optional[int]:
    """Resolve the AR-predicted ratio_index from the generated token stream.

    AR's natural trajectory ends with `</recaption><answer><boi>
    <img_size_*><img_ratio_X>`, where `<img_ratio_X>` is the stop token.
    We scan the tail for the first id that maps to a ratio so we tolerate
    the model wandering before settling.

    Returns None if no ratio token was emitted (e.g., AR ran to
    `max_tokens` ceiling — this is the `bot_task=vanilla` failure mode).
    """
    if generated_token_ids is None:
        return None
    lookup = _ratio_token_id_to_index()
    # Iterate tail-first for efficiency (ratio is the very last emitted token
    # on a well-formed AR trajectory).
    for tid in reversed(list(generated_token_ids)):
        idx = lookup.get(int(tid))
        if idx is not None:
            return idx
    return None


def resolve_dimensions(
    generated_token_ids,
    *,
    base_size: int = 1024,
    fallback_height: int = 1024,
    fallback_width: int = 1024,
) -> tuple[int, int, bool]:
    """Look up (height, width) from the AR-predicted ratio token.

    Returns:
      (height, width, ar_predicted) — `ar_predicted=True` when the ratio
      came from the AR output, `False` when the fallbacks were used (AR did
      not emit a ratio token, or the index was out of range).
    """
    ratio_idx = extract_ratio_index(generated_token_ids)
    if ratio_idx is None:
        return fallback_height, fallback_width, False
    table = build_ratio_size_table(base_size)
    if 0 <= ratio_idx < len(table):
        height, width = table[ratio_idx]
        return height, width, True
    # Out-of-range ratio_idx: fall back; should not happen if the stop set
    # is the full <img_ratio_0..36> range, but defend against future ratio
    # additions to the tokenizer.
    return fallback_height, fallback_width, False


def project_ar_payload_to_dit(
    *,
    user_prompt: str,
    ar_token_ids: list[int],
    ar_generated_text: str,
    fallback_height: int = 1024,
    fallback_width: int = 1024,
    base_size: int = 1024,
    image_base_size: Optional[int] = None,
) -> dict[str, Any]:
    """Produce DiT input dict from AR output fields.

    Args:
      user_prompt: Original user-supplied prompt text. Forwarded for DiT's
        prompt builder (DiT consumes both the original prompt and the AR-
        rewritten text — the latter as enriched conditioning).
      ar_token_ids: AR's full generated token stream (including the
        terminating `<img_ratio_*>` token). Source of truth for image
        dimensions.
      ar_generated_text: Detokenized AR output. Will be truncated at
        `</recaption>` / `</think>`.
      fallback_height, fallback_width: Used when AR didn't emit a ratio
        token (e.g., `bot_task=vanilla` on a short prompt that ran to
        max_tokens).
      base_size: Resolution-table base used by the ratio token lookup.
      image_base_size: Optional override of base_size for this request
        (matches vllm-omni's `image_base_size` request field). Defaults to
        base_size.

    Returns:
      Dict consumable by the DiT pipeline's request_builder:
        - prompt: original user prompt (str)
        - height, width: int — AR-inferred or fallback
        - extra:
            - ar_generated_text: truncated cot/recaption text
            - ar_predicted_size: bool — did AR emit a ratio token?
            - ratio_index: int | None
    """
    effective_base = image_base_size if image_base_size is not None else base_size

    cot_text = _truncate_at_cot_end(ar_generated_text or "")
    height, width, ar_predicted = resolve_dimensions(
        ar_token_ids,
        base_size=effective_base,
        fallback_height=fallback_height,
        fallback_width=fallback_width,
    )
    ratio_index = extract_ratio_index(ar_token_ids)

    return {
        "prompt": user_prompt,
        "height": height,
        "width": width,
        "extra": {
            "ar_generated_text": cot_text,
            "ar_predicted_size": ar_predicted,
            "ratio_index": ratio_index,
        },
    }


def project_stage_payload_ar_to_dit(payload):
    """StagePayload-level AR→DiT projector for the pipeline coordinator.

    Wired into `StageConfig.project_payload` on the AR stage. Takes the AR
    stage's terminal `StagePayload` (populated by the AR result_adapter —
    keys: `ar_token_ids`, `ar_generated_text`) and returns a new
    `StagePayload` with DiT-shape fields.

    K/V data is handled separately via the SHM/NCCL relay (see KVExporter
    + pack_per_layer_kv_for_relay).
    """
    from sglang_omni.proto import StagePayload

    data = dict(payload.data) if payload.data else {}
    req = payload.request
    user_prompt = ""
    if req is not None:
        rp = getattr(req, "prompt", None)
        if isinstance(rp, str):
            user_prompt = rp
    if not user_prompt:
        user_prompt = str(data.get("prompt") or "")

    params = (req.params or {}) if req is not None else {}
    base_size = int(params.get("image_base_size", 1024))
    fallback_size_str = params.get("size") or ""
    fallback_h, fallback_w = _parse_size_str(fallback_size_str, default=(1024, 1024))

    ar_token_ids = list(data.get("ar_token_ids") or [])
    ar_generated_text = str(data.get("ar_generated_text") or "")

    dit_input = project_ar_payload_to_dit(
        user_prompt=user_prompt,
        ar_token_ids=ar_token_ids,
        ar_generated_text=ar_generated_text,
        fallback_height=fallback_h,
        fallback_width=fallback_w,
        base_size=base_size,
    )

    out_data = dict(data)
    out_data.update(dit_input)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=out_data,
    )


def _parse_size_str(size_str: str, *, default: tuple[int, int]) -> tuple[int, int]:
    """Parse `"WxH"` / `"HxW"` / `"auto"` / `""` into (height, width).

    OpenAI's API uses `WxH` (e.g. `"1024x1792"`). vllm-omni's pipeline
    expects (height, width). We always interpret the first number as width
    to match the OpenAI convention, then swap.
    """
    if not size_str or size_str == "auto":
        return default
    try:
        a, b = size_str.lower().split("x")
        width = int(a)
        height = int(b)
        return height, width
    except (ValueError, AttributeError):
        return default
