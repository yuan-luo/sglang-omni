# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AR → DiT projector.

These tests exercise pure-Python logic — ratio table construction, token-id
to ratio-index lookup, text truncation, dimension resolution — and do not
require a CUDA device, a model, or sglang's runtime.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.hunyuan_image3.ar_to_dit import (
    _parse_size_str,
    _ratio_token_id_to_index,
    _truncate_at_cot_end,
    build_ratio_size_table,
    extract_ratio_index,
    project_ar_payload_to_dit,
    resolve_dimensions,
)
from sglang_omni.models.hunyuan_image3.prompt_utils import (
    HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS,
)


# ---------------------------------------------------------------------------
# build_ratio_size_table
# ---------------------------------------------------------------------------
class TestBuildRatioSizeTable:
    def test_size_is_37(self):
        table = build_ratio_size_table(1024)
        assert len(table) == 37

    def test_sorted_ascending_by_aspect(self):
        table = build_ratio_size_table(1024)
        ratios = [h / w for (h, w) in table[:-4]]  # last 4 are named extras, may not sort
        assert ratios == sorted(ratios)

    def test_named_extras_appended(self):
        table = build_ratio_size_table(1024)
        assert (1024, 768) in table
        assert (1280, 720) in table
        assert (768, 1024) in table
        assert (720, 1280) in table

    def test_cached_returns_same_object(self):
        a = build_ratio_size_table(1024)
        b = build_ratio_size_table(1024)
        assert a is b

    def test_different_base_size_different_table(self):
        t1024 = build_ratio_size_table(1024)
        t512 = build_ratio_size_table(512)
        assert t1024 != t512


# ---------------------------------------------------------------------------
# _ratio_token_id_to_index
# ---------------------------------------------------------------------------
class TestRatioTokenIdToIndex:
    def test_has_all_37_entries(self):
        table = _ratio_token_id_to_index()
        assert len(table) == 37

    def test_ratio_0_maps_to_0(self):
        ratio_0 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
        assert _ratio_token_id_to_index()[ratio_0] == 0

    def test_ratio_32_maps_to_32(self):
        ratio_32 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_32>"]
        assert _ratio_token_id_to_index()[ratio_32] == 32

    def test_ratio_33_maps_to_33(self):
        # The 4 named extras are appended in order ratio_33..ratio_36.
        ratio_33 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"]
        assert _ratio_token_id_to_index()[ratio_33] == 33

    def test_ratio_36_maps_to_36(self):
        ratio_36 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"]
        assert _ratio_token_id_to_index()[ratio_36] == 36

    def test_indices_are_dense(self):
        values = sorted(_ratio_token_id_to_index().values())
        assert values == list(range(37))


# ---------------------------------------------------------------------------
# extract_ratio_index
# ---------------------------------------------------------------------------
class TestExtractRatioIndex:
    def test_none_when_no_tokens(self):
        assert extract_ratio_index(None) is None
        assert extract_ratio_index([]) is None

    def test_none_when_no_ratio_in_stream(self):
        ids = [HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<|startoftext|>"], 100, 200, 300]
        assert extract_ratio_index(ids) is None

    def test_finds_trailing_ratio(self):
        # Realistic tail: <answer><boi><img_size_base><img_ratio_4>
        ids = [
            100,
            200,
            HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<answer>"],
            HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<boi>"],
            HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"] + 4,
        ]
        assert extract_ratio_index(ids) == 4

    def test_scans_from_tail(self):
        # If multiple ratios appear (malformed sequence), use the last.
        ratio_0 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
        ids = [ratio_0 + 7, 100, 200, ratio_0 + 13]
        assert extract_ratio_index(ids) == 13

    def test_extras_ratio_33(self):
        assert (
            extract_ratio_index(
                [HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"]]
            )
            == 33
        )


# ---------------------------------------------------------------------------
# resolve_dimensions
# ---------------------------------------------------------------------------
class TestResolveDimensions:
    def test_falls_back_when_no_ratio(self):
        h, w, predicted = resolve_dimensions(
            [], fallback_height=512, fallback_width=768
        )
        assert (h, w) == (512, 768)
        assert predicted is False

    def test_uses_ar_ratio_when_present(self):
        # ratio_36 → last named extra (720, 1280).
        ratio_36 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"]
        h, w, predicted = resolve_dimensions(
            [ratio_36],
            fallback_height=1024,
            fallback_width=1024,
        )
        assert (h, w) == (720, 1280)
        assert predicted is True

    def test_ratio_0_is_widest(self):
        ratio_0 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
        h, w, predicted = resolve_dimensions([ratio_0])
        # Index 0 is widest aspect: (512, 2048).
        assert h == 512
        assert w == 2048
        assert predicted is True


# ---------------------------------------------------------------------------
# _truncate_at_cot_end
# ---------------------------------------------------------------------------
class TestTruncateAtCotEnd:
    def test_truncates_at_recaption_close(self):
        text = "<recaption>detailed prompt</recaption><answer><boi><img_size_base><img_ratio_4>"
        assert (
            _truncate_at_cot_end(text)
            == "<recaption>detailed prompt</recaption>"
        )

    def test_truncates_at_think_close_when_no_recaption(self):
        text = "<think>reasoning</think>tail"
        assert _truncate_at_cot_end(text) == "<think>reasoning</think>"

    def test_prefers_recaption_over_think(self):
        # If both appear, recaption is the canonical end-of-rewrite anchor.
        text = "<think>r</think><recaption>x</recaption>tail"
        result = _truncate_at_cot_end(text)
        assert result.endswith("</recaption>")
        assert "tail" not in result

    def test_passthrough_when_no_marker(self):
        assert _truncate_at_cot_end("plain text") == "plain text"

    def test_empty(self):
        assert _truncate_at_cot_end("") == ""


# ---------------------------------------------------------------------------
# project_ar_payload_to_dit
# ---------------------------------------------------------------------------
class TestProjectArPayloadToDit:
    def test_basic_with_ratio(self):
        ratio_4 = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"] + 4
        result = project_ar_payload_to_dit(
            user_prompt="a cat",
            ar_token_ids=[100, 200, ratio_4],
            ar_generated_text="<recaption>detailed</recaption><answer>...",
        )
        assert result["prompt"] == "a cat"
        assert result["height"] > 0
        assert result["width"] > 0
        assert result["extra"]["ar_predicted_size"] is True
        assert result["extra"]["ratio_index"] == 4
        assert result["extra"]["ar_generated_text"] == "<recaption>detailed</recaption>"

    def test_fallback_when_ar_no_ratio(self):
        result = project_ar_payload_to_dit(
            user_prompt="a cat",
            ar_token_ids=[100, 200],
            ar_generated_text="<recaption>x</recaption>",
            fallback_height=768,
            fallback_width=1024,
        )
        assert result["height"] == 768
        assert result["width"] == 1024
        assert result["extra"]["ar_predicted_size"] is False
        assert result["extra"]["ratio_index"] is None

    def test_empty_ar_text(self):
        result = project_ar_payload_to_dit(
            user_prompt="a cat",
            ar_token_ids=[],
            ar_generated_text="",
        )
        assert result["extra"]["ar_generated_text"] == ""


# ---------------------------------------------------------------------------
# _parse_size_str
# ---------------------------------------------------------------------------
class TestParseSizeStr:
    def test_canonical_wxh(self):
        # "1024x1792" means width=1024, height=1792 in OpenAI convention.
        assert _parse_size_str("1024x1792", default=(1, 1)) == (1792, 1024)

    def test_empty_returns_default(self):
        assert _parse_size_str("", default=(512, 512)) == (512, 512)

    def test_auto_returns_default(self):
        assert _parse_size_str("auto", default=(512, 512)) == (512, 512)

    def test_garbage_returns_default(self):
        assert _parse_size_str("not_a_size", default=(1, 1)) == (1, 1)
        assert _parse_size_str("x1024", default=(1, 1)) == (1, 1)

    def test_case_insensitive(self):
        assert _parse_size_str("768X1024", default=(0, 0)) == (1024, 768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
