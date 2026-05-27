# SPDX-License-Identifier: Apache-2.0
"""Unit tests for prompt_utils.

Pure-Python — no tokenizer or model required. `build_prompt_tokens` is
exercised against a deterministic fake tokenizer that maps strings to a
length-based id stream and resolves special tokens via the static
HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS map.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.hunyuan_image3.prompt_utils import (
    HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS,
    MAX_IMAGES_PER_REQUEST,
    available_bot_tasks,
    available_tasks,
    build_prompt,
    build_prompt_tokens,
    resolve_stop_token_ids,
    resolve_sys_type,
)


# ---------------------------------------------------------------------------
# Fake tokenizer
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """Deterministic stand-in for HunyuanImage-3's HF tokenizer.

    - `convert_tokens_to_ids` returns the canonical id for known specials
      (so segment boundaries are exactly checkable).
    - `encode` returns one synthetic id per character in the input,
      offsetting into a reserved range so synthetic ids never collide
      with special ids.
    """

    SYNTHETIC_OFFSET = 1_000  # well below first special at 127957

    def convert_tokens_to_ids(self, token: str) -> int:
        return HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS[token]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        # 1 id per char. add_special_tokens is ignored — the prompt builder
        # always passes False, which is exactly what we want here.
        return [self.SYNTHETIC_OFFSET + ord(c) for c in text]


# ---------------------------------------------------------------------------
# available_*
# ---------------------------------------------------------------------------
class TestAvailability:
    def test_available_tasks_complete(self):
        assert set(available_tasks()) == {"t2t", "i2t", "it2i", "t2i"}

    def test_available_bot_tasks_complete(self):
        bots = available_bot_tasks()
        assert bots[0] is None
        assert set(bots) == {None, "think", "recaption", "think_recaption", "vanilla"}


# ---------------------------------------------------------------------------
# resolve_sys_type
# ---------------------------------------------------------------------------
class TestResolveSysType:
    def test_default(self):
        assert resolve_sys_type(None) == "en_unified"

    def test_think(self):
        assert resolve_sys_type("think") == "en_unified"

    def test_recaption(self):
        assert resolve_sys_type("recaption") == "en_unified"

    def test_think_recaption(self):
        assert resolve_sys_type("think_recaption") == "en_think_recaption"

    def test_vanilla(self):
        assert resolve_sys_type("vanilla") == "en_vanilla"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown bot_task"):
            resolve_sys_type("garbage")


# ---------------------------------------------------------------------------
# resolve_stop_token_ids
# ---------------------------------------------------------------------------
class TestResolveStopTokenIds:
    def test_t2i_uses_full_ratio_range(self):
        stops = resolve_stop_token_ids(task="t2i", bot_task=None)
        # Main range <img_ratio_0..32> = 128044..128076 (33 entries).
        assert HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"] in stops
        assert HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_32>"] in stops
        # Other slices <img_ratio_33..36> = 130103..130106 (4 entries).
        assert HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"] in stops
        assert HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"] in stops
        assert len(stops) == 37

    def test_it2i_uses_ratio_range(self):
        stops = resolve_stop_token_ids(task="it2i", bot_task="think")
        assert len(stops) == 37

    def test_t2t_stops_on_answer(self):
        stops = resolve_stop_token_ids(task="t2t", bot_task=None)
        assert stops == [HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<answer>"]]

    def test_i2t_stops_on_answer(self):
        stops = resolve_stop_token_ids(task="i2t", bot_task=None)
        assert stops == [HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<answer>"]]

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            resolve_stop_token_ids(task="garbage", bot_task=None)


# ---------------------------------------------------------------------------
# build_prompt — pure string form
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    def test_t2i_no_bot_task(self):
        s = build_prompt("a cat", task="t2i", bot_task=None)
        assert s.startswith("<|startoftext|>")
        assert "User: a cat" in s
        assert "Assistant: " in s
        # No trigger tag for plain unified mode.
        assert not s.rstrip().endswith("<think>")
        assert not s.rstrip().endswith("<recaption>")

    def test_t2i_recaption_appends_trigger_tag(self):
        s = build_prompt("a cat", task="t2i", bot_task="recaption")
        assert s.endswith("<recaption>")

    def test_t2i_think_appends_trigger_tag(self):
        s = build_prompt("a cat", task="t2i", bot_task="think")
        assert s.endswith("<think>")

    def test_t2i_think_recaption_uses_think_trigger(self):
        s = build_prompt("a cat", task="t2i", bot_task="think_recaption")
        # think_recaption uses en_think_recaption system prompt + <think> trigger.
        assert s.endswith("<think>")

    def test_it2i_emits_img_placeholders(self):
        s = build_prompt("describe", task="it2i", bot_task=None, num_images=2)
        assert "User: <img><img>describe" in s

    def test_t2i_does_not_emit_img(self):
        s = build_prompt("a cat", task="t2i", bot_task=None)
        assert "<img>" not in s

    def test_vanilla_skips_chat_template(self):
        s = build_prompt("a cat", task="t2i", bot_task="vanilla")
        # Vanilla pretrain template: <|startoftext|>{sys}{user} — no User:/Assistant:
        assert "User:" not in s
        assert "Assistant:" not in s
        assert s.endswith("a cat")

    def test_vanilla_with_non_t2i_raises(self):
        with pytest.raises(ValueError, match="vanilla.*only valid"):
            build_prompt("x", task="i2t", bot_task="vanilla")

    def test_unknown_bot_task_raises(self):
        with pytest.raises(ValueError, match="Unknown bot_task"):
            build_prompt("x", task="t2i", bot_task="garbage")

    def test_num_images_out_of_range_raises(self):
        with pytest.raises(ValueError, match="num_images must be"):
            build_prompt("x", task="it2i", bot_task=None, num_images=0)
        with pytest.raises(ValueError, match="num_images must be"):
            build_prompt(
                "x",
                task="it2i",
                bot_task=None,
                num_images=MAX_IMAGES_PER_REQUEST + 1,
            )


# ---------------------------------------------------------------------------
# build_prompt_tokens — segmented tokenization with fake tokenizer
# ---------------------------------------------------------------------------
class TestBuildPromptTokens:
    def test_starts_with_bos(self):
        tok = _FakeTokenizer()
        result = build_prompt_tokens("a", tok, task="t2i", bot_task=None)
        assert result.token_ids[0] == HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<|startoftext|>"]

    def test_trailing_trigger_tag(self):
        tok = _FakeTokenizer()
        result = build_prompt_tokens("a", tok, task="t2i", bot_task="recaption")
        assert result.token_ids[-1] == HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<recaption>"]

    def test_no_trigger_tag_when_plain(self):
        tok = _FakeTokenizer()
        result = build_prompt_tokens("a", tok, task="t2i", bot_task=None)
        # Last token should be a synthetic char id from " ", not a special.
        assert result.token_ids[-1] < HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<|endoftext|>"]

    def test_it2i_embeds_img_ids(self):
        tok = _FakeTokenizer()
        img_id = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img>"]
        result = build_prompt_tokens(
            "a", tok, task="it2i", bot_task=None, num_images=3
        )
        # Three <img> ids should appear, consecutively.
        n_img = sum(1 for t in result.token_ids if t == img_id)
        assert n_img == 3

    def test_system_prompt_type_reported(self):
        tok = _FakeTokenizer()
        result = build_prompt_tokens(
            "a", tok, task="t2i", bot_task="think_recaption"
        )
        assert result.system_prompt_type == "en_think_recaption"

    def test_sys_type_override(self):
        tok = _FakeTokenizer()
        result = build_prompt_tokens(
            "a",
            tok,
            task="t2i",
            bot_task=None,
            sys_type="en_vanilla",
        )
        assert result.system_prompt_type == "en_vanilla"

    def test_vanilla_path(self):
        # Vanilla bypasses the segmented builder: it rebuilds the full string
        # via `build_prompt` and runs `tokenizer.encode` over it. A real HF
        # tokenizer's special-token trie would re-fuse `<|startoftext|>` into
        # one id; our fake tokenizer encodes char-by-char, so we can't pin
        # the BOS id here — we only verify the path returns *some* ids and
        # tags the system prompt type correctly.
        tok = _FakeTokenizer()
        result = build_prompt_tokens(
            "a", tok, task="t2i", bot_task="vanilla"
        )
        assert len(result.token_ids) > 0
        assert result.system_prompt_type == "en_vanilla"
        # Last token is a synthetic char (from user prompt "a"), not a trigger.
        assert result.token_ids[-1] == tok.SYNTHETIC_OFFSET + ord("a")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
