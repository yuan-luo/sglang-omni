# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for thinker + talker pipeline.

Usage:
    python tests/test_thinker_talker_e2e.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --prompt "Hello, how are you?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import pytest
import torch
from transformers import AutoTokenizer

from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.models.qwen3_omni.pipeline.stages import (
    create_decode_executor,
    create_sglang_talker_executor_from_config,
    create_sglang_thinker_executor_from_config,
)
from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state
from sglang_omni.proto import OmniRequest, StagePayload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

NUM_CODE_GROUPS = 16


def _build_payload(
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[StagePayload, int]:
    """Tokenize prompt and build the initial StagePayload."""
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = prompt

    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)

    state = PipelineState(
        prompt={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_text": prompt_text,
        },
    )
    payload = StagePayload(
        request_id="e2e-test-0",
        request=OmniRequest(
            inputs=prompt,
            params={"max_new_tokens": max_new_tokens, "temperature": 0.0},
        ),
        data=state.to_dict(),
    )
    return payload, input_ids.shape[0]


def _verify_thinker(
    state: PipelineState,
    tokenizer: AutoTokenizer,
    input_token_count: int,
) -> str:
    """Verify thinker output and return decoded text."""
    thinker_out = state.thinker_out
    assert thinker_out is not None, "Thinker produced no output"

    output_ids = thinker_out["output_ids"]
    assert len(output_ids) > 0, "Thinker produced zero tokens"

    thinker_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    assert len(thinker_text.strip()) > 0, "Thinker text is empty after decoding"
    print(f"[PASS] Thinker: {len(output_ids)} tokens")
    print(f"       Text: {thinker_text[:200]!r}")

    # Hidden states shape
    talker_inputs = state.talker_inputs
    assert talker_inputs, "talker_inputs not populated"
    thinker_embed = talker_inputs["thinker_embed"]
    thinker_hidden = talker_inputs["thinker_hidden"]
    assert isinstance(thinker_embed, torch.Tensor), "thinker_embed is not a Tensor"
    assert isinstance(thinker_hidden, torch.Tensor), "thinker_hidden is not a Tensor"
    assert (
        thinker_embed.dim() == 2
    ), f"thinker_embed should be 2D, got {thinker_embed.dim()}D"
    assert (
        thinker_hidden.dim() == 2
    ), f"thinker_hidden should be 2D, got {thinker_hidden.dim()}D"

    embed_seq_len = thinker_embed.shape[0]
    hidden_seq_len = thinker_hidden.shape[0]
    expected_seq_len = input_token_count + len(output_ids)
    assert (
        embed_seq_len == hidden_seq_len
    ), f"embed/hidden seq_len mismatch: {embed_seq_len} vs {hidden_seq_len}"
    assert (
        abs(embed_seq_len - expected_seq_len) <= 1
    ), f"seq_len {embed_seq_len} too far from expected {expected_seq_len}"
    assert (
        thinker_embed.shape[1] == thinker_hidden.shape[1]
    ), f"hidden_size mismatch: {thinker_embed.shape[1]} vs {thinker_hidden.shape[1]}"
    print(
        f"[PASS] Hidden states: embed={thinker_embed.shape}, hidden={thinker_hidden.shape}"
    )

    # Hidden states values
    assert not torch.all(thinker_embed == 0), "thinker_embed is all zeros"
    assert not torch.all(thinker_hidden == 0), "thinker_hidden is all zeros"
    assert torch.isfinite(thinker_embed).all(), "thinker_embed contains NaN/Inf"
    assert torch.isfinite(thinker_hidden).all(), "thinker_hidden contains NaN/Inf"
    print("[PASS] Hidden states: non-zero, finite")

    return thinker_text


def _verify_talker(state: PipelineState) -> None:
    """Verify talker output codec codes structure."""
    talker_out = state.talker_out
    assert talker_out is not None, "Talker produced no output"

    codec_codes = talker_out["codec_codes"]
    assert isinstance(
        codec_codes, list
    ), f"codec_codes should be list, got {type(codec_codes)}"
    assert (
        len(codec_codes) == NUM_CODE_GROUPS
    ), f"Expected {NUM_CODE_GROUPS} code groups, got {len(codec_codes)}"
    first_group_len = len(codec_codes[0])
    assert first_group_len > 0, "codec_codes[0] is empty"
    for i, group in enumerate(codec_codes):
        assert (
            len(group) == first_group_len
        ), f"codec_codes[{i}] len={len(group)} != codec_codes[0] len={first_group_len}"
    print(f"[PASS] Talker: codec_codes shape=[{NUM_CODE_GROUPS}, {first_group_len}]")


def _verify_decode(result: dict, thinker_text: str) -> None:
    """Verify decode output and cross-check with thinker text."""
    assert isinstance(result, dict), f"Decode returned non-dict: {type(result)}"
    assert "text" in result, "Decode result missing 'text'"
    assert "events" in result, "Decode result missing 'events'"
    assert len(result["events"]) > 0, "Decode produced zero events"

    decode_text = result["text"]
    assert len(decode_text.strip()) > 0, "Decode text is empty"
    assert decode_text == thinker_text, (
        f"Decode text inconsistent with thinker:\n"
        f"  thinker: {thinker_text[:100]!r}\n"
        f"  decode:  {decode_text[:100]!r}"
    )
    print(f"[PASS] Decode: {len(result['events'])} events, text matches thinker")


async def run_e2e(
    model_path: str,
    prompt: str,
    gpu_id: int = 0,
    max_new_tokens: int = 64,
    thinker_max_seq_len: int = 4096,
    talker_max_seq_len: int = 4096,
) -> dict:
    """Run the full thinker → talker → decode pipeline and return results."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Creating executors...")
    thinker_executor = create_sglang_thinker_executor_from_config(
        model_path=model_path,
        gpu_id=gpu_id,
        thinker_max_seq_len=thinker_max_seq_len,
    )
    talker_executor = create_sglang_talker_executor_from_config(
        model_path=model_path,
        gpu_id=gpu_id,
        talker_max_seq_len=talker_max_seq_len,
    )
    decode_executor = create_decode_executor(model_path)
    payload, input_token_count = _build_payload(tokenizer, prompt, max_new_tokens)

    # Thinker
    logger.info("Running thinker...")
    await thinker_executor.start()
    try:
        await thinker_executor.add_request(payload)
        payload = await thinker_executor.get_result()
    finally:
        await thinker_executor.stop()
    thinker_text = _verify_thinker(load_state(payload), tokenizer, input_token_count)

    # Talker
    logger.info("Running talker...")
    await talker_executor.start()
    try:
        await talker_executor.add_request(payload)
        payload = await talker_executor.get_result()
    finally:
        await talker_executor.stop()
    _verify_talker(load_state(payload))

    # Decode
    logger.info("Running decode...")
    await decode_executor.add_request(payload)
    payload = await decode_executor.get_result()
    result = payload.data
    _verify_decode(result, thinker_text)

    print("\n=== E2E TEST PASSED ===")
    return result


MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


@pytest.mark.e2e
def test_thinker_talker_e2e() -> None:
    """Pytest entry: thinker → talker → decode pipeline."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for thinker-talker E2E test")
    asyncio.run(run_e2e(model_path=MODEL_PATH, prompt="Hello, how are you?"))


def main():
    parser = argparse.ArgumentParser(description="Thinker + Talker E2E test")
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--thinker-max-seq-len", type=int, default=4096)
    parser.add_argument("--talker-max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    asyncio.run(
        run_e2e(
            model_path=args.model_path,
            prompt=args.prompt,
            gpu_id=args.gpu_id,
            max_new_tokens=args.max_new_tokens,
            thinker_max_seq_len=args.thinker_max_seq_len,
            talker_max_seq_len=args.talker_max_seq_len,
        )
    )


if __name__ == "__main__":
    main()
