# SPDX-License-Identifier: Apache-2.0
"""Model-specific frontend preprocessing for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.frontends import (
    build_audio_mm_inputs,
    build_image_mm_inputs,
    compute_audio_cache_key,
    compute_image_cache_key,
    ensure_audio_list,
    ensure_chat_template,
    ensure_image_list,
    normalize_messages,
)
from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.proto import StagePayload


class Qwen3OmniFrontend:
    """CPU-side preprocessing and tokenization using the HF processor."""

    def __init__(self, model_path: str):
        self.model_dir = model_path
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        ensure_chat_template(self.tokenizer, model_id=self.model_dir)

    def _build_multimodal_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        num_images: int,
        num_audios: int,
    ) -> list[dict[str, Any]]:
        """Convert simple messages to HF's structured multimodal format."""
        if num_images == 0 and num_audios == 0:
            return messages

        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Only inject placeholders into the last user message
            if i == len(messages) - 1 and role == "user":
                content_parts: list[dict[str, Any]] = []
                # Placeholders come BEFORE text (Qwen3-Omni format)
                for _ in range(num_images):
                    content_parts.append({"type": "image"})
                for _ in range(num_audios):
                    content_parts.append({"type": "audio"})
                content_parts.append({"type": "text", "text": content})
                result.append({"role": role, "content": content_parts})
            else:
                result.append(msg)

        return result

    def __call__(self, payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, dict):
            messages = inputs.get("messages", [])
            raw_images = inputs.get("images")
            raw_audios = inputs.get("audio") or inputs.get("audios")
            audio_target_sr = int(inputs.get("audio_target_sr", 16000))

            # Compute cache keys BEFORE conversion (paths are cheap to hash)
            image_cache_key = compute_image_cache_key(raw_images)
            audio_cache_key = compute_audio_cache_key(raw_audios)

            images = ensure_image_list(raw_images)
            audios = ensure_audio_list(raw_audios, target_sr=audio_target_sr)
        else:
            messages = inputs
            images = []
            audios = []
            image_cache_key = None
            audio_cache_key = None

        messages_norm = normalize_messages(messages)
        messages_mm = self._build_multimodal_messages(
            messages_norm,
            num_images=len(images),
            num_audios=len(audios),
        )
        prompt_text = self.processor.apply_chat_template(
            messages_mm,
            add_generation_prompt=True,
            tokenize=False,
        )

        hf_inputs = self.processor(
            text=prompt_text,
            images=images or None,
            audio=audios or None,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = hf_inputs["input_ids"][0]
        attention_mask = hf_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask[0]
        else:
            attention_mask = torch.ones_like(input_ids)

        mm_inputs: dict[str, Any] = {
            "image": build_image_mm_inputs(hf_inputs),
            "audio": build_audio_mm_inputs(hf_inputs),
        }

        # Build encoder_inputs with cache_key for efficient caching
        image_encoder_inputs = {**mm_inputs["image"]}
        if image_cache_key:
            image_encoder_inputs["cache_key"] = image_cache_key

        audio_encoder_inputs = {**mm_inputs["audio"]}
        if audio_cache_key:
            audio_encoder_inputs["cache_key"] = audio_cache_key

        state = PipelineState(
            raw_inputs=inputs,
            mm_inputs=mm_inputs,
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            encoder_inputs={
                "image_encoder": image_encoder_inputs,
                "audio_encoder": audio_encoder_inputs,
            },
            stream_state={"token_ids": [], "text": ""},
        )
        payload.data = state.to_dict()
        return payload
