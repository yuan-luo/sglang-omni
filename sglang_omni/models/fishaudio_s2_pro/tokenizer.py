# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro tokenizer adapter using HuggingFace PreTrainedTokenizerFast.

Uses the ``Conversation`` class from fish_speech to build Qwen3-style chat
prompts with system/user/assistant messages, matching the official
``generate_fish_audio_s2.py`` reference implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """A voice-cloning reference for S2-Pro TTS."""

    audio_bytes: bytes
    text: str
    vq_codes: torch.Tensor | None = None


class S2ProTokenizerAdapter:
    """Wraps ``PreTrainedTokenizerFast`` for the S2-Pro model.

    Builds Qwen3 chat-format prompts using the ``Conversation`` class:
    - System message: reference text + VQ codes
    - User message: target text
    - Assistant message: ``<|voice|>`` modality marker
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eos_token_ids(self) -> list[int]:
        return [self._tok.convert_tokens_to_ids("<|im_end|>")]

    @property
    def semantic_begin_id(self) -> int:
        return self._tok.convert_tokens_to_ids("<|semantic:0|>")

    @property
    def semantic_end_id(self) -> int:
        return self._tok.convert_tokens_to_ids("<|semantic:4095|>")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)

    def build_prompt(
        self,
        text: str,
        references: list[Reference] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Build a S2-Pro prompt using the Qwen3 chat format.

        The prompt format follows the official ``generate_fish_audio_s2.py``:
        - System message with reference text (``<|speaker:0|>`` prefixed) and VQ codes
        - User message with target text (``<|speaker:0|>`` prefixed)
        - Assistant message with ``<|voice|>`` modality marker

        Returns:
            ``(input_ids, vq_parts, vq_mask_tokens)`` where:
            - ``input_ids``: [seq_len] token IDs
            - ``vq_parts``: [num_vq, num_codebooks] codebook values (transposed)
            - ``vq_mask_tokens``: [seq_len] boolean mask for VQ positions
        """
        from fish_speech.content_sequence import TextPart, VQPart
        from fish_speech.conversation import Conversation, Message

        conversation = Conversation()

        if references:
            ref_texts = []
            all_codes = []
            for ref in references:
                ref_text = f"<|speaker:0|>{ref.text}" if ref.text else ""
                if ref_text:
                    ref_texts.append(ref_text)
                if ref.vq_codes is not None:
                    all_codes.append(ref.vq_codes)

            reference_text = "\n".join(ref_texts)

            system_parts = [
                TextPart(
                    text="convert the provided text to speech reference to the following:\n\nText:\n",
                    cal_loss=False,
                ),
                TextPart(text=reference_text, cal_loss=False),
                TextPart(text="\n\nSpeech:\n", cal_loss=False),
            ]
            if all_codes:
                combined = torch.cat(all_codes, dim=1)
                system_parts.append(VQPart(codes=combined, cal_loss=False))

            conversation.append(
                Message(
                    role="system",
                    parts=system_parts,
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

        text_with_tag = f"<|speaker:0|>{text}"
        conversation.append(
            Message(
                role="user",
                parts=[TextPart(text=text_with_tag, cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )

        conversation.append(
            Message(
                role="assistant",
                parts=[],
                cal_loss=False,
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )

        encoded = conversation.encode(
            self._tok,
            add_shift=False,
        )

        input_ids = encoded.tokens
        vq_mask_tokens = (
            encoded.vq_mask_tokens if encoded.vq_mask_tokens is not None else None
        )

        vq_parts = None
        if encoded.vq_parts:
            vq_parts = torch.cat(encoded.vq_parts, dim=1).mT

        return input_ids, vq_parts, vq_mask_tokens
