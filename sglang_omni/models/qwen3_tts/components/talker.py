# SPDX-License-Identifier: Apache-2.0
"""Talker component for the Qwen3-TTS pipeline.

Manages per-request GPU state (KV cache, generation step, accumulated codes)
and implements prefill + single decode-step logic.  Only the minimal
inter-stage tensors are returned; the heavy KV cache stays on-device.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from transformers import DynamicCache

from sglang_omni.models.utils.hf import instantiate_module, load_hf_config
from sglang_omni.models.weight_loader import load_module, resolve_dtype

logger = logging.getLogger(__name__)

# Weight prefixes in the HF checkpoint.
_TALKER_PREFIX = "talker."
_CODE_PREDICTOR_PREFIX = "talker.code_predictor."


# ---------------------------------------------------------------------------
# Per-request state kept on-device between talker ↔ code_predictor round-trips
# ---------------------------------------------------------------------------


@dataclass
class TalkerState:
    """Mutable state for one in-flight TTS request."""

    past_key_values: DynamicCache
    attention_mask: torch.Tensor
    rope_deltas: torch.Tensor
    trailing_text_hidden: torch.Tensor  # (1, T_text, D)
    tts_pad_embed: torch.Tensor  # (1, 1, D)
    generation_step: int = 0
    all_codec_ids: list[torch.Tensor] = field(default_factory=list)
    cache_seq_len: int = 0  # tracks how long the KV cache is


# ---------------------------------------------------------------------------
# Model component
# ---------------------------------------------------------------------------


def _load_talker(
    model_id: str,
    *,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    """Instantiate ``Qwen3TTSTalkerForConditionalGeneration`` and load only
    its own weights (excluding the code-predictor sub-model)."""
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )

    cfg: Qwen3TTSConfig = load_hf_config(model_id)
    talker_cfg = cfg.talker_config

    talker = instantiate_module(
        Qwen3TTSTalkerForConditionalGeneration, talker_cfg
    )
    # Delete the code_predictor *before* loading weights so load_module
    # doesn't need to find matching keys for it.
    del talker.code_predictor

    return load_module(
        talker,
        model_id,
        prefix=_TALKER_PREFIX,
        exclude=_CODE_PREDICTOR_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=False,  # code_predictor keys are excluded
    )


class TalkerComponent(nn.Module):
    """Talker sub-model extracted from ``Qwen3TTSForConditionalGeneration``.

    Provides ``prefill`` / ``decode_step`` for the pipeline's cyclic loop.
    The code-predictor weights are **not** loaded here — they live in a
    separate stage so one talker can fan out to N code predictors.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)

        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

        cfg: Qwen3TTSConfig = load_hf_config(model_id)
        self.config = cfg
        talker_cfg = cfg.talker_config

        # Load talker weights (excluding code predictor).
        self.talker = _load_talker(
            model_id, torch_dtype=torch_dtype, device=device
        )

        self._device = torch.device(device)
        self._dtype = torch_dtype or torch.bfloat16

        # Config shortcuts
        self.num_code_groups: int = talker_cfg.num_code_groups
        self.codec_eos_token_id: int = talker_cfg.codec_eos_token_id
        self.codec_think_id: int = talker_cfg.codec_think_id
        self.codec_think_bos_id: int = talker_cfg.codec_think_bos_id
        self.codec_think_eos_id: int = talker_cfg.codec_think_eos_id
        self.codec_nothink_id: int = talker_cfg.codec_nothink_id
        self.codec_pad_id: int = talker_cfg.codec_pad_id
        self.codec_bos_id: int = talker_cfg.codec_bos_id
        self.tts_bos_token_id: int = cfg.tts_bos_token_id
        self.tts_eos_token_id: int = cfg.tts_eos_token_id
        self.tts_pad_token_id: int = cfg.tts_pad_token_id
        self.vocab_size: int = talker_cfg.vocab_size
        # Suppress tokens: upper 1024 of vocab except EOS
        self.suppress_tokens: list[int] = [
            i
            for i in range(self.vocab_size - 1024, self.vocab_size)
            if i != self.codec_eos_token_id
        ]
        self.spk_id: dict[str, int] = talker_cfg.spk_id

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _text_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Project text token IDs through text_embedding + text_projection."""
        return self.talker.text_projection(
            self.talker.get_text_embeddings()(token_ids)
        )

    def _codec_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed codec token IDs through the talker's codec embedding."""
        return self.talker.get_input_embeddings()(token_ids)

    def _special_embeds(self, input_ids_dtype: torch.dtype):
        """Return (tts_bos, tts_eos, tts_pad) embeddings, each (1, 1, D)."""
        ids = torch.tensor(
            [[self.tts_bos_token_id, self.tts_eos_token_id, self.tts_pad_token_id]],
            device=self._device,
            dtype=input_ids_dtype,
        )
        return self.talker.text_projection(
            self.talker.get_text_embeddings()(ids)
        ).chunk(3, dim=1)

    # ------------------------------------------------------------------
    # Prefill — first visit from frontend
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: list[int],
        speaker: str,
        language_id: int | None,
        non_streaming_mode: bool,
        sampling: dict[str, Any],
    ) -> tuple[TalkerState, dict[str, Any]]:
        """Run the talker prefill and return (state, inter-stage dict).

        Returns
        -------
        state : TalkerState
            On-device state to keep across decode steps.
        output : dict
            Minimal data to send to the code-predictor stage.
        """
        ids_tensor = torch.tensor(
            [input_ids], device=self._device, dtype=torch.long
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._special_embeds(
            ids_tensor.dtype
        )

        # --- Codec prefix embeddings ---
        if language_id is None:
            codec_prefix = [
                self.codec_nothink_id,
                self.codec_think_bos_id,
                self.codec_think_eos_id,
            ]
        else:
            codec_prefix = [
                self.codec_think_id,
                self.codec_think_bos_id,
                language_id,
                self.codec_think_eos_id,
            ]

        codec_prefix_embed = self._codec_embed(
            torch.tensor([codec_prefix], device=self._device, dtype=torch.long)
        )

        # Speaker embedding
        spk_lower = speaker.lower()
        spk_id_val = self.spk_id[spk_lower]
        speaker_embed = self._codec_embed(
            torch.tensor(spk_id_val, device=self._device, dtype=torch.long)
        )

        codec_suffix_embed = self._codec_embed(
            torch.tensor(
                [[self.codec_pad_id, self.codec_bos_id]],
                device=self._device,
                dtype=torch.long,
            )
        )

        # codec_input_embedding = [prefix, speaker, suffix]
        codec_input_embedding = torch.cat(
            [
                codec_prefix_embed,
                speaker_embed.view(1, 1, -1),
                codec_suffix_embed,
            ],
            dim=1,
        )

        # --- Role embedding: <|im_start|>assistant\n ---
        role_embed = self._text_embed(ids_tensor[:, :3])

        # --- Build combined input embedding ---
        n_codec = codec_input_embedding.shape[1]
        pad_part = tts_pad_embed.expand(-1, n_codec - 2, -1)
        overlay = torch.cat([pad_part, tts_bos_embed], dim=1)
        combined = overlay + codec_input_embedding[:, :-1]

        talker_input = torch.cat([role_embed, combined], dim=1)

        # --- Text drip-feeding ---
        if non_streaming_mode:
            text_ids = ids_tensor[:, 3:-5]
            text_embed = self._text_embed(text_ids)
            text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
            n_text = text_embed.shape[1]

            codec_pad_repeat = self._codec_embed(
                torch.tensor(
                    [[self.codec_pad_id] * n_text],
                    device=self._device,
                    dtype=torch.long,
                )
            )
            text_plus_codec = text_embed + codec_pad_repeat
            last_token = tts_pad_embed + codec_input_embedding[:, -1:]

            talker_input = torch.cat(
                [talker_input, text_plus_codec, last_token], dim=1
            )
            trailing_text_hidden = tts_pad_embed
        else:
            first_text_embed = (
                self._text_embed(ids_tensor[:, 3:4])
                + codec_input_embedding[:, -1:]
            )
            talker_input = torch.cat([talker_input, first_text_embed], dim=1)

            remaining = ids_tensor[:, 4:-5]
            trailing_text_hidden = torch.cat(
                [self._text_embed(remaining), tts_eos_embed], dim=1
            )

        # --- Attention mask ---
        seq_len = talker_input.shape[1]
        attention_mask = torch.ones(
            (1, seq_len), device=self._device, dtype=torch.long
        )

        # --- Forward through talker model ---
        outputs = self.talker(
            inputs_embeds=talker_input,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            generation_step=-1,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        first_token_id = self._sample(logits, sampling)
        past_hidden = outputs.past_hidden

        state = TalkerState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            rope_deltas=self.talker.rope_deltas,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            generation_step=0,
            all_codec_ids=[],
            cache_seq_len=seq_len,
        )

        output = {
            "past_hidden": past_hidden.cpu(),
            "first_token_id": first_token_id,
            "sampling": sampling,
        }
        return state, output

    # ------------------------------------------------------------------
    # Decode step — returning from code_predictor
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def decode_step(
        self,
        state: TalkerState,
        sum_embedding: torch.Tensor,
        codec_ids: torch.Tensor,
        sampling: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Run one talker decode step after receiving MTP results.

        Parameters
        ----------
        state : TalkerState
            Mutable per-request state.
        sum_embedding : torch.Tensor
            Sum of all codebook embeddings from MTP, shape (1, 1, D).
        codec_ids : torch.Tensor
            All codec IDs for this step, shape (num_code_groups,).
        sampling : dict
            Sampling parameters.

        Returns
        -------
        done : bool
            Whether generation is complete (EOS or max_tokens).
        output : dict
            Data to send to the next stage (code_predictor or codec_decoder).
        """
        state.all_codec_ids.append(codec_ids.cpu())

        first_code = int(codec_ids[0])
        max_tokens = sampling.get("max_new_tokens", 2048)
        done = (first_code == self.codec_eos_token_id) or (
            state.generation_step >= max_tokens
        )

        if done:
            if state.all_codec_ids:
                all_codes = torch.stack(state.all_codec_ids, dim=0)
            else:
                all_codes = torch.zeros(
                    (0, self.num_code_groups), dtype=torch.long
                )
            return True, {"all_codes": all_codes}

        sum_embedding = sum_embedding.to(self._device, dtype=self._dtype)
        inputs_embeds = sum_embedding

        gen_step = state.generation_step
        if gen_step < state.trailing_text_hidden.shape[1]:
            inputs_embeds = (
                inputs_embeds
                + state.trailing_text_hidden[:, gen_step].unsqueeze(1)
            )
        else:
            inputs_embeds = inputs_embeds + state.tts_pad_embed

        state.attention_mask = torch.cat(
            [
                state.attention_mask,
                torch.ones((1, 1), device=self._device, dtype=torch.long),
            ],
            dim=1,
        )
        state.cache_seq_len += 1

        cache_position = torch.tensor(
            [state.cache_seq_len - 1], device=self._device, dtype=torch.long
        )
        delta = cache_position[0] + state.rope_deltas
        position_ids = torch.arange(1, device=self._device)
        position_ids = position_ids.view(1, -1).expand(1, -1)
        position_ids = position_ids.add(delta)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.talker.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=state.attention_mask,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )

        hidden = outputs.last_hidden_state
        logits = self.talker.codec_head(hidden)[:, -1, :]

        next_token_id = self._sample(logits, sampling)
        past_hidden = hidden

        state.past_key_values = outputs.past_key_values
        state.generation_step += 1

        output = {
            "past_hidden": past_hidden.cpu(),
            "first_token_id": next_token_id,
            "sampling": sampling,
        }
        return False, output

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, sampling: dict[str, Any]) -> int:
        """Sample a single token from logits (1, vocab)."""
        logits = logits.clone()

        for tid in self.suppress_tokens:
            logits[:, tid] = float("-inf")

        if not sampling.get("do_sample", True):
            return int(logits.argmax(dim=-1).item())

        temperature = sampling.get("temperature", 0.9)
        if temperature > 0:
            logits = logits / temperature

        top_k = sampling.get("top_k", 50)
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = logits.topk(top_k)
            logits[logits < vals[:, -1:]] = float("-inf")

        top_p = sampling.get("top_p", 1.0)
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
