# SPDX-License-Identifier: Apache-2.0
"""Code-predictor (MTP) component for Qwen3-TTS.

Stateless per request: receives ``past_hidden`` + first token ID from the
talker, generates the remaining codebook tokens, and returns the
``sum_embedding`` + full ``codec_ids`` back.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CodePredictorComponent:
    """Load the code-predictor sub-model and run AR generation for codebooks 2–N.

    The code predictor has its own 5-layer transformer, separate per-codebook
    embeddings, a ``small_to_mtp_projection`` and per-codebook ``lm_head``s.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | None = None,
    ) -> None:
        torch_dtype = _resolve_dtype(dtype)

        from qwen_tts.core.models import (
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

        full_model: Qwen3TTSForConditionalGeneration = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
        )

        talker_cfg = full_model.config.talker_config
        self.num_code_groups: int = talker_cfg.num_code_groups

        # Extract code_predictor and the talker's codec_embedding (needed for
        # first-codebook embedding to compute sum_embedding).
        self.code_predictor = full_model.talker.code_predictor
        self.talker_codec_embedding = full_model.talker.model.codec_embedding

        # Free everything else
        del full_model
        torch.cuda.empty_cache()

        self.device = torch.device(device)
        self.dtype = next(self.code_predictor.parameters()).dtype

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        past_hidden: torch.Tensor,
        first_token_id: int,
        sampling: dict[str, Any],
    ) -> dict[str, Any]:
        """Run AR generation for codebooks 2..N.

        Parameters
        ----------
        past_hidden : torch.Tensor
            Hidden state from the talker, shape ``(1, 1, D)``.
        first_token_id : int
            First codebook token ID sampled by the talker.
        sampling : dict
            Sampling parameters (subtalker_*).

        Returns
        -------
        dict with keys:
            ``sum_embedding`` : torch.Tensor (1, 1, D) — sum of all codebook embeddings.
            ``codec_ids`` : torch.Tensor (num_code_groups,) — all codec token IDs.
        """
        past_hidden = past_hidden.to(self.device, dtype=self.dtype)

        # Embed first token with the talker's codec embedding
        first_id_tensor = torch.tensor(
            [[first_token_id]], device=self.device, dtype=torch.long
        )
        last_id_hidden = self.talker_codec_embedding(first_id_tensor)  # (1, 1, D)

        # Prefill the code predictor with [past_hidden, last_id_hidden]
        prefill_embeds = torch.cat([past_hidden, last_id_hidden], dim=1)  # (1, 2, D)

        # Project through small_to_mtp_projection
        prefill_embeds_proj = self.code_predictor.small_to_mtp_projection(
            prefill_embeds
        )

        # Run prefill through the code predictor transformer
        outputs = self.code_predictor.model(
            inputs_embeds=prefill_embeds_proj,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state  # (1, 2, D_mtp)
        past_kv = outputs.past_key_values

        # The first lm_head applies to the token at position 1 (after past_hidden)
        logits = self.code_predictor.lm_head[0](hidden_states[:, -1:, :])  # (1, 1, V_mtp)
        token_id = self._sample(logits[:, 0, :], sampling)

        generated_ids = [token_id]

        # AR loop for codebooks 2..N-1
        for step in range(1, self.num_code_groups - 1):
            # Embed the just-generated token with the step-specific embedding
            token_tensor = torch.tensor(
                [[token_id]], device=self.device, dtype=torch.long
            )
            step_embed = self.code_predictor.model.codec_embedding[step - 1](
                token_tensor
            )  # (1, 1, D_talker)
            step_embed_proj = self.code_predictor.small_to_mtp_projection(
                step_embed
            )

            outputs = self.code_predictor.model(
                inputs_embeds=step_embed_proj,
                past_key_values=past_kv,
                use_cache=True,
            )
            hidden_states = outputs.last_hidden_state
            past_kv = outputs.past_key_values

            logits = self.code_predictor.lm_head[step](
                hidden_states[:, -1:, :]
            )
            token_id = self._sample(logits[:, 0, :], sampling)
            generated_ids.append(token_id)

        # Build full codec_ids: [first_token_id] + generated_ids
        all_ids = [first_token_id] + generated_ids
        codec_ids = torch.tensor(all_ids, dtype=torch.long)  # (G,)

        # Compute sum_embedding = talker_codec_embed(first) + sum(mtp_embed[i](gen[i]))
        sum_emb = last_id_hidden  # (1, 1, D)
        for i, gid in enumerate(generated_ids):
            gid_tensor = torch.tensor(
                [[gid]], device=self.device, dtype=torch.long
            )
            emb = self.code_predictor.model.codec_embedding[i](gid_tensor)
            sum_emb = sum_emb + emb

        return {
            "sum_embedding": sum_emb.cpu(),
            "codec_ids": codec_ids,
            "sampling": sampling,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, sampling: dict[str, Any]) -> int:
        """Sample a single token from logits (1, vocab)."""
        if not sampling.get("subtalker_dosample", True):
            return int(logits.argmax(dim=-1).item())

        temperature = sampling.get("subtalker_temperature", 0.9)
        if temperature > 0:
            logits = logits / temperature

        top_k = sampling.get("subtalker_top_k", 50)
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = logits.topk(top_k)
            logits[logits < vals[:, -1:]] = float("-inf")

        top_p = sampling.get("subtalker_top_p", 1.0)
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dtype(dtype: str | None) -> torch.dtype:
    if dtype is None:
        return torch.bfloat16
    return getattr(torch, dtype, torch.bfloat16)
