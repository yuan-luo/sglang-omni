# SPDX-License-Identifier: Apache-2.0
"""Codec decoder component for Qwen3-TTS.

Takes accumulated codec tokens (T × num_code_groups) and decodes them to a
waveform via the speech tokenizer bundled with the model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CodecDecoderComponent:
    """Load the speech tokenizer and decode codec tokens to audio."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str | None = None,
    ) -> None:
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        from transformers import AutoConfig

        # Resolve the speech_tokenizer sub-directory.
        from transformers.utils.hub import cached_file

        speech_tok_cfg = cached_file(model_id, "speech_tokenizer/config.json")
        if speech_tok_cfg is None:
            raise FileNotFoundError(
                f"Could not find speech_tokenizer/config.json in {model_id}"
            )
        import os

        speech_tok_dir = os.path.dirname(speech_tok_cfg)

        torch_dtype = _resolve_dtype(dtype)
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            speech_tok_dir,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.device = torch.device(device)

    @torch.inference_mode()
    def decode(self, all_codes: torch.Tensor) -> tuple[np.ndarray, int]:
        """Decode codec tokens to a waveform.

        Parameters
        ----------
        all_codes : torch.Tensor
            Shape ``(T, num_code_groups)`` — accumulated codec IDs.

        Returns
        -------
        waveform : np.ndarray
            1-D float32 waveform.
        sample_rate : int
            Output sample rate (typically 24000).
        """
        if all_codes.numel() == 0:
            return np.zeros(0, dtype=np.float32), 24000

        wavs, sr = self.tokenizer.decode([{"audio_codes": all_codes}])
        return wavs[0], sr


def _resolve_dtype(dtype: str | None) -> torch.dtype:
    if dtype is None:
        return torch.bfloat16
    return getattr(torch, dtype, torch.bfloat16)
