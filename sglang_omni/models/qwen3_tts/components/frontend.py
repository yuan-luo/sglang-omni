# SPDX-License-Identifier: Apache-2.0
"""CPU-side frontend for Qwen3-TTS: tokenise text and prepare stage payload."""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoConfig, AutoProcessor

from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


class Qwen3TTSFrontend:
    """Tokenise input text and validate speaker / language for Qwen3-TTS.

    This runs on CPU and produces the data dict consumed by the talker stage.
    """

    def __init__(self, model_id: str) -> None:
        # Register model types so AutoConfig/AutoProcessor can resolve them.
        from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        self.processor = AutoProcessor.from_pretrained(
            model_id, fix_mistral_regex=True
        )

        # Load config to extract speaker/language tables.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        talker_cfg = config.talker_config
        self.spk_id: dict[str, int] = talker_cfg.spk_id
        self.codec_language_id: dict[str, int] = talker_cfg.codec_language_id
        self.spk_is_dialect: dict[str, Any] = talker_cfg.spk_is_dialect
        self.supported_speakers: set[str] = set(self.spk_id.keys())
        self.supported_languages: set[str] = {"auto"} | {
            k for k in self.codec_language_id if "dialect" not in k
        }

    # ----- chat template helpers -----

    @staticmethod
    def _build_assistant_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize(self, text: str) -> list[int]:
        enc = self.processor(text=text, return_tensors="pt", padding=True)
        return enc["input_ids"].squeeze(0).tolist()

    # ----- validation -----

    def _validate_speaker(self, speaker: str) -> None:
        if speaker.lower() not in self.supported_speakers:
            raise ValueError(
                f"Unsupported speaker: {speaker!r}. "
                f"Supported: {sorted(self.supported_speakers)}"
            )

    def _validate_language(self, language: str) -> None:
        if language.lower() not in self.supported_languages:
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Supported: {sorted(self.supported_languages)}"
            )

    # ----- main entry -----

    def __call__(self, payload: StagePayload) -> StagePayload:
        """Process *payload* and prepare data for the talker stage."""
        params = payload.request.params
        text: str = payload.request.inputs
        speaker: str = params.get("speaker", "Chelsie")
        language: str = params.get("language", "Auto")

        self._validate_speaker(speaker)
        self._validate_language(language)

        # Resolve language → language_id (None for "auto")
        lang_lower = language.lower()
        if lang_lower == "auto":
            language_id = None
        else:
            language_id = self.codec_language_id[lang_lower]

        # Dialect override for Chinese speakers
        spk_lower = speaker.lower()
        dialect = self.spk_is_dialect.get(spk_lower, False)
        if dialect and lang_lower in ("chinese", "auto"):
            language_id = self.codec_language_id[dialect]

        # Tokenize text into IDs
        prompt = self._build_assistant_text(text)
        input_ids = self._tokenize(prompt)

        # Non-streaming mode flag
        non_streaming_mode: bool = params.get("non_streaming_mode", True)

        # Sampling params (use defaults from generate_config.json style)
        sampling = {
            "do_sample": params.get("do_sample", True),
            "top_k": params.get("top_k", 50),
            "top_p": params.get("top_p", 1.0),
            "temperature": params.get("temperature", 0.9),
            "repetition_penalty": params.get("repetition_penalty", 1.05),
            "max_new_tokens": params.get("max_new_tokens", 2048),
            "subtalker_dosample": params.get("subtalker_dosample", True),
            "subtalker_top_k": params.get("subtalker_top_k", 50),
            "subtalker_top_p": params.get("subtalker_top_p", 1.0),
            "subtalker_temperature": params.get("subtalker_temperature", 0.9),
        }

        payload.data = {
            "input_ids": input_ids,
            "speaker": speaker,
            "language_id": language_id,
            "non_streaming_mode": non_streaming_mode,
            "sampling": sampling,
        }
        return payload
