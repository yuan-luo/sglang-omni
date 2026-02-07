# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the Qwen3-TTS pipeline."""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

from sglang_omni.executors import FrontendExecutor
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frontend (CPU)
# ---------------------------------------------------------------------------


def create_frontend_executor(model_id: str) -> FrontendExecutor:
    from sglang_omni.models.qwen3_tts.components.frontend import Qwen3TTSFrontend

    frontend = Qwen3TTSFrontend(model_id)

    def _frontend(payload: StagePayload) -> StagePayload:
        return frontend(payload)

    return FrontendExecutor(_frontend)


# ---------------------------------------------------------------------------
# Talker (GPU) — stateful, manages per-request KV caches
# ---------------------------------------------------------------------------


def create_talker_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> FrontendExecutor:
    from sglang_omni.models.qwen3_tts.components.talker import (
        TalkerComponent,
        TalkerState,
    )

    import torch

    component = TalkerComponent(model_id, device=device, dtype=dtype)
    # Per-request mutable state, keyed by request_id.
    _states: dict[str, TalkerState] = {}

    def _talker(payload: StagePayload) -> StagePayload:
        rid = payload.request_id
        data: dict[str, Any] = payload.data

        if rid not in _states:
            # --- First visit (from frontend): prefill ---
            state, output = component.prefill(
                input_ids=data["input_ids"],
                speaker=data["speaker"],
                language_id=data["language_id"],
                non_streaming_mode=data["non_streaming_mode"],
                sampling=data["sampling"],
            )
            _states[rid] = state
            output["done"] = False
            payload.data = output
            return payload

        # --- Subsequent visit (from code_predictor): decode step ---
        state = _states[rid]
        sum_embedding = data["sum_embedding"]
        codec_ids = data["codec_ids"]
        sampling = data["sampling"]

        if isinstance(sum_embedding, np.ndarray):
            sum_embedding = torch.from_numpy(sum_embedding)
        if isinstance(codec_ids, (list, np.ndarray)):
            codec_ids = torch.as_tensor(codec_ids, dtype=torch.long)

        done, output = component.decode_step(state, sum_embedding, codec_ids, sampling)

        if done:
            del _states[rid]
            output["done"] = True
        else:
            output["done"] = False

        payload.data = output
        return payload

    return FrontendExecutor(_talker)


# ---------------------------------------------------------------------------
# Code predictor (GPU) — stateless per request
# ---------------------------------------------------------------------------


def create_code_predictor_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> FrontendExecutor:
    from sglang_omni.models.qwen3_tts.components.code_predictor import (
        CodePredictorComponent,
    )

    import torch

    component = CodePredictorComponent(model_id, device=device, dtype=dtype)

    def _mtp(payload: StagePayload) -> StagePayload:
        data: dict[str, Any] = payload.data
        past_hidden = data["past_hidden"]
        first_token_id = data["first_token_id"]
        sampling = data["sampling"]

        if isinstance(past_hidden, np.ndarray):
            past_hidden = torch.from_numpy(past_hidden)

        result = component.generate(
            past_hidden=past_hidden,
            first_token_id=first_token_id,
            sampling=sampling,
        )
        payload.data = result
        return payload

    return FrontendExecutor(_mtp)


# ---------------------------------------------------------------------------
# Codec decoder (GPU/CPU)
# ---------------------------------------------------------------------------


def create_codec_decoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> FrontendExecutor:
    from sglang_omni.models.qwen3_tts.components.codec_decoder import (
        CodecDecoderComponent,
    )

    import torch

    component = CodecDecoderComponent(model_id, device=device, dtype=dtype)

    def _decode(payload: StagePayload) -> StagePayload:
        data: dict[str, Any] = payload.data
        all_codes = data["all_codes"]

        if isinstance(all_codes, np.ndarray):
            all_codes = torch.from_numpy(all_codes)

        waveform, sample_rate = component.decode(all_codes)

        # Encode waveform as WAV bytes
        buf = io.BytesIO()
        import soundfile as sf

        sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
        audio_bytes = buf.getvalue()

        payload.data = {
            "audio": audio_bytes,
            "sample_rate": sample_rate,
            "modality": "audio",
        }
        return payload

    return FrontendExecutor(_decode)
