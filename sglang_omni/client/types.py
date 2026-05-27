# SPDX-License-Identifier: Apache-2.0
"""Client request, streaming, and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Message:
    """Chat-style message."""

    role: str
    content: Any

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class UsageInfo:
    """Token usage details."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    engine_time_s: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UsageInfo | None":
        if not data:
            return None
        return cls(
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            total_tokens=data.get("total_tokens"),
            engine_time_s=data.get("engine_time_s"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.engine_time_s is not None:
            d["engine_time_s"] = self.engine_time_s
        return d


@dataclass
class SamplingParams:
    """Sampling configuration."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    max_new_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass
class GenerateRequest:
    """Client-level request (API-agnostic)."""

    model: str | None = None

    prompt: str | None = None
    prompt_token_ids: list[int] | None = None
    messages: list[Message] | None = None

    sampling: SamplingParams = field(default_factory=SamplingParams)
    stage_sampling: dict[str, SamplingParams] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    stream: bool = True
    max_tokens: int | None = None

    # Multi-modal support
    output_modalities: list[str] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "messages": [m.to_dict() for m in self.messages] if self.messages else None,
            "sampling": self.sampling.to_dict(),
            "stage_sampling": (
                {key: params.to_dict() for key, params in self.stage_sampling.items()}
                if self.stage_sampling
                else None
            ),
            "stage_params": self.stage_params,
            "extra_params": dict(self.extra_params),
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "output_modalities": self.output_modalities,
            "metadata": dict(self.metadata),
        }


@dataclass
class GenerateChunk:
    """Streaming chunk from the client."""

    request_id: str
    index: int = 0
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    logprobs: list[float] | None = None
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    stage_id: int | None = None
    stage_name: str | None = None
    modality: str = "text"
    audio_data: Any = None
    sample_rate: int | None = None
    # Image-generation pipelines emit one or more images per request. Each
    # element is the raw encoded image bytes (PNG/JPEG/etc per
    # `image_format`). The list grows across chunks in case the pipeline
    # streams partial-batch outputs, though current pipelines only emit a
    # single terminal chunk with all `n` images.
    image_data: list[bytes] | None = None
    image_format: str | None = None
    image_size: str | None = None  # "768x1024" / "1024x1024" / ...
    # AR-generated text (recaption / think trace). Truncated at
    # `</recaption>` or `</think>` by the pipeline.
    cot_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "index": self.index,
            "token_ids": list(self.token_ids),
            "text": self.text,
            "logprobs": self.logprobs,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage else None,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "modality": self.modality,
            "audio_data": self.audio_data,
            "sample_rate": self.sample_rate,
            "image_data": self.image_data,
            "image_format": self.image_format,
            "image_size": self.image_size,
            "cot_output": self.cot_output,
        }


class AbortLevel(Enum):
    """Abort severity."""

    SOFT = "soft"
    HARD = "hard"


@dataclass
class AbortResult:
    """Abort response from the client."""

    success: bool
    level_applied: AbortLevel
    partial_output: GenerateChunk | None = None


# ---------------------------------------------------------------------------
# High-level result types
# ---------------------------------------------------------------------------


@dataclass
class CompletionAudio:
    """Audio data from a non-streaming completion."""

    id: str
    data: str  # base64
    transcript: str | None = None


@dataclass
class CompletionResult:
    """Result of a non-streaming completion call."""

    request_id: str
    text: str
    audio: CompletionAudio | None = None
    finish_reason: str = "stop"
    usage: UsageInfo | None = None


@dataclass
class CompletionStreamChunk:
    """A single chunk from a streaming completion call."""

    request_id: str
    text: str = ""
    modality: str = "text"
    audio_b64: str | None = None  # already base64-encoded
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    stage_name: str | None = None


@dataclass
class SpeechResult:
    """Result of a text-to-speech call."""

    audio_bytes: bytes
    mime_type: str
    format: str
    usage: UsageInfo | None = None


@dataclass
class ImageItem:
    """A single image returned by an image-generation call.

    Exactly one of `b64_json` or `url` is populated depending on the
    request's `response_format`. `revised_prompt` is the AR-rewritten
    prompt text when a PE bot_task was active, otherwise None.
    """

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


@dataclass
class ImagesResult:
    """Result of an image-generation call."""

    images: list[ImageItem]
    output_format: str
    size: str | None = None
    # AR chain-of-thought / recaption text aggregated across the request's
    # produced images (typically all identical for `n=1`). Truncated at
    # `</recaption>` or `</think>`. None when AR text capture is disabled
    # or `bot_task=vanilla`.
    cot_output: str | None = None
    usage: UsageInfo | None = None


class ClientError(Exception):
    """Error raised by the Client layer."""
