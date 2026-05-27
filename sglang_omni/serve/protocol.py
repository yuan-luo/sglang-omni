# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible request/response protocol definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared / Common
# ---------------------------------------------------------------------------


class UsageResponse(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Chat Completion
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionAudio(BaseModel):
    """Audio data returned in a chat completion response."""

    id: str
    data: str  # base64-encoded audio
    expires_at: int | None = None
    transcript: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    messages: list[ChatMessage]

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None

    # Streaming
    stream: bool = False

    # Multi-modal output control
    modalities: list[str] | None = None  # e.g. ["text", "audio"]

    # Audio output configuration
    audio: dict[str, Any] | None = None  # {"voice": "...", "format": "wav"}

    # Audio input (sglang-omni extension)
    # Can be a list of audio file paths (local paths or URLs)
    audios: list[str] | None = None

    # Image input (sglang-omni extension)
    # Can be a list of image file paths (local paths or URLs)
    images: list[str] | None = None

    # Video input (sglang-omni extension)
    # Can be a list of video file paths (local paths or URLs)
    videos: list[str] | None = None
    video_fps: float | None = None
    video_max_frames: int | None = None
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_total_pixels: int | None = None

    # Per-stage sampling overrides (sglang-omni specific)
    stage_sampling: dict[str, dict[str, Any]] | None = None
    stage_params: dict[str, dict[str, Any]] | None = None

    # Talker-specific overrides for Qwen3-Omni speech output
    talker_temperature: float | None = None
    talker_top_p: float | None = None
    talker_top_k: int | None = None
    talker_repetition_penalty: float | None = None
    talker_max_new_tokens: int | None = None

    # Misc
    request_id: str | None = None
    user: str | None = None

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: dict[str, Any]
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageResponse | None = None


class ChatCompletionStreamDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    audio: ChatCompletionAudio | None = None


class ChatCompletionStreamChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: UsageResponse | None = None


# ---------------------------------------------------------------------------
# Speech (TTS)
# ---------------------------------------------------------------------------


class SpeechReference(BaseModel):
    """Reference item for voice cloning in /v1/audio/speech."""

    audio_path: str | None = None
    text: str | None = None
    vq_codes: list[list[int]] | list[int] | None = None


class CreateSpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request.

    Standard OpenAI fields plus extensions for advanced TTS models
    (e.g. voice cloning, style instructions).
    """

    model_config = ConfigDict(populate_by_name=True)

    # Standard OpenAI fields
    model: str | None = None
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False

    # Advanced TTS extensions
    task_type: str | None = None  # e.g. "Base", "CustomVoice", "VoiceDesign"
    language: str | None = None
    instructions: str | None = None  # style/emotion instructions

    # Voice cloning parameters
    ref_audio: str | None = None  # path or URL to reference audio
    ref_text: str | None = None  # transcript of reference audio
    references: list[SpeechReference] | None = None  # S2-Pro-style refs

    # Generation parameters
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None

    # Per-stage overrides (sglang-omni specific)
    stage_params: dict[str, dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------


class ModelPermission(BaseModel):
    """Model permission info."""

    id: str = "modelperm-default"
    object: str = "model_permission"
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True


class ModelCard(BaseModel):
    """A single model entry."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "sglang-omni"
    permission: list[ModelPermission] = Field(
        default_factory=lambda: [ModelPermission()]
    )
    root: str | None = None


class ModelList(BaseModel):
    """Response for GET /v1/models."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Image generation (POST /v1/images/generations)
# ---------------------------------------------------------------------------
class ImagesGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request, with diffusion extensions."""

    model_config = ConfigDict(populate_by_name=True)

    # OpenAI canonical fields.
    model: str | None = None
    prompt: str
    n: int = 1
    size: str | None = None  # "1024x1024" / "768x1024" / "auto"
    response_format: str = "b64_json"  # "b64_json" | "url"
    user: str | None = None

    # Diffusion extensions.
    output_format: str = "png"  # "png" | "jpeg" | "webp"
    output_compression: int | None = None  # 0..100 for jpeg/webp
    seed: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None

    # Image-aware prompt routing.
    # `bot_task` ∈ {None, "vanilla", "recaption", "think", "think_recaption"}
    # selects (sys_type, trigger_tag) for the AR backbone.
    bot_task: str | None = None
    # `sys_type` overrides the default system-prompt template for the
    # selected bot_task. Rarely needed; allows callers to swap in a custom
    # template (e.g. en_recaption for direct rewrite without unified flow).
    sys_type: str | None = None
    # When `sys_type="custom"`, the caller supplies the literal system prompt
    # body here.
    system_prompt: str | None = None

    # Per-stage sampling overrides — keyed by stage name. Use this to bound
    # AR decode (e.g. {"ar": {"max_tokens": 1024}}) without touching DiT.
    stage_sampling: dict[str, dict[str, Any]] | None = None

    # Request id (server-generated if absent).
    request_id: str | None = None


class ImagesGenerationDataItem(BaseModel):
    """Single image produced by a generation request."""

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImagesGenerationResponse(BaseModel):
    """OpenAI-compatible image generation response."""

    created: int
    data: list[ImagesGenerationDataItem]
    output_format: str
    size: str | None = None
    # AR chain-of-thought / recaption text. Populated for
    # `bot_task ∈ {recaption, think, think_recaption}`; truncated at the
    # closing `</recaption>` or `</think>` tag. None for `bot_task=vanilla`
    # and when AR text capture is disabled.
    cot_output: str | None = None
