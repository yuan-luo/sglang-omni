# SPDX-License-Identifier: Apache-2.0
"""Stage names and routing functions for the Qwen3-TTS pipeline.

Routing graph::

    frontend → talker ⇄ code_predictor → codec_decoder → END
"""

from __future__ import annotations

from typing import Any

from sglang_omni.proto import StagePayload

# -- stage name constants --------------------------------------------------

FRONTEND_STAGE = "frontend"
TALKER_STAGE = "talker"
CODE_PREDICTOR_STAGE = "code_predictor"
CODEC_DECODER_STAGE = "codec_decoder"


# -- routing functions (signatures match the framework) --------------------


def frontend_next(request_id: str, output: Any) -> str:
    """Frontend always routes to the talker."""
    del request_id, output
    return TALKER_STAGE


def talker_next(request_id: str, output: Any) -> str:
    """Talker routes to code_predictor (still generating) or codec_decoder (done)."""
    del request_id
    if isinstance(output, StagePayload):
        data = output.data
    else:
        data = output
    if isinstance(data, dict) and data.get("done", False):
        return CODEC_DECODER_STAGE
    return CODE_PREDICTOR_STAGE


def code_predictor_next(request_id: str, output: Any) -> str:
    """Code predictor always loops back to the talker."""
    del request_id, output
    return TALKER_STAGE


def codec_decoder_next(request_id: str, output: Any) -> None:
    """Codec decoder is the terminal stage."""
    del request_id, output
    return None
