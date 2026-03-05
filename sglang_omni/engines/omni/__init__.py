# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine for all model types."""

from .engine import OmniEngine
from .factory import (
    create_ar_engine,
    create_encoder_engine,
    create_sglang_ar_engine,
    create_talker_codec_engine,
)
from .model_runner import ModelRunner
from .runtime.ar import ARRequestData
from .runtime.encoder import EncoderRequestData
from .scheduler import Scheduler
from .types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

__all__ = [
    # Types
    "SchedulerRequest",
    "SchedulerStatus",
    "SchedulerOutput",
    "RequestOutput",
    "ModelRunnerOutput",
    # Core components
    "Scheduler",
    "ModelRunner",
    "OmniEngine",
    # Encoder
    "EncoderRequestData",
    "create_encoder_engine",
    # AR (Simple)
    "ARRequestData",
    "create_ar_engine",
    # AR (SGLang)
    "SGLangARRequestData",
    "create_sglang_ar_engine",
    # Talker
    "create_talker_codec_engine",
]


_LAZY_EXPORTS = {
    "SGLangARRequestData": ".runtime.sglang_ar",
}


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is not None:
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
