# SPDX-License-Identifier: Apache-2.0
"""Runtime module - model-type-specific batching and I/O logic."""

from .ar import (
    ARBatchData,
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARRequestData,
    ARResourceManager,
)
from .common import (
    EosIterationController,
    SimpleResourceManager,
    SinglePassIterationController,
)
from .encoder import (
    EncoderBatchData,
    EncoderBatchPlanner,
    EncoderInputPreparer,
    EncoderOutputProcessor,
    EncoderRequestData,
)
from .interfaces import (
    BatchPlanner,
    CacheManager,
    InputPreparer,
    IterationController,
    OutputProcessor,
    ResourceManager,
)

_SGLANG_EXPORTS = {
    "SGLangARRequestData",
    "SGLangBatchPlanner",
    "SGLangResourceManager",
    "SGLangOutputProcessor",
    "SGLangIterationController",
    "SGLangModelRunner",
}

_TALKER_EXPORTS = {
    "TalkerARRequestData",
    "TalkerBatchPlanner",
    "TalkerResourceManager",
    "TalkerOutputProcessor",
    "TalkerIterationController",
    "TalkerModelRunner",
}

__all__ = [
    # Protocols
    "BatchPlanner",
    "CacheManager",
    "ResourceManager",
    "IterationController",
    "InputPreparer",
    "OutputProcessor",
    "SimpleResourceManager",
    "SinglePassIterationController",
    "EosIterationController",
    # Encoder
    "EncoderRequestData",
    "EncoderBatchData",
    "EncoderBatchPlanner",
    "EncoderInputPreparer",
    "EncoderOutputProcessor",
    # AR (Simple)
    "ARRequestData",
    "ARBatchData",
    "ARBatchPlanner",
    "ARResourceManager",
    "ARInputPreparer",
    "AROutputProcessor",
    # AR (SGLang)
    "SGLangARRequestData",
    "SGLangBatchPlanner",
    "SGLangResourceManager",
    "SGLangOutputProcessor",
    "SGLangIterationController",
    "SGLangModelRunner",
    # Talker
    "TalkerARRequestData",
    "TalkerBatchPlanner",
    "TalkerResourceManager",
    "TalkerOutputProcessor",
    "TalkerIterationController",
    "TalkerModelRunner",
]


def __getattr__(name: str):
    if name in _SGLANG_EXPORTS:
        from . import sglang_ar

        return getattr(sglang_ar, name)
    if name in _TALKER_EXPORTS:
        from . import sglang_talker

        return getattr(sglang_talker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
