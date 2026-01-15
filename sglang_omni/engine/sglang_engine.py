# SPDX-License-Identifier: Apache-2.0
"""SGLang Engine implementation for SGLang-Omni."""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from sglang_omni.engine.base import Engine

logger = logging.getLogger(__name__)


# Type alias for input processor
InputProcessor = Callable[[str, Any], "GenerateReqInput"]


def default_input_processor(request_id: str, data: Any, **kwargs) -> "GenerateReqInput":
    """Convert input data to GenerateReqInput.
    
    Args:
        request_id: Request ID
        data: Input data (str, list, or dict)
        **kwargs: Additional fields (sampling_params, return_hidden_states, etc.)
    
    Returns:
        GenerateReqInput object ready for tokenizer_manager.generate_request()
    """
    from sglang.srt.managers.io_struct import GenerateReqInput
    
    req_kwargs = {"rid": request_id, **kwargs}
    
    if isinstance(data, str):
        req_kwargs["text"] = data
    elif isinstance(data, list):
        req_kwargs["input_ids"] = data
    elif isinstance(data, dict):
        # Text
        if "prompt" in data:
            req_kwargs["text"] = data["prompt"]
        elif "text" in data:
            req_kwargs["text"] = data["text"]
        
        # Token IDs
        if "input_ids" in data:
            req_kwargs["input_ids"] = data["input_ids"]
        
        # Embeddings (from previous stage)
        embeds = data.get("prompt_embeds") or data.get("hidden_states") or data.get("input_embeds")
        if embeds is not None:
            if hasattr(embeds, "tolist"):
                req_kwargs["input_embeds"] = embeds.tolist()
            else:
                req_kwargs["input_embeds"] = embeds
        
        # Multimodal
        for key in ["image_data", "audio_data", "video_data"]:
            if key in data and data[key] is not None:
                req_kwargs[key] = data[key]
        
        # Sampling params (merge with kwargs if provided)
        if "sampling_params" in data:
            existing = req_kwargs.get("sampling_params", {})
            req_kwargs["sampling_params"] = {**existing, **data["sampling_params"]}
    else:
        req_kwargs["text"] = str(data)
    
    return GenerateReqInput(**req_kwargs)


class SGLangEngine(Engine):
    """SGLang Engine wrapper implementing the sglang_omni Engine interface.
    
    Accepts all SGLang Engine/ServerArgs parameters plus omni-specific ones.
    
    Omni-specific args:
        input_processor: Transform input data to GenerateReqInput
        return_hidden_states: Request hidden states (requires --enable-return-hidden-states)
        default_sampling_params: Default sampling parameters
    
    SGLang Engine args (passed through to sglang.srt.entrypoints.Engine):
        model_path, tp_size, dtype, quantization, context_length, 
        mem_fraction_static, max_running_requests, disable_radix_cache,
        enable_return_hidden_states, skip_tokenizer_init, trust_remote_code, etc.
        See sglang.srt.server_args.ServerArgs for full list.
    
    Example:
        >>> engine = SGLangEngine(
        ...     model_path="meta-llama/Llama-2-7b-hf",
        ...     tp_size=2,
        ...     dtype="float16",
        ...     enable_return_hidden_states=True,
        ... )
    """
    
    def __init__(
        self,
        # Omni-specific args
        input_processor: Optional[InputProcessor] = None,
        return_hidden_states: bool = True,
        default_sampling_params: Optional[Dict[str, Any]] = None,
        # All SGLang Engine args passed through
        **kwargs,
    ):
        self.input_processor = input_processor or default_input_processor
        self.return_hidden_states = return_hidden_states
        self.default_sampling_params = default_sampling_params or {
            "max_new_tokens": 512,
            "temperature": 0.7,
        }
        
        # SGLang engine kwargs (model_path, tp_size, dtype, etc.)
        self.engine_kwargs = kwargs
        
        # Lazy initialization
        self._sglang_engine = None
        self._initialized = False
        
        # Request tracking
        self._tasks: Dict[str, asyncio.Task] = {}
        self._aborted: set = set()
    
    def _ensure_initialized(self) -> None:
        """Lazily initialize the SGLang engine."""
        if self._initialized:
            return
        
        try:
            from sglang.srt.entrypoints.engine import Engine as SGLangRuntimeEngine
        except ImportError:
            raise ImportError(
                "SGLang is required. Install with: pip install sglang"
            )
        
        model_path = self.engine_kwargs.get("model_path", "unknown")
        logger.info(f"Initializing SGLang Engine with model: {model_path}")
        
        self._sglang_engine = SGLangRuntimeEngine(**self.engine_kwargs)
        self._initialized = True
        
        logger.info("SGLang Engine initialized successfully")
    
    # ==================== Engine Interface Implementation ====================
    
    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request to SGLang engine."""
        if request_id in self._aborted:
            return
        
        self._ensure_initialized()
        
        # Build GenerateReqInput
        req_input = self.input_processor(
            request_id, data,
            sampling_params=self.default_sampling_params,
            return_hidden_states=self.return_hidden_states,
        )
        
        # Start generation task - returns raw SGLang output
        async def generate():
            generator = self._sglang_engine.tokenizer_manager.generate_request(req_input, None)
            return await generator.__anext__()
        
        self._tasks[request_id] = asyncio.create_task(generate())
    
    async def get_result(self, request_id: str) -> Any:
        """Get result (blocks until ready)."""
        if request_id in self._aborted:
            raise asyncio.CancelledError(f"Request {request_id} was aborted")
        
        task = self._tasks.get(request_id)
        if task is None:
            raise ValueError(f"Request {request_id} not found")
        
        try:
            return await task
        finally:
            self._tasks.pop(request_id, None)
    
    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        self._aborted.add(request_id)
        
        # Cancel local task
        task = self._tasks.pop(request_id, None)
        if task and not task.done():
            task.cancel()
        
        # Abort via tokenizer_manager (same as normal sglang)
        if self._initialized and self._sglang_engine is not None:
            self._sglang_engine.tokenizer_manager.abort_request(request_id)
    
    def transform_output(
        self, request_id: str, output: Any, next_stage: str | None
    ) -> Any:
        """Transform SGLang raw output for next stage.
        
        SGLang output format:
            {"text": "...", "output_ids": [...], "meta_info": {"hidden_states": [...], ...}}
        
        Args:
            request_id: Request ID
            output: Raw SGLang output from get_result
            next_stage: Name of next stage, or None if final output
        """
        if next_stage is None:
            return output
        
        if not isinstance(output, dict):
            return output
        
        # Build input for next stage
        next_input = {"request_id": request_id}
        meta_info = output.get("meta_info", {})
        
        # hidden_states -> prompt_embeds for next stage
        if "hidden_states" in meta_info:
            next_input["prompt_embeds"] = meta_info["hidden_states"]
        
        # output_ids -> input_ids for next stage
        if "output_ids" in output and output["output_ids"]:
            next_input["input_ids"] = output["output_ids"]
        
        # text -> previous_output_text
        if "text" in output:
            next_input["previous_output_text"] = output["text"]
        
        return next_input
    
    # ==================== Lifecycle ====================
    
    def shutdown(self) -> None:
        """Shutdown the SGLang engine."""
        if self._sglang_engine is not None:
            try:
                self._sglang_engine.shutdown()
            except Exception as e:
                logger.warning(f"Error during engine shutdown: {e}")
            self._sglang_engine = None
            self._initialized = False
            logger.info("SGLang Engine shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass

