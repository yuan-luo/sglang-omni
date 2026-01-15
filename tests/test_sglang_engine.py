# SPDX-License-Identifier: Apache-2.0
"""Tests for SGLangEngine."""

import asyncio
import pytest
from typing import Any, Callable, Dict, Optional

from sglang_omni.engine.base import Engine
from sglang_omni.engine.sglang_engine import default_input_processor


class MockSGLangEngine(Engine):
    """Mock SGLang Engine for testing without actual model loading.
    
    Returns raw SGLang output format:
        {"text": "...", "output_ids": [...], "meta_info": {"hidden_states": [...], ...}}
    """
    
    def __init__(
        self,
        mock_response: Optional[Callable[[str, Any], Dict]] = None,
        delay: float = 0.0,
    ):
        self.mock_response = mock_response or self._default_mock_response
        self.delay = delay
        self._results: Dict[str, Any] = {}
        self._aborted: set = set()
    
    def _default_mock_response(self, request_id: str, data: Any) -> Dict:
        """Generate a default mock response (raw SGLang format)."""
        return {
            "text": f"Mock response for {request_id}",
            "output_ids": [1, 2, 3],
            "meta_info": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "finish_reason": {"type": "stop"},
                "hidden_states": [[0.1, 0.2, 0.3] * 10],
            },
        }
    
    async def add_request(self, request_id: str, data: Any) -> None:
        if request_id in self._aborted:
            return
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if request_id in self._aborted:
            return
        
        # Store raw output (like real SGLangEngine)
        self._results[request_id] = self.mock_response(request_id, data)
    
    async def get_result(self, request_id: str) -> Any:
        if request_id in self._aborted:
            raise asyncio.CancelledError(f"Request {request_id} was aborted")
        return self._results.pop(request_id)
    
    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        self._results.pop(request_id, None)
    
    def transform_output(
        self, request_id: str, output: Any, next_stage: str | None
    ) -> Any:
        """Transform raw SGLang output for next stage."""
        if next_stage is None:
            return output
        
        if not isinstance(output, dict):
            return output
        
        next_input = {"request_id": request_id}
        meta_info = output.get("meta_info", {})
        
        if "hidden_states" in meta_info:
            next_input["prompt_embeds"] = meta_info["hidden_states"]
        
        if "output_ids" in output and output["output_ids"]:
            next_input["input_ids"] = output["output_ids"]
        
        if "text" in output:
            next_input["previous_output_text"] = output["text"]
        
        return next_input


# ==================== Tests for default_input_processor ====================

class TestDefaultInputProcessor:
    """Tests for default_input_processor."""
    
    def test_string_input(self):
        """String input should become text field."""
        try:
            result = default_input_processor("req_1", "Hello world", sampling_params={})
            assert result.text == "Hello world"
            assert result.rid == "req_1"
        except ImportError:
            pytest.skip("SGLang not installed")
    
    def test_list_input(self):
        """List input should become input_ids field."""
        try:
            result = default_input_processor("req_1", [1, 2, 3], sampling_params={})
            assert result.input_ids == [1, 2, 3]
        except ImportError:
            pytest.skip("SGLang not installed")
    
    def test_dict_with_prompt(self):
        """Dict with 'prompt' should map to text."""
        try:
            result = default_input_processor("req_1", {"prompt": "Hello"}, sampling_params={})
            assert result.text == "Hello"
        except ImportError:
            pytest.skip("SGLang not installed")
    
    def test_dict_with_hidden_states(self):
        """Dict with hidden_states should map to input_embeds."""
        try:
            embeds = [[0.1, 0.2, 0.3]]
            result = default_input_processor("req_1", {"hidden_states": embeds}, sampling_params={})
            assert result.input_embeds == embeds
        except ImportError:
            pytest.skip("SGLang not installed")
    
    def test_dict_with_multimodal(self):
        """Dict with multimodal data should pass through."""
        try:
            result = default_input_processor(
                "req_1", 
                {"prompt": "Describe", "image_data": "path/to/img.jpg"},
                sampling_params={}
            )
            assert result.text == "Describe"
            assert result.image_data == "path/to/img.jpg"
        except ImportError:
            pytest.skip("SGLang not installed")


# ==================== Tests for MockSGLangEngine ====================

class TestMockSGLangEngine:
    """Tests for MockSGLangEngine."""
    
    @pytest.mark.asyncio
    async def test_basic_flow(self):
        """Test basic add_request -> get_result flow."""
        engine = MockSGLangEngine()
        
        await engine.add_request("req_1", "Hello")
        result = await engine.get_result("req_1")
        
        # Result is raw SGLang format
        assert "text" in result
        assert "output_ids" in result
        assert "meta_info" in result
    
    @pytest.mark.asyncio
    async def test_abort(self):
        """Test abort functionality."""
        engine = MockSGLangEngine(delay=1.0)
        
        task = asyncio.create_task(engine.add_request("req_1", "Hello"))
        await asyncio.sleep(0.1)
        await engine.abort("req_1")
        await task
        
        with pytest.raises(asyncio.CancelledError):
            await engine.get_result("req_1")
    
    @pytest.mark.asyncio
    async def test_transform_output_for_next_stage(self):
        """Test transform_output converts raw output for next stage."""
        engine = MockSGLangEngine()
        
        await engine.add_request("req_1", "Hello")
        raw_output = await engine.get_result("req_1")
        
        # Transform for next stage
        next_input = engine.transform_output("req_1", raw_output, "next_stage")
        
        assert next_input["request_id"] == "req_1"
        assert "prompt_embeds" in next_input  # from hidden_states
        assert "input_ids" in next_input  # from output_ids
        assert "previous_output_text" in next_input  # from text
    
    @pytest.mark.asyncio
    async def test_transform_output_final(self):
        """Test transform_output returns raw output when next_stage is None."""
        engine = MockSGLangEngine()
        
        await engine.add_request("req_1", "Hello")
        raw_output = await engine.get_result("req_1")
        
        # No transformation for final output
        final_output = engine.transform_output("req_1", raw_output, None)
        
        assert final_output == raw_output
    
    @pytest.mark.asyncio
    async def test_multiple_requests(self):
        """Test handling multiple concurrent requests."""
        engine = MockSGLangEngine()
        
        await asyncio.gather(
            engine.add_request("req_1", "Hello"),
            engine.add_request("req_2", "World"),
            engine.add_request("req_3", "Test"),
        )
        
        results = await asyncio.gather(
            engine.get_result("req_1"),
            engine.get_result("req_2"),
            engine.get_result("req_3"),
        )
        
        assert len(results) == 3
        assert all("text" in r for r in results)


# ==================== Integration-like tests ====================

class TestEngineInterface:
    """Test that engines implement the Engine interface correctly."""
    
    @pytest.mark.asyncio
    async def test_mock_implements_interface(self):
        """MockSGLangEngine should implement Engine interface."""
        engine = MockSGLangEngine()
        
        assert hasattr(engine, 'add_request')
        assert hasattr(engine, 'get_result')
        assert hasattr(engine, 'abort')
        assert hasattr(engine, 'transform_output')
