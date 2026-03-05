# Fix: Pass image tokens to SGLang thinker

## Problem

When only the thinker is enabled (talker disabled), the model's response is completely
unrelated to the input image. After the migration to the SGLang model runner, two critical
data paths were broken:

1. **Image embeddings not injected**: `model_inputs["image_embeds"]` was stored in
   `SGLangARRequestData.model_inputs` but never consumed by the SGLang execution path.
   The model received `input_ids` with image placeholder tokens (e.g., token 151655) but
   no actual image features — `embed_tokens()` produced generic text embeddings for them.

2. **M-RoPE positions missing**: Qwen3-Omni uses 3D spatial positions `[temporal, height, width]`
   for image tokens via `MRotaryEmbedding`. The SGLang `Req.multimodal_inputs` was never set,
   so `mrope_positions` was `None` and the model fell back to sequential 1D positions.

## Changes

### `sglang_omni/models/qwen3_omni/pipeline/engine_io.py`

- Added `_compute_mrope_positions()`: computes M-RoPE 3D positions from `input_ids` and
  `image_grid_thw`/`video_grid_thw` using SGLang's `MRotaryEmbedding.get_rope_index()`.
  All tensor inputs are moved to CPU before calling (the upstream function creates CPU
  intermediates internally).

- Modified `build_sglang_thinker_request()`:
  - Accepts new `thinker_config` parameter.
  - Computes mrope positions and sets `req.multimodal_inputs` with a `MultimodalInputs`
    dataclass containing `mrope_positions` and `mrope_position_delta`.
  - Attaches `model_inputs` dict to `req.omni_model_inputs` for later use by the model runner.

### `sglang_omni/engines/omni/runtime/sglang_ar.py`

- Added `SGLangModelRunner._inject_multimodal_embeds()`: during prefill (extend), for each
  request with `omni_model_inputs`:
  - Computes text embeddings via `embed_tokens(input_ids)`.
  - Scatters image/video/audio embeddings into placeholder positions using token ID matching.
  - Handles chunked prefill via per-modality consumed-offset tracking.
  - Navigates the model hierarchy (`model.thinker.model.embed_tokens`) to find the embedding
    layer, since the SGLang model is `Qwen3OmniMoeForConditionalGeneration` (a wrapper).
  - Cleans up `omni_model_inputs` after the final prefill chunk.
  - Stores result as `forward_batch.omni_input_embeds` (custom attribute, NOT
    `forward_batch.input_embeds` which would trigger SGLang's kwarg path).

- Added `SGLangModelRunner._forward_with_omni_embeds()`: bypasses the outer
  `Qwen3VLForConditionalGeneration.forward()` (which doesn't accept `input_embeds`) and
  calls the inner language model directly:
  - Inits attention backend metadata.
  - Uses mrope positions if enabled.
  - Calls `language_model(input_ids=None, input_embeds=..., forward_batch=..., positions=...)`.
  - Runs `logits_processor` to produce output logits.

- Modified `SGLangModelRunner.execute()`: after `ForwardBatch.init_new()`, calls
  `_inject_multimodal_embeds()` for extend batches. If multimodal embeds were injected,
  routes through `_forward_with_omni_embeds()` instead of the standard path.

### `sglang_omni/models/qwen3_omni/pipeline/stages.py`

- Modified `create_sglang_thinker_executor()`: loads `thinker_config` via
  `load_thinker_config()` and passes it to `build_sglang_thinker_request()`.

### `sglang_omni/models/qwen3_omni/thinker.py`

- Modified `Qwen3OmniMoeThinkerTextModel.forward()`: reads `omni_input_embeds` and
  deepstack data from `forward_batch` if not passed directly as kwargs (for future use
  when the custom model is registered).

- Modified `_deepstack_process()`: handles both 1D boolean masks (SGLang path) and
  multi-dim masks (HF path).

## Why `_forward_with_omni_embeds()` bypass is necessary

### What we tried

We attempted to eliminate the custom `_forward_with_omni_embeds()` bypass (plan.md Step 6)
by using SGLang's standard `forward_extend` path:

1. Changed `forward_batch.omni_input_embeds = input_embeds` → `forward_batch.input_embeds = input_embeds`
2. Deleted `_forward_with_omni_embeds()` entirely
3. Removed the `has_omni_embeds` branching in `execute()` — always used
   `self.model_worker.forward_batch_generation(forward_batch)`
4. Removed the `omni_input_embeds` fallback in `Qwen3OmniMoeThinkerTextModel.forward()`

The rationale was that SGLang's `forward_extend()` already handles `forward_batch.input_embeds`:
```python
if forward_batch.input_embeds is not None:
    kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
return self.model.forward(input_ids, positions, forward_batch, **kwargs)
```

And `plan.md` claimed `self.model` IS `Qwen3OmniMoeThinkerTextModel` (the inner language
model, `EntryClass` at `thinker.py:820`), which accepts `input_embeds` in its `forward()`.

### Why it failed

At runtime, `self.model` is **NOT** `Qwen3OmniMoeThinkerTextModel`. It is
`Qwen3VLForConditionalGeneration` — SGLang's built-in outer wrapper model.

The `EntryClass = Qwen3OmniMoeThinkerTextModel` in `thinker.py` is used by our custom
pipeline config registry (`sglang_omni/models/registry.py`), **not** by SGLang's model
loading. SGLang resolves the model class from the HuggingFace config's `architectures`
field via its own model registry (`sglang.srt.models.registry.ModelRegistry`).

Verified at runtime:
```
ModelRegistry.models["Qwen3VLForConditionalGeneration"]
  → sglang.srt.models.qwen3_vl.Qwen3VLForConditionalGeneration
```

`Qwen3VLForConditionalGeneration.forward()` does NOT accept `input_embeds` as a keyword
argument, so SGLang's `forward_extend()` fails with:
```
TypeError: Qwen3VLForConditionalGeneration.forward() got an unexpected keyword argument 'input_embeds'
```

### Why the bypass must stay

The custom `_forward_with_omni_embeds()` navigates past the outer wrapper to call the inner
language model directly (`outer.model`), which does accept `input_embeds`. This is the only
way to inject pre-merged multimodal embeddings without modifying SGLang's built-in model
classes or re-registering a different model class in SGLang's registry.

To properly eliminate the bypass in the future, we would need to either:
- Register `Qwen3OmniMoeThinkerTextModel` as the SGLang model class (requires overriding
  SGLang's model registry resolution for this architecture), or
- Modify `Qwen3VLForConditionalGeneration.forward()` upstream to accept `input_embeds`

### Trade-offs of the current bypass

- `can_run_cuda_graph=False` — disables CUDA graph for multimodal prefills
- Skips `piecewise_cuda_graph_runner.can_run()` / `replay()` path
- Skips potential `torch.compile` optimizations
- Manually reimplements `attn_backend.init_forward_metadata()` and logits processing

These are acceptable since multimodal prefills are a small fraction of total forward passes
(only the first step per request), and decode steps use the standard path with CUDA graph.

## Known Limitations

- **Deepstack not yet supported in SGLang path**: the per-layer visual feature injection
  (`deepstack_visual_embeds`) is collected but not passed through to the upstream language
  model during the custom forward. The upstream model expects a different tensor format
  (`[num_visual_tokens, hidden_size * num_layers]`) and needs extend-chunk-aware slicing.
  This is an optimization that improves quality but is not required for basic image understanding.
