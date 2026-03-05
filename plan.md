# Plan: Fix Image Tokens Not Passed to SGLang Thinker

## Problem Statement

When only the thinker is enabled (talker disabled), the model's response is completely
unrelated to the input image. Image tokens are not being passed to the thinker after the
recent migration to the SGLang model runner.

## Root Cause Analysis

### Working path (HF-based thinker)

```
Preprocessor → Image Encoder → merge_for_thinker() → build_thinker_request()
  → ARRequestData(model_inputs={"image_embeds": tensor, "deepstack_visual_embeds": [...]})
  → ARBatchPlanner.build_batch() → ARBatchData(model_inputs=...)
  → ARInputPreparer.prepare() → passes model_inputs as **kwargs to model.forward()
  → Qwen3OmniSplitThinker.forward(image_embeds=tensor, ...) → merges into input_embeds
```

The HF path properly passes `image_embeds`, `deepstack_visual_embeds`, `image_grid_thw`,
etc. through `ARRequestData.model_inputs` → `ARBatchData.model_inputs` →
`ARInputPreparer.prepare()` → model `forward(**kwargs)`.

### Broken path (SGLang-based thinker)

```
Preprocessor → Image Encoder → merge_for_thinker() → build_sglang_thinker_request()
  → SGLangARRequestData(model_inputs={"image_embeds": tensor, ...})
  → SGLangBatchPlanner.select_requests() → prefill_manager.add_one_request(data.req)
    ← data.req is a plain SGLang Req with only origin_input_ids, NO image data!
  → SGLangModelRunner.execute()
    → schedule_batch.get_model_worker_batch() → ForwardBatch.init_new()
    → model_worker.forward_batch_generation(forward_batch)
    → Qwen3OmniMoeThinkerTextModel.forward(input_ids, positions, forward_batch)
      ← receives NO image_embeds, NO input_embeds, NO multimodal_inputs
      ← falls back to: hidden_states = self.embed_tokens(input_ids)
      ← image placeholder tokens get generic text embeddings, not image features!
```

**Two critical data paths are broken:**

1. **Image embeddings**: `model_inputs["image_embeds"]` is stored in
   `SGLangARRequestData.model_inputs` but never read by the SGLang execution path.
   The SGLang `Req` object has no awareness of these embeddings. The model receives
   `input_ids` with image placeholder tokens (e.g., token_id=151655) but no actual
   image features to scatter into those positions.

2. **M-RoPE positions**: Qwen3-Omni uses Multi-dimensional Rotary Position Embeddings
   (`MRotaryEmbedding`) where image tokens need 3D spatial positions `[temporal, height,
   width]` instead of sequential 1D positions. SGLang computes these from
   `Req.multimodal_inputs.mrope_positions`, which is `None` because
   `Req.multimodal_inputs` is never set. The model falls back to sequential positions,
   producing wrong attention patterns even if embeddings were somehow injected.

### Key files involved

| File | Role |
|------|------|
| `sglang_omni/models/qwen3_omni/pipeline/engine_io.py` | `build_sglang_thinker_request()` - builds the SGLang Req |
| `sglang_omni/engines/omni/runtime/sglang_ar.py` | `SGLangModelRunner.execute()` - runs the forward pass |
| `sglang_omni/models/qwen3_omni/thinker.py` | `Qwen3OmniMoeThinkerTextModel` - the SGLang model |
| `sglang_omni/engines/omni/factory.py` | `create_sglang_ar_engine()` - assembles the engine |
| `sglang_omni/models/qwen3_omni/pipeline/merge.py` | `build_thinker_inputs()` - prepares model_inputs dict |
| `sglang_omni/models/qwen3_omni/components/thinker.py` | `Qwen3OmniSplitThinker` - HF thinker with merge logic |

## Proposed Fix

### Strategy: Inject pre-merged input_embeds + M-RoPE positions via SGLang's existing mechanisms

SGLang already supports:
- `Req.input_embeds` → `ForwardBatch.input_embeds` → passed as `input_embeds` kwarg to model
- `Req.multimodal_inputs` → `ForwardBatch.mm_inputs` → used to compute `mrope_positions`

We will use both mechanisms to pass image data through the SGLang pipeline.

### Step 1: Compute merged input_embeds in `build_sglang_thinker_request()`

**File: `sglang_omni/models/qwen3_omni/pipeline/engine_io.py`**

In `build_sglang_thinker_request()`, after constructing the `Req`:

1. Access the model's `embed_tokens` layer (or replicate the embedding merge logic)
2. Compute `inputs_embeds = embed_tokens(input_ids)`
3. Scatter `image_embeds` into positions where `input_ids == image_token_id`
4. Set `req.input_embeds = merged_embeds.tolist()` (SGLang expects `List[List[float]]`)

**Problem**: This approach is memory-inefficient (`List[List[float]]` for thousands of
tokens x 3584 hidden_dim) and requires access to the model's embed_tokens layer which
lives inside the SGLang ModelRunner on GPU.

**Better alternative**: Store model_inputs on the `Req` object and do the merge inside
`SGLangModelRunner.execute()` where we have access to the model.

### Step 2: Attach model_inputs to Req and merge in SGLangModelRunner (PREFERRED)

**File: `sglang_omni/models/qwen3_omni/pipeline/engine_io.py`**

```python
# In build_sglang_thinker_request():
req.omni_model_inputs = model_inputs  # custom attribute on Req
```

**File: `sglang_omni/engines/omni/runtime/sglang_ar.py`**

In `SGLangModelRunner.execute()`, after creating `ForwardBatch` but before the forward
pass, for extend (prefill) batches:

```python
# Collect omni_model_inputs from batch requests
omni_inputs = {}
for req in schedule_batch.reqs:
    inputs = getattr(req, 'omni_model_inputs', None)
    if inputs:
        omni_inputs[req.rid] = inputs

if omni_inputs and schedule_batch.forward_mode.is_extend():
    # Get embed_tokens from the model
    model = self.model_worker.model_runner.model
    embed_tokens = model.embed_tokens

    # Compute text embeddings
    input_embeds = embed_tokens(forward_batch.input_ids)

    # For each request, scatter image/audio/video embeds
    # (use per-request offsets from schedule_batch to map positions)
    merged_embeds = _merge_multimodal_embeds(
        input_ids=forward_batch.input_ids,
        input_embeds=input_embeds,
        omni_inputs=omni_inputs,
        schedule_batch=schedule_batch,
        config=model.config,  # for image_token_id, etc.
    )
    forward_batch.input_embeds = merged_embeds

    # Also store deepstack data on forward_batch for the model to use
    forward_batch.omni_deepstack = _collect_deepstack(omni_inputs, schedule_batch)
```

### Step 3: Compute and attach M-RoPE positions

**File: `sglang_omni/models/qwen3_omni/pipeline/engine_io.py`**

In `build_sglang_thinker_request()`, compute mrope_positions from `image_grid_thw`:

```python
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

# Compute mrope_positions [3, seq_len] from input_ids and image_grid_thw
mrope_positions = MRotaryEmbedding.get_next_input_positions(
    input_ids=input_ids_list,
    image_grid_thw=model_inputs.get("image_grid_thw"),
    video_grid_thw=model_inputs.get("video_grid_thw"),
    attention_mask=...,
)

# Create MultimodalInputs and attach to Req
from sglang.srt.managers.mm_utils import MultimodalInputs
mm_inputs = MultimodalInputs(mm_items=[])
mm_inputs.mrope_positions = mrope_positions
req.multimodal_inputs = mm_inputs
```

This ensures `ForwardBatch._compute_mrope_positions()` gets the correct 3D positions.

### Step 4: Handle deepstack_visual_embeds in the model

**File: `sglang_omni/models/qwen3_omni/thinker.py`**

Modify `Qwen3OmniMoeThinkerTextModel.forward()` to read deepstack data from
`forward_batch`:

```python
def forward(self, input_ids, positions, forward_batch, input_embeds=None, ...):
    if input_embeds is None:
        hidden_states = self.embed_tokens(input_ids)
    else:
        hidden_states = input_embeds

    # Read deepstack from forward_batch (set by SGLangModelRunner)
    deepstack = getattr(forward_batch, 'omni_deepstack', None)
    if deepstack is not None:
        deepstack_visual_embeds = deepstack.get('deepstack_visual_embeds')
        visual_pos_masks = deepstack.get('visual_pos_masks')
    ...
```

### Step 5: Clean up model_inputs after prefill

In `SGLangBatchPlanner._post_step_operations()` or in
`SGLangIterationController.update_request()`, clear `req.omni_model_inputs` after the
first prefill step to free memory (image embeddings are only needed during prefill, not
decode):

```python
if req.omni_model_inputs is not None:
    req.omni_model_inputs = None
```

### Step 6: Remove custom forward bypass — use SGLang's standard `forward_extend` path

**Problem with current implementation (`change.md`):**

The current code introduces `_forward_with_omni_embeds()` which bypasses SGLang's
`forward_extend()` and calls the inner language model directly. This was based on the
incorrect assumption that setting `forward_batch.input_embeds` would pass `input_embeds`
to the **outer** `Qwen3VLForConditionalGeneration.forward()` which doesn't accept it.

**Why this is wrong:**

The SGLang-registered model (`EntryClass` at `thinker.py:820`) is
`Qwen3OmniMoeThinkerTextModel` — the **inner language model**, not the outer wrapper.
Its `forward()` signature (line 632) already accepts `input_embeds`:

```python
def forward(self, input_ids, positions, forward_batch, input_embeds=None, ...)
```

SGLang's `ModelRunner.forward_extend()` does:
```python
if forward_batch.input_embeds is not None:
    kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
return self.model.forward(input_ids, positions, forward_batch, **kwargs)
```

Since `self.model` IS `Qwen3OmniMoeThinkerTextModel`, the `input_embeds` kwarg is
received correctly. There is no outer wrapper in the SGLang path.

**Performance impact of the current bypass:**

- `can_run_cuda_graph=False` — disables CUDA graph for all multimodal prefills
- Skips `piecewise_cuda_graph_runner.can_run()` check and `replay()` path
- Skips potential `torch.compile` optimizations
- Manually reimplements `attn_backend.init_forward_metadata()` and logits processing

**Fix:**

1. In `_inject_multimodal_embeds()`, set `forward_batch.input_embeds = input_embeds`
   instead of `forward_batch.omni_input_embeds`.
2. Delete `_forward_with_omni_embeds()` entirely.
3. In `execute()`, remove the `has_omni_embeds` branching — always use
   `self.model_worker.forward_batch_generation(forward_batch)`.
4. In `Qwen3OmniMoeThinkerTextModel.forward()`, the `input_embeds` kwarg is already
   handled natively — remove the `omni_input_embeds` fallback from `forward_batch`.
5. Keep deepstack data on `forward_batch` as custom attributes (read by the model).

**File changes:**

| File | Change |
|------|--------|
| `sglang_omni/engines/omni/runtime/sglang_ar.py` | Remove `_forward_with_omni_embeds()`, simplify `execute()`, change `omni_input_embeds` → `input_embeds` |
| `sglang_omni/models/qwen3_omni/thinker.py` | Remove `omni_input_embeds` fallback from `forward()` |

## Implementation Order

1. **Step 3 first** (M-RoPE positions) - Compute and attach `mrope_positions` to the
   `Req.multimodal_inputs`. This is the simplest change and addresses position encoding.

2. **Step 2** (Image embedding merge in SGLangModelRunner) - This is the core fix. After
   this step, image features will be properly merged into `input_embeds`.

3. **Step 4** (Deepstack in model) - Handle per-layer visual embeddings. This is needed
   for full correctness of the Qwen3-Omni deepstack mechanism.

4. **Step 5** (Cleanup) - Memory optimization.

5. **Step 6** (Remove custom forward bypass) - Use SGLang's standard `forward_extend`
   path instead of the custom `_forward_with_omni_embeds()`. This re-enables CUDA graph
   for multimodal prefills and removes unnecessary code.
