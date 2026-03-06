# Plan: Fix Missing Deepstack Intermediate-Layer Visual Injection in SGLang Thinker

## 1. Background: How Qwen3-Omni's Thinker Expects Visual Data

Qwen3-Omni uses a two-channel mechanism to feed visual information into the thinker (language model):

**Channel A — Input Embedding Replacement (Layer 0)**:
The preprocessor tokenizes video frames into placeholder tokens (`<video>`, token ID 151656). During prefill, `_inject_multimodal_embeds()` replaces these placeholders with the actual `video_embeds` from the image encoder. This gives the model visual information at the input embedding layer only.

**Channel B — Deepstack Intermediate-Layer Injection (Layers 0, 1, 2, ...)**:
The image encoder also produces `deepstack_visual_embeds` — a list of per-layer visual feature tensors. These are meant to be injected as additive residuals at intermediate transformer layers (controlled by `deepstack_visual_indexes` in the model config, typically layers 0, 1, 2). This multi-layer injection is how Qwen3-Omni is architecturally designed to process vision — the model receives progressively refined visual features at multiple depths.

In the **native SGLang path** (when SGLang runs Qwen3-Omni directly without the sglang_omni pipeline), both channels work correctly. The function `general_mm_embed_routine()` in `sglang/srt/managers/mm_utils.py` handles this:

```python
# From general_mm_embed_routine (native SGLang):
input_embeds, other_info = embed_mm_inputs(...)    # Channel A
if use_deepstack:
    kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]  # Channel B

hidden_states = language_model(
    input_ids=None,
    forward_batch=forward_batch,
    input_embeds=input_embeds,
    **kwargs,  # ← input_deepstack_embeds reaches the language model here
)
```

Inside the language model (`Qwen3MoeLLMModel.forward()`), the deepstack data is consumed per-layer:

```python
# Qwen3MoeLLMModel.forward():
for layer_idx, layer in enumerate(self.layers[...]):
    deepstack_embeds = self.get_deepstack_embeds(layer_idx - 1, input_deepstack_embeds)
    hidden_states, residual = layer(
        positions, hidden_states, forward_batch, residual,
        post_residual_addition=deepstack_embeds,  # ← added to residual at this layer
    )
```

And `get_deepstack_embeds` slices the concatenated tensor:

```python
def get_deepstack_embeds(self, layer_idx, input_deepstack_embeds):
    if input_deepstack_embeds is None:
        return None
    sep = self.hidden_size * layer_idx
    return input_deepstack_embeds[:, sep : sep + self.hidden_size]
```

The expected format of `input_deepstack_embeds` is a **2D tensor** of shape `[seq_len, hidden_size * num_deepstack_layers]`. Layer *i*'s features occupy columns `[hidden_size*i : hidden_size*(i+1)]`. Non-visual token positions are zeros.

## 2. Current State: Channel B Is Broken

In the sglang_omni pipeline, `_forward_with_omni_embeds()` in `sglang_omni/engines/omni/runtime/sglang_ar.py` bypasses `general_mm_embed_routine` and calls the language model directly. It correctly implements Channel A (embedding replacement), but **Channel B is broken**.

The original code (commit `d609c32`, before any fix):

```python
def _forward_with_omni_embeds(self, forward_batch, input_embeds,
                               deepstack_visual_embeds=None,    # ← RECEIVED from upstream
                               visual_pos_masks=None):          # ← RECEIVED from upstream
    ...
    hidden_states = outer.model(
        input_ids=None,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        # input_deepstack_embeds is NEVER passed
        # deepstack_visual_embeds is silently dropped
    )
```

The upstream pipeline correctly produces deepstack data:
- Image encoder (`components/image_encoder.py`) produces `deepstack_visual_embeds_video` (line 105)
- Merge function (`pipeline/merge.py`) packs it into `thinker_inputs["model_inputs"]`
- `_inject_multimodal_embeds()` extracts it and returns it as the second element of the tuple

But `_forward_with_omni_embeds` receives this data and discards it. The language model's `get_deepstack_embeds()` always returns `None`, and `post_residual_addition` is never applied at any intermediate layer.

**Result**: The model only sees visual information at layer 0 (via flat embedding replacement). All intermediate-layer deepstack injection is lost.

## 3. The Fix

### 3.1 What Needs to Change

Only one method needs modification: `_forward_with_omni_embeds()` in `sglang_omni/engines/omni/runtime/sglang_ar.py`.

The fix has two parts:

**Part A — Format conversion**: Convert the pipeline's per-layer list of tensors into the concatenated 2D tensor that SGLang's `get_deepstack_embeds` expects.

- Input from pipeline: `deepstack_visual_embeds` = list of `N` tensors, each `[num_visual_tokens, hidden_size]`
- Required output: `input_deepstack_embeds` = tensor `[seq_len, hidden_size * N]`
- `visual_pos_masks` (boolean tensor `[seq_len]`) indicates which positions in the full sequence are visual tokens
- Non-visual positions must be zero

**Part B — Passthrough**: Pass the converted tensor as `input_deepstack_embeds` to `outer.model()`.

### 3.2 Detailed Steps

1. Read the current `_forward_with_omni_embeds` method in `sglang_omni/engines/omni/runtime/sglang_ar.py`.

2. Between the `positions` computation and the `outer.model()` call, add:
   - Guard: if `deepstack_visual_embeds is None` or `visual_pos_masks is None`, set `ds_input = None`.
   - Otherwise:
     a. Move each per-layer tensor to the correct device/dtype (matching `input_embeds`).
     b. Concatenate along `dim=-1` → shape `[num_visual_tokens, hidden_size * num_layers]`.
     c. Create a zero tensor of shape `[seq_len, hidden_size * num_layers]` where `seq_len = input_embeds.shape[0]`.
     d. Index-assign: `full_ds[visual_pos_masks] = concatenated_ds`.
     e. Set `ds_input = full_ds`.

3. Add `input_deepstack_embeds=ds_input` to the `outer.model()` call.

4. Do NOT change any other method or file. The upstream pipeline and the language model already have the correct interfaces — only this last-mile passthrough is missing.

### 3.3 Format Reference

For verification, here is how the native SGLang model consumes the tensor:

```python
# Qwen3MoeLLMModel:
self.deepstack_embed_to_decoder_layer = range(3)  # layers 0, 1, 2

def get_deepstack_embeds(self, layer_idx, input_deepstack_embeds):
    if input_deepstack_embeds is None:
        return None
    if layer_idx not in self.deepstack_embed_to_decoder_layer:
        return None
    sep = self.hidden_size * layer_idx
    return input_deepstack_embeds[:, sep : sep + self.hidden_size]
    # Returns [seq_len, hidden_size] for the given layer
```

This returned tensor is then passed as `post_residual_addition` to the transformer layer, where it is added to the residual stream inside `LayerCommunicator.prepare_attn()` → `RMSNorm(hidden_states, residual, post_residual_addition)`.

## 4. Code Style Requirements

Follow the code style agent rules (see `/data/chenyang/.claude/agents/code-style-agent.md`):

- **P0 Correctness**: Guard against `None` inputs. Do not wrap in broad try/except.
- **P1 Performance**: The conversion happens once per prefill step (not per decode step), so it is not on the hottest path. Still, avoid unnecessary copies — use `.to(device=..., dtype=...)` in a single call, not separate `.to(device)` then `.to(dtype)`.
- **P2 Maintainability**: Add a brief comment explaining the format conversion (why concatenate along dim=-1, why zero-pad to seq_len). Keep it under 20 lines of new code.
- **P3 Style**: Match the existing code style in `sglang_ar.py` — no type annotations beyond what's already there, use `| None` not `Optional`, consistent variable naming with the surrounding code.

## 5. Validation

The fix must pass the existing integration test:

```bash
python tests/test_video_integration.py
```

This test:
- Starts the sglang-omni server with `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Sends a video (`test_file.webm`) + text prompt to `/v1/chat/completions`
- Round 1: "Where am I right now?" → expects keywords like station, gate, 12, UCI, etc.
- Round 2: "Is there a specific school shown in the video?" → expects "UCI" in response
- Checks server stability (no crash, health endpoint OK)

The test should continue to pass. The model's output may improve in visual detail (due to deepstack being active), but the keyword assertions should still match.

**Do NOT modify the test file.**

## 6. Files to Modify

| File | Change |
|------|--------|
| `sglang_omni/engines/omni/runtime/sglang_ar.py` | Add deepstack format conversion + passthrough in `_forward_with_omni_embeds` |

No other files need modification.
