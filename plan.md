# Talker Engine Refactoring Plan

## Background

The Qwen3-Omni pipeline has two main stages:
- **Thinker**: Auto-regressive text generation. Captures hidden states (embedding layer -1 and layer 24) for relay to talker.
- **Talker**: Receives thinker hidden states, runs backbone forward, then generates 16-layer codec codes (layer-0 via argmax + layers 1-15 via code predictor).

On `main`, the thinker already **reuses the generic AR engine** (`create_sglang_ar_engine()`), with all thinker-specific logic (request building, result processing, hidden state relay) handled at the **executor layer** via `request_builder`, `result_builder`, and `stream_builder` in `stages.py`.

On the `talker_chenyang` branch, the talker was implemented as a **completely separate engine** with 662 lines of new code in `sglang_talker.py`, including:
- `TalkerARRequestData` (custom dataclass)
- `TalkerBatchPlanner` (single-request, no continuous batching)
- `TalkerResourceManager` (counter-based, max 1)
- `TalkerOutputProcessor` (returns codec dict)
- `TalkerIterationController` (single-pass, always finished)
- `TalkerModelRunner` (orchestrates full codec pipeline internally)

Plus a new factory function `create_sglang_talker_engine()` in `factory.py`, lazy exports in `runtime/__init__.py`, and dedicated request/result builders in `engine_io.py`.

## Problem

The talker's `Qwen3OmniTalker.forward()` has the **same signature** as the thinker:
```python
def forward(self, input_ids, positions, forward_batch, inputs_embeds=None) -> hidden_states
```

It is a standard SGLang `nn.Module` using `RadixAttention` and `FusedMoE`. The AR engine's `SGLangModelRunner` can drive this forward pass directly. The separate talker engine duplicates scheduling/resource/iteration infrastructure that already exists in the AR engine path, making the codebase harder to maintain and extend.

## Refactoring Goal

**Delete `sglang_talker.py` entirely.** Make the talker reuse `create_sglang_ar_engine()` like the thinker does, with all talker-specific logic in the executor layer (`stages.py` and `engine_io.py`).

The E2E test (`tests/test_thinker_talker_e2e.py`) must continue to pass with identical behavior:
1. Thinker produces text tokens + captures hidden states
2. Talker produces `codec_codes` (shape `[16, N]`) and `summed_codec_embeds`
3. Decode stage returns both text + codec outputs

## Design: How Talker Reuses the AR Engine

### Key Insight

The talker's codec generation is a **two-phase** process:
1. **Backbone forward**: Standard transformer forward pass (same as thinker) → produces `hidden_states`
2. **Post-processing**: `compute_logits()` → `code_predictor_forward()` → codec codes

Phase 1 is exactly what the AR engine's `SGLangModelRunner` already does. Phase 2 is talker-specific post-processing that belongs in the executor's `result_builder`.

### Execution Strategy

The talker does NOT need iterative token-by-token decoding. It needs:
- **One prefill pass** (the full input sequence) to get backbone hidden states
- **Immediate completion** after that single pass (no decode loop)

This maps to: use the AR engine with `max_new_tokens=1` (or a custom finish condition), run one forward step, then extract the hidden states and run codec generation in the `result_builder`.

### Concrete Approach

**Option A: Single-step AR with hidden state capture (Recommended)**

1. Load `Qwen3OmniTalker` as the model in a new `ModelWorker`
2. Use `create_sglang_ar_engine()` with `capture_hidden_layers` to capture the backbone's final hidden states
3. In `request_builder`: build a `SGLangARRequestData` from talker inputs (thinker_embed → inputs_embeds)
4. The AR engine runs one prefill step, capturing the backbone output
5. In `result_builder`: extract captured hidden states, run `talker_model.compute_logits()` + `talker_model.code_predictor_forward()` to produce codec codes

**Challenge**: The AR engine currently captures hidden states via forward hooks on specific layers. For talker, we need the **final output** of the backbone (the last hidden state), not an intermediate layer. This is already available as the model's forward output.

**Option B: Custom ModelRunner wrapping (Alternative)**

Instead of modifying the AR engine, create a thin `TalkerModelRunner` that wraps `SGLangModelRunner` and adds codec post-processing. This keeps the AR engine untouched but still reuses its scheduling/batching/KV-cache infrastructure.

### Recommended: Option A with Modifications

The cleanest path:

1. **Model loading**: Load `Qwen3OmniTalker` into `ModelWorker` (it has `forward(input_ids, positions, forward_batch, inputs_embeds)` — compatible with SGLang's model runner interface)
2. **Request building**: Convert thinker hidden states to `inputs_embeds` via `talker_model.prepare_input_embeds()` BEFORE submitting to the AR engine. Store as `SGLangARRequestData` with `max_new_tokens=1`.
3. **Hidden state capture**: Capture the final layer's output (or the model's forward output) via existing `HiddenStateCaptureHook` infrastructure, targeting the last decoder layer.
4. **Finish after one step**: The AR engine will generate 1 token and finish (due to `max_new_tokens=1`). The generated token is irrelevant — we only care about the captured hidden states.
5. **Result building**: In `result_builder`, take the captured hidden states, run `compute_logits()` and `code_predictor_forward()` on the talker model, produce codec codes.

## Detailed Task Breakdown

### Task 1: Adapt `create_sglang_talker_executor()` in `stages.py`

**Current** (lines 351-406): Creates a separate talker engine via `create_sglang_talker_engine()`.

**New**:
- Use `create_sglang_ar_engine()` with the talker model loaded in ModelWorker
- Configure `capture_hidden_layers` to capture the talker backbone's last layer output
- Set `max_new_tokens=1` in request params so the AR engine finishes after one step

**Changes to `_request_builder`**:
- Call `talker_model.prepare_input_embeds(thinker_embed, thinker_hidden, is_multimodal_mask)` to get `inputs_embeds`
- Build `SGLangARRequestData` with the input_ids derived from the sequence length, and pass `inputs_embeds` via model_inputs or a dedicated field
- The SGLang Req needs `origin_input_ids` matching the sequence length

**Changes to `_result_builder`**:
- Extract captured hidden states from `result.extra_model_outputs`
- Call `talker_model.compute_logits(hidden_states)` → layer-0 codes
- Call `talker_model.code_predictor_forward(layer0_codes, hidden_states)` → full codec codes
- Store codec_codes and summed_embeds on `state.talker_out`

### Task 2: Handle `inputs_embeds` passthrough in SGLang AR engine

**Problem**: The current `SGLangModelRunner.execute()` does not support passing `inputs_embeds` to the model. The thinker passes raw token IDs; the talker needs to pass pre-computed embeddings.

**Solution options**:
a. Store `inputs_embeds` on the SGLang `Req` object (custom field) and inject it during `ForwardBatch` construction
b. Use the existing `model_inputs` dict on `SGLangARRequestData` to carry `inputs_embeds`, and modify the model runner to check for it
c. Use the existing SGLang `Req.image_inputs` or similar multimodal input mechanism to carry embeddings

**Recommended**: Option (a) — add an `inputs_embeds` field to the request data, and modify `SGLangModelRunner.execute()` to inject it into the `ForwardBatch` before calling forward. This is a small, contained change.

### Task 3: Ensure talker model is loadable by `ModelWorker`

**Check**: `Qwen3OmniTalker` must be compatible with SGLang's `ModelWorker` initialization:
- Has `forward(input_ids, positions, forward_batch, inputs_embeds=None)`  ✓
- Has `load_weights(weights)` ✓
- Model config must be compatible with `ModelWorkerConfig`

**If incompatible**: May need a thin adapter or custom `ModelWorkerConfig` for the talker. Investigate `ModelWorker.__init__` to see what it expects.

### Task 4: Handle codec post-processing outside the engine

**In `result_builder`**:
```python
def _result_builder(payload, result):
    state = load_state(payload)
    # Extract hidden states captured by AR engine
    extra = result.extra_model_outputs
    hidden_states = extra.get("talker_backbone_output")  # from last layer capture

    # Run codec generation (these are model methods, not engine logic)
    hidden_states = hidden_states.unsqueeze(0)  # add batch dim
    logits = talker_model.compute_logits(hidden_states.squeeze(0))
    layer0_codes = logits.argmax(dim=-1).unsqueeze(0)
    codec_codes, summed_embeds = talker_model.code_predictor_forward(
        layer0_codes, hidden_states
    )

    state.talker_out = {
        "codec_codes": codec_codes.squeeze(0).tolist(),
        "step": 1,
        "is_final": True,
    }
    return store_state(payload, state)
```

### Task 5: Update `create_sglang_talker_executor_from_config()`

**Current** (lines 409-438): Loads talker model, patches attention with SDPA, creates talker engine.

**New**:
- Still loads talker model and config
- Instead of patching attention with SDPA, load model into `ModelWorker` (which provides proper KV cache infrastructure)
- Call `create_sglang_ar_engine()` with talker's `ServerArgs`
- Keep a reference to `talker_model` for codec post-processing in `result_builder`

### Task 6: Delete `sglang_talker.py` and clean up

**Delete**:
- `sglang_omni/engines/omni/runtime/sglang_talker.py` (662 lines)

**Modify**:
- `sglang_omni/engines/omni/runtime/__init__.py`: Remove all `_TALKER_EXPORTS`, `_TalkerAttr` class, and talker lazy proxies
- `sglang_omni/engines/omni/__init__.py`: Remove `create_sglang_talker_engine` export
- `sglang_omni/engines/omni/factory.py`: Remove `create_sglang_talker_engine()` function
- `sglang_omni/models/qwen3_omni/pipeline/engine_io.py`: Remove `build_sglang_talker_request()` and `TalkerARRequestData` imports; update `apply_talker_result()` to work with `SGLangARRequestData` instead
- `sglang_omni/models/qwen3_omni/pipeline/stages.py`: Rewrite `create_sglang_talker_executor()` and `create_sglang_talker_executor_from_config()` to use AR engine

### Task 7: Verify E2E test passes

Run `tests/test_thinker_talker_e2e.py` and verify:
1. Thinker produces text tokens + hidden states ✓ (unchanged)
2. Talker produces `codec_codes` with shape `[16, N]` ✓
3. Decode stage returns both text + codec outputs ✓
4. No regressions in streaming behavior

## Risk Assessment

### High Risk: `inputs_embeds` passthrough
The SGLang model runner pipeline (`ScheduleBatch → ModelWorkerBatch → ForwardBatch`) does not natively support `inputs_embeds`. This is the most complex part — need to ensure the embedding injection happens at the right point without breaking the KV cache / attention backend.

**Mitigation**: If injection into the standard pipeline is too invasive, fall back to a thin wrapper `TalkerSGLangModelRunner` that constructs its own `ForwardBatch` (similar to what the current `TalkerModelRunner._backbone_via_model_worker()` does), but reuses the AR engine's scheduler/resource/iteration infrastructure.

### Medium Risk: ModelWorker compatibility
The `ModelWorker` may have assumptions about model structure (e.g., `lm_head`, embedding layer) that `Qwen3OmniTalker` doesn't satisfy.

**Mitigation**: Check `ModelWorker.__init__` and `ModelWorkerConfig` requirements. May need to skip weight loading through ModelWorker and load separately.

### Low Risk: Finish condition
Using `max_new_tokens=1` to finish after one step is a clean workaround. The generated token is discarded; only captured hidden states matter.

## Files Changed Summary

| File | Action | Lines |
|------|--------|-------|
| `sglang_omni/engines/omni/runtime/sglang_talker.py` | **DELETE** | -662 |
| `sglang_omni/engines/omni/runtime/__init__.py` | Modify: remove talker exports | ~-40 |
| `sglang_omni/engines/omni/factory.py` | Modify: remove `create_sglang_talker_engine` | ~-40 |
| `sglang_omni/engines/omni/__init__.py` | Modify: remove talker exports | ~-5 |
| `sglang_omni/models/qwen3_omni/pipeline/stages.py` | Modify: rewrite talker executor creation | ~±80 |
| `sglang_omni/models/qwen3_omni/pipeline/engine_io.py` | Modify: update talker request/result helpers | ~±40 |
| `sglang_omni/engines/omni/runtime/sglang_ar.py` | Modify: support `inputs_embeds` passthrough (if needed) | ~+20 |

**Net change**: ~-600 lines (delete 662, add ~60)

## Execution Order

1. **Read & understand** `ModelWorker` initialization to assess talker model compatibility
2. **Prototype** `inputs_embeds` injection in `SGLangModelRunner` (Task 2)
3. **Rewrite** `create_sglang_talker_executor_from_config` to use AR engine (Tasks 1, 5)
4. **Update** `engine_io.py` result/request builders (Task 4)
5. **Run E2E test** to verify (Task 7)
6. **Delete** `sglang_talker.py` and clean up exports (Task 6)
7. **Final E2E test** to confirm clean state
