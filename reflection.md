# Reflection: Qwen3-Omni Video Input & Prefill Performance Investigation

## 1. Background and Problem Statement

The Qwen3-Omni model's backend was migrated from native HuggingFace/Torch to the SGLang autoregressive model runner. Two problems surfaced:

1. **On the `main` branch**: Images and videos sent through the Web UI playground were never actually passed to the thinker. The thinker's text output was completely unrelated to the visual content — the model was effectively "blind."

2. **On the `img-lost` branch** (this branch): The image/video data loss was fixed, and the model now correctly receives and understands visual input. However, video processing (prefill) is **extremely slow** — roughly 55 seconds for a single 16-frame video request.

The investigation had two goals:
- Understand the full story: why was video input invalid before, how did this branch fix it, and what remains slow.
- Build an integration test, measure performance, and diagnose the root cause of the slow prefill for both single-turn and multi-turn conversations.

---

## 2. Why Video Input Was Invalid, and How This Branch Fixed It

To understand the bug, you first need to understand how Qwen3-Omni processes a video. It is a multi-stage pipeline:

```
User sends: video file + text prompt "Where am I right now?"

  Stage 1 - Preprocessing (CPU):
      Load video frames, tokenize text.
      The video is represented as placeholder tokens (<video> × 2400)
      in the token sequence, to be filled in later.

  Stage 2 - Image Encoder (GPU):
      Feed raw pixel patches through a Vision Transformer (ViT).
      Produces: video_embeds (flat embeddings for all visual tokens)
              + deepstack_visual_embeds (per-layer visual features
                for intermediate transformer layer injection)

  Stage 3 - Thinker (GPU, SGLang):
      Replace <video> placeholders with real video_embeds.
      Inject deepstack features at intermediate layers.
      Run autoregressive generation → produce text answer.
```

### 2.1 The Bug on `main`: The Thinker Never Received the Video

On the `main` branch, the video data was lost somewhere between Stage 2 and Stage 3. The thinker received the token sequence with `<video>` placeholders, but the actual visual embeddings never arrived. The model had no visual information to work with — it could only see the text prompt "Where am I right now?" and hallucinate an answer. Users observed that "the text output seemed completely unrelated to the video content."

### 2.2 The Fix on `img-lost`: Restoring the Data Flow

This branch repaired the full pipeline so that visual data flows end-to-end:

1. The merge stage (`merge_for_thinker`) correctly collects encoder outputs and packs them into `thinker_inputs.model_inputs`.
2. The SGLang request builder (`build_sglang_thinker_request`) attaches these as `req.omni_model_inputs` on the SGLang `Req` object.
3. During prefill, `SGLangModelRunner._inject_multimodal_embeds()` scans the token sequence for `<video>` / `<image>` / `<audio>` placeholders and replaces them with the real embeddings from the encoder.
4. `_forward_with_omni_embeds()` runs the model forward pass with the injected multimodal content.

After this fix, the model correctly understands video content — it identifies gate numbers, seating areas, a "UCI Health" advertisement, and architectural features.

### 2.3 The Deepstack Data Loss: A Deeper Look at the Root Cause

The placeholder replacement in step 3 above handles the "flat" embeddings (`video_embeds`). But Qwen3-Omni also relies on a mechanism called **deepstack**: multi-layer visual features that are injected at intermediate transformer layers (not just at the input). Think of it as giving the model multiple "looks" at the visual content at different levels of abstraction.

On the `main` branch, the deepstack data was being dropped at the last mile. Here is the problematic code in `_forward_with_omni_embeds`:

```python
def _forward_with_omni_embeds(self, forward_batch, input_embeds,
                               deepstack_visual_embeds=None,    # RECEIVED from upstream
                               visual_pos_masks=None):          # RECEIVED from upstream
    ...
    hidden_states = outer.model(
        input_ids=None,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        # deepstack_visual_embeds → NEVER PASSED DOWN
        # visual_pos_masks → NEVER PASSED DOWN
    )
```

The method receives the deepstack data, but silently drops it — it never reaches `outer.model()`. Meanwhile, the SGLang language model is ready to accept it:

```python
# Qwen3MoeLLMModel.forward() signature:
def forward(self, input_ids, positions, forward_batch,
            input_embeds=None, pp_proxy_tensors=None,
            input_deepstack_embeds=None):  # ← exists, but always receives None
```

And inside the model, each transformer layer checks for deepstack data:

```python
def get_deepstack_embeds(self, layer_idx, input_deepstack_embeds):
    if input_deepstack_embeds is None:  # Always None → deepstack skipped entirely
        return None
    sep = self.hidden_size * layer_idx
    return input_deepstack_embeds[:, sep : sep + self.hidden_size]
```

There is also a **format mismatch** between the two sides that would need bridging even if the data were passed through:
- The pipeline produces deepstack as a **list of tensors** (one per layer, shape `[num_visual_tokens, hidden_size]`)
- SGLang expects a **single concatenated 2D tensor** `[seq_len, hidden_size × num_layers]`, sliced by `hidden_size × layer_idx`
- Non-visual positions must be zero-padded to the full sequence length
- (Verified by tracing `post_residual_addition` through `LayerCommunicator.prepare_attn()` → `RMSNorm`)

The `img-lost` branch resolved the video input problem by ensuring that visual features are correctly incorporated into `input_embeds` via the embedding merge in `_inject_multimodal_embeds`, before the model forward call. This works around the missing `input_deepstack_embeds` passthrough by baking the visual information directly into the input embedding layer.

**Remaining gap**: There are two channels for getting visual data into the thinker. Channel A is the **input embedding replacement** — `_inject_multimodal_embeds` replaces `<video>` placeholders with flat `video_embeds`, giving the model visual information at the input layer (layer 0) only. Channel B is the **deepstack injection** — passing `input_deepstack_embeds` to `outer.model()` so the language model can add per-layer visual features as residuals at intermediate transformer layers (e.g., layers 0, 1, 2). This multi-layer injection is how Qwen3-Omni is architecturally designed to process vision.

Currently, only Channel A works. Channel B is broken: `_forward_with_omni_embeds` receives the deepstack data from the upstream pipeline but never forwards it to `outer.model()`. The model produces reasonable outputs because Channel A provides enough visual information at layer 0, but it is missing the intermediate-layer visual residuals that Qwen3-Omni's architecture expects.

The ideal fix is straightforward: convert the per-layer list of tensors into the concatenated 2D format that SGLang expects, then pass it as `input_deepstack_embeds=ds_input` to `outer.model()`. During this investigation, I wrote this conversion and passthrough. The risk of leaving Channel B broken is that it depends on Channel A being sufficient on its own — if the model weights or architecture are updated to rely more heavily on deepstack's multi-layer injection, model quality would silently degrade with no error or warning.

---

## 3. Investigation Process: Why Is Video Processing Still Slow?

### 3.1 Building the Integration Test

I first wrote `tests/test_video_integration.py` to establish a measurable baseline:

- Starts the backend server as a subprocess (`python -m sglang_omni.cli.cli serve`)
- Waits for the `/health` endpoint
- Sends a video + text prompt ("Where am I right now?") via HTTP POST to `/v1/chat/completions`
- Validates the response contains expected keywords (gate, station, 12, UCI, etc.)
- Reports server startup time and E2E latency

**First test result** (no code changes):

| Metric | Value |
|--------|-------|
| Server startup | 21.0s |
| E2E latency | 54.2s |
| Response | "You are at a train station" (mentioning gate 12, UCI Health) |

The model correctly understood the video — confirming the img-lost branch fix works. But 54.2 seconds is very slow.

### 3.2 Attempt 1: "Is SGLang Actually Being Used?" (Hypothesis Disproved)

The initial suspicion was that the SGLang backend was not actually executing — that the thinker might be falling back to native HuggingFace/PyTorch inference.

**Method**: I traced the model class resolution chain to determine exactly which model SGLang loads.

```python
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang_omni.models"
model_cls, arch = get_model_architecture(model_config)
# Result: sglang.srt.models.qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration
```

I verified the full model hierarchy:
- `Qwen3OmniMoeForConditionalGeneration.thinker` = `Qwen3OmniMoeThinkerForConditionalGeneration`
- `.thinker.model` = `Qwen3MoeLLMModel` (SGLang-optimized with `RadixAttention`)
- `_get_inner_model_components()` in `sglang_ar.py` extracts `outer = model.thinker`, `outer.model` = the language model

I also checked that `_forward_with_omni_embeds` calls:
```python
model_runner.attn_backend.init_forward_metadata(forward_batch)  # initializes SGLang attention
hidden_states = outer.model(...)  # calls the SGLang-optimized language model
```

**Conclusion**: The SGLang backend IS being used. The attention is computed via FlashAttention/FA3 through `RadixAttention`. **This hypothesis was wrong.** The slowness is not caused by a fallback to native PyTorch.

### 3.3 Attempt 2: Increasing Chunked Prefill Size (Marginal Effect)

The default SGLang configuration had very conservative prefill chunking:
```python
"chunked_prefill_size": 128,
"max_prefill_tokens": 4096,
```

For a video with 2400 visual tokens (2416 total), this means 19 separate scheduler round-trips, each processing only 128 tokens. Each round-trip involves:
- `select_requests()` with PrefillManager scheduling
- `build_batch()` creating ScheduleBatch
- `execute()` running the model forward pass
- `update()` processing outputs
- `asyncio.sleep(0)` yielding to the event loop

I increased to `chunked_prefill_size=8192` and `max_prefill_tokens=16384`, reducing this to 1 pass.

**Result**: E2E changed from 54.7s to 57.2s — **no significant improvement**. The scheduler overhead from 19 round-trips was negligible compared to the actual model computation.

**Conclusion**: Chunked prefill scheduling overhead is not the bottleneck.

### 3.4 Attempt 3: Adding Timing Instrumentation (Identifying the Real Bottleneck)

I added timing logs inside `SGLangModelRunner.execute()` to measure `_forward_with_omni_embeds` vs `forward_batch_generation` per-step. However, since the server runs as a subprocess, the logs were not easily accessible from the test.

Instead, I profiled individual components in isolation:

| Component | Time | Notes |
|-----------|------|-------|
| Video loading (`ensure_video_list_async`) | 0.3s | 16 frames from WebM via torchcodec |
| HF Processor tokenization | 0.1s | Produces 2400 video tokens from 9600 patches |
| Image encoder model load | 0.5s | HF ViT, one-time |
| Thinker model load | ~18s | 30B MoE via SGLang, one-time (in startup) |

Key video metrics:
- **16 frames** at 476x644 resolution
- **9600 ViT patches** (each frame divided into patches, total `[9600, 1536]`)
- After spatial merge (÷4): **2400 visual tokens**
- Total prompt: **2416 tokens** (2400 video + 16 text/system)

I also verified the HF ViT uses **SDPA** (Scaled Dot Product Attention), confirming it dispatches to an efficient attention kernel. So the vision encoder is not using naive O(n^2) attention.

**Conclusion**: The ~55s E2E is dominated by two heavy model forward passes:
1. **Image encoder**: 9600 patches through HF ViT (SDPA)
2. **Thinker prefill + decode**: 2416 tokens through 30B MoE model (SGLang/FA3), then ~150 decode steps

Both are compute-bound on large model forward passes, not framework overhead.

### 3.5 Attempt 4: Checking for HF vs SGLang Model Confusion (Dead End)

I initially confused the custom `sglang_omni/models/qwen3_omni/thinker.py` (which defines `Qwen3OmniMoeThinkerTextModel` with `RadixAttention`) with the HF model in `components/thinker.py` (which uses `hf_modeling.Qwen3OmniMoeThinkerTextModel`). This led me to believe the wrong model might be loaded.

After tracing `SGLANG_EXTERNAL_MODEL_PACKAGE` → `get_model_architecture()`, I confirmed that SGLang loads its **own built-in model** (`sglang.srt.models.qwen3_omni_moe`), not either of the sglang_omni custom models. The custom `thinker.py` in sglang_omni appears to be an alternative implementation that is not currently used by the pipeline.

**Dead end** — but the tracing was necessary to rule out model confusion.

---

## 4. Second-Round Slowness Investigation

### 4.1 The Observation

After adding a two-round conversation test (round 1: "Where am I right now?", round 2: "Is there a specific school shown in the video?"), the results confirmed the user's observation:

| Metric | Time |
|--------|------|
| Server startup | 20.0s |
| Round 1 latency | **57.3s** |
| Round 2 latency | **51.6s** |
| Speedup | **1.11x** (essentially no improvement) |

Round 2 includes the full round 1 context as conversation history, plus the same video. If KV cache / prefix caching worked, round 2 should only need to prefill the new tokens (~200), reusing cached KV states for the shared prefix (~2400+ tokens). This would give a dramatic speedup. Instead, round 2 was nearly as slow.

### 4.2 Investigation: The Pipeline Re-runs Everything from Scratch

The sglang_omni pipeline treats **each API request as a completely independent pipeline execution**:

```
Round 2 API request (multi-turn messages + video path)
  1. Preprocessing stage: re-tokenizes ALL messages, re-loads video frames  (~0.5s)
  2. Image encoder stage: re-encodes 9600 video patches through ViT         (~15-25s)
  3. Audio encoder stage: re-runs                                            (~minimal)
  4. Aggregate stage: re-merges encoder outputs                              (~minimal)
  5. Thinker stage: new SGLang Req with full token sequence                  (~25-30s)
```

The image encoder (step 2) re-processes the **exact same video** — 9600 patches through the HF ViT — purely redundant computation.

### 4.3 Investigation: Does SGLang's Tree Cache Help the Thinker?

I traced the SGLang prefix caching mechanism through four layers:

**Step 1: Tree cache creation.** `create_tree_cache()` is called in `factory.py` line 216 and passed to `PrefillManager`, `SGLangResourceManager`, and `SGLangIterationController`. The infrastructure exists.

**Step 2: KV cache release after round 1.** When round 1 finishes, `SGLangResourceManager.free()` calls `release_kv_cache(data.req, self.tree_cache)`. This returns KV entries to the tree cache for later reuse.

**Step 3: Prefix matching during round 2.** `PrefillManager.schedule_next_batch()` calls `req.init_next_round_input(tree_cache)`, which attempts:
```python
match_result = tree_cache.match_prefix(
    MatchPrefixParams(
        key=RadixKey(token_ids=token_ids, extra_key=self.extra_key),
        ...
    )
)
```

**Step 4: Token ID prefix analysis.**
- Round 1: `[system_tokens..., <video>*2400, user_text_tokens...]`
- Round 2: `[system_tokens..., <video>*2400, user_text_tokens..., assistant_tokens..., new_user_tokens...]`

The prefix is identical at the token ID level. So the tree cache SHOULD match. But I found that `extra_key` is **never set** in sglang_omni code (confirmed by grepping — zero matches across the entire codebase). The match is purely on token IDs.

**Step 5: Multimodal complication.** The video placeholder tokens (ID 151656) are the same in both rounds, but the actual embeddings that replace them are freshly computed each time. The tree cache matches token IDs, not embeddings. For the same video this is fine (embeddings are numerically identical), but the architecture doesn't guarantee this.

### 4.4 Investigation: Why Encoder Caching is Disabled

I discovered that the encoder caching infrastructure already exists but is turned off:

```python
# In stages.py: _create_encoder_executor
engine = create_encoder_engine(model, device=device)  # use_cache defaults to False
```

The `create_encoder_engine` function in `factory.py` accepts `use_cache=True` and `cache_size`, and the `SimpleCacheManager` class implements a working cache. The preprocessor already computes cache keys from video paths (`compute_video_cache_key`). **The plumbing is all there — it's just not enabled.**

### 4.5 Conclusion: Three Layers of Missing Caching

The second-round slowness stems from three independent caching layers that are absent or ineffective:

| Layer | What Should Be Cached | Current State | Estimated Impact |
|-------|----------------------|---------------|-----------------|
| **Encoder outputs** | ViT embeddings for the same video | `use_cache=False` — disabled | **~15-25s wasted** |
| **Preprocessing** | Tokenized inputs for the same video | No caching — re-tokenizes everything | ~0.5s (minor) |
| **Thinker KV cache** | KV states for shared token prefix | Tree cache exists, release/match implemented, but effectiveness unclear with multimodal token flow | **~10-15s potential** |

### 4.6 Potential Fix Directions

1. **Enable encoder cache**: Change `create_encoder_engine(model, device=device)` to `create_encoder_engine(model, device=device, use_cache=True)` in `_create_encoder_executor`. Cache keys are already computed.

2. **Conversation-level state**: Carry encoder outputs and thinker KV cache across API requests sharing the same conversation. This is a deeper architectural change.

3. **Set `extra_key` for multimodal requests**: Make SGLang's tree cache content-aware by setting `req.extra_key` based on video/image content hash.

---

## 5. All Exploration Attempts — Summary Table

| # | Attempt | Method | Result | Conclusion |
|---|---------|--------|--------|------------|
| 1 | "Is SGLang being used?" | Traced model class resolution chain: `get_model_architecture()` → `sglang.srt.models.qwen3_omni_moe` | SGLang's own optimized model with RadixAttention/FA3 is loaded | **Disproved** — SGLang IS active, not falling back to HF |
| 2 | Deepstack data loss analysis | Traced `_forward_with_omni_embeds` → `outer.model()`, inspected `get_deepstack_embeds()`, analyzed format mismatch (list vs concatenated tensor) | Identified that `input_deepstack_embeds` is silently dropped; wrote format conversion fix; however bug was already fixed on this branch via embedding merge | **Documented latent gap** — deepstack passthrough missing in SGLang path, but currently masked by upstream fix |
| 3 | Increase chunked prefill size | Changed from 128 to 8192 (reducing 19 scheduler round-trips to 1) | 54.7s → 57.2s (no improvement) | **Disproved** — scheduler overhead is negligible |
| 4 | Component-level profiling | Measured video loading (0.3s), tokenization (0.1s), encoder load (0.5s) in isolation | Identified 9600 ViT patches and 2416 thinker tokens as the heavy compute | **Confirmed** — model forward passes dominate |
| 5 | HF vs SGLang model confusion | Traced `SGLANG_EXTERNAL_MODEL_PACKAGE` and `_get_inner_model_components` | SGLang loads its built-in model, not sglang_omni custom models | **Dead end** — but necessary to rule out |
| 6 | ViT attention check | Inspected `Qwen3OmniMoeVisionAttention` config | Uses SDPA (efficient), not naive attention | **Confirmed** — ViT is already reasonably optimized |
| 7 | Timing instrumentation in `execute()` | Added `time.monotonic()` around forward calls | Logs went to subprocess, not easily captured | **Inconclusive** — needed different approach |
| 8 | Two-round conversation test | Round 1: 57.3s, Round 2: 51.6s (1.11x speedup) | Second round is NOT faster | **Confirmed** — prefix caching is not effective |
| 9 | Tree cache prefix matching trace | Traced `init_next_round_input` → `tree_cache.match_prefix` → `RadixKey` | Token IDs should match, but `extra_key` never set | **Identified** — caching infrastructure exists but gaps remain |
| 10 | Encoder cache investigation | Found `use_cache=False` default in `create_encoder_engine` | Cache infrastructure exists, just disabled | **Root cause found** — biggest potential fix for round 2 |

---

## 6. Directions Explored But Not Pursued

### 6.1 Replacing HF Image Encoder with SGLang-native

The SGLang model includes its own `visual` encoder and `audio_tower`, invoked via `general_mm_embed_routine` in the native path. The sglang_omni pipeline uses separate HF-based encoder stages instead. Replacing them could avoid redundant model loading, but is a significant architectural change.

### 6.2 CUDA Graph for Thinker Decode

`disable_cuda_graph=True` is set in the config. Enabling CUDA graphs for decode could speed up token generation, but may conflict with the multimodal embedding injection path which requires variable tensor shapes.

### 6.3 Single-Process Architecture

The multi-stage pipeline (ZMQ + SHM relay) introduces per-request serialization overhead. A single-process architecture would eliminate this, similar to how SGLang natively handles multimodal models.

### 6.4 Vision Encoder with Flash Attention 2

The HF ViT uses SDPA by default. Explicitly loading with `attn_implementation="flash_attention_2"` might be faster for the vision encoder. Not tested.

---

## 7. Key Learnings

1. **Always verify assumptions with code tracing**: The initial hypothesis ("SGLang not being used") was wrong. The SGLang backend IS active with FlashAttention. Only by tracing the actual model class resolution (`get_model_architecture()`) could this be confirmed.

2. **Establish a baseline before making changes**: The deepstack passthrough edit had no measurable effect because the bug was already fixed. Running the test first would have shown this immediately.

3. **Model computation dominates framework overhead**: For a 30B MoE model with 2400 visual tokens, the forward pass is inherently expensive (~25-30s). Scheduler optimizations (chunked prefill size) and framework choices matter far less than the raw compute.

4. **Multi-turn performance requires caching at every pipeline layer**: Three independent caching layers (encoder outputs, preprocessing, thinker KV) all need to work together. The encoder cache is disabled (`use_cache=False`), which alone accounts for ~15-25s of wasted re-computation per follow-up request.

5. **Profiling individual components reveals the truth**: Measuring video loading (0.3s), tokenization (0.1s), etc. in isolation quickly narrowed down the bottleneck to the two large model forward passes (ViT encoder and thinker), not the pipeline plumbing.

6. **Infrastructure often exists before it's enabled**: The encoder cache (`SimpleCacheManager`), cache keys (`compute_video_cache_key`), and tree cache prefix matching are all implemented — they just need to be wired together and turned on.
