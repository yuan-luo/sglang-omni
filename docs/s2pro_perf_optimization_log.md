# S2-Pro SGLang Performance Optimization Log

> Date: 2026-03-08
> Branch: `feat/fishaudio_s1_s2pro`
> Goal: Fix `torch.compile` WER bug, optimize sampling throughput, enable CUDA graph for codebook loop

---

## Baseline (before changes)

| Config | tok/s (BS=1) | WER | Notes |
|--------|-------------|-----|-------|
| `compile=True` (old `max-autotune`) | ~60 | **99%** | Fast but completely broken output |
| `compile=False` (eager) | ~26 | ~1% | Correct but slow |
| Batched BS=8 (eager) | ~19.7 per-req | ~1% | Per-req throughput degrades 69% vs BS=1 |

---

## P0: Fix torch.compile + Optimize Sampling

### Problem 1: torch.compile produces garbage audio (WER 99%)

**Root cause**: `torch.compile(mode="max-autotune")` internally captures CUDA graphs via Inductor. The sampling function used `torch.multinomial` which requires CPU-GPU synchronization — incompatible with CUDA graph replay. The random state gets corrupted, producing nonsense tokens.

**Fix**: Changed compile mode to `"max-autotune-no-cudagraphs"` in `s2pro_sglang_ar.py`:

```python
# Before
torch.compile(..., mode="max-autotune")

# After
torch.compile(..., mode="max-autotune-no-cudagraphs")
```

### Problem 2: Sampling function too slow (sort-based, O(V log V))

**Root cause**: The initial fix replaced `topk+scatter` with `sort+argsort+gather` (the `_logits_to_probs` approach from the reference `flash-fish-inference-bak`). This was CUDA-graph-safe but operated on the **full vocabulary** (4096 elements for codebooks):

```python
# Old slow path (2x sort on 4096 elements)
sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # O(V log V)
cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)   # O(V)
# ... mask, temperature, softmax on all V elements ...
inverse_indices = torch.argsort(sorted_indices, dim=-1)              # O(V log V)
return torch.gather(probs_sort, dim=-1, index=inverse_indices)       # O(V)
```

**Fix**: Rewrote `_sample_with_topk` to use `torch.topk` (O(V + k log k)) and operate on only k=30 elements:

```python
def _sample_with_topk(logits, temperature, top_p, top_k=30, ...):
    # 1. topk: O(V) scan + O(k log k) sort, returns [bs, k] sorted descending
    top_vals, top_indices = torch.topk(logits, k=top_k, dim=-1)

    # 2. Top-p filtering on k=30 elements (not 4096)
    probs_raw = F.softmax(top_vals, dim=-1)
    cum_probs = torch.cumsum(probs_raw, dim=-1)
    mask = cum_probs > top_p
    mask[..., 0] = False

    # 3. Temperature scaling + re-softmax on k elements
    top_vals = top_vals / torch.clip(temperature, min=1e-5)
    top_vals = top_vals.masked_fill(mask, -float("inf"))
    probs = F.softmax(top_vals, dim=-1)

    # 4. Gumbel-max sampling (no CUDA sync, no torch.multinomial)
    q = torch.empty_like(probs).exponential_(1)
    sampled_idx = torch.argmax(probs / q, dim=-1, keepdim=True)

    # 5. Map back to original vocab index via gather (not scatter)
    return torch.gather(top_indices, dim=-1, index=sampled_idx).to(dtype=torch.int)
```

**Why this is safe for CUDA graphs and torch.compile**:
- `torch.topk` with int `k`: deterministic, no CPU-GPU sync
- `gather` instead of `scatter`: deterministic with duplicate indices
- `exponential_()` + `argmax` (Gumbel-max): replaces `torch.multinomial` which requires sync
- All ops on fixed-shape tensors `[bs, 30]`

### Files changed

| File | Change |
|------|--------|
| `runtime/s2pro_ar.py` | Rewrote `_sample_with_topk`: removed `_logits_to_probs` (sort-based) and `_multinomial_no_sync`, replaced with topk+gather+Gumbel-max |
| `runtime/s2pro_sglang_ar.py` | Changed compile mode to `max-autotune-no-cudagraphs` |

### P0 Results

| Iteration | tok/s | WER | Method |
|-----------|-------|-----|--------|
| v1 (sort-based) | ~50 | 1.10% | argsort+gather on full vocab |
| **v2 (topk-based)** | **52.9** | **1.10%** | torch.topk on k=30 elements |

v2 recovered ~3 tok/s by replacing O(V log V) sort with O(V + k log k) topk. TTFB improved from 217ms to 207ms. RTF from 0.43 to 0.41.

---

## P1: CUDA Graph for Codebook Loop

### Problem 1: CUDA graph capture crashes with torch.compile

**Error** (first attempt):
```
TorchRuntimeError: Dynamo failed to run FX node with fake tensors:
  rms_norm ... normalized_shape=[2560], but got input of size[1, 1, 1024]
```

**Root cause**: `S2ProFastGraphRunner` used the **compiled** function (`self._codebook_fn`) for CUDA graph capture. Dynamo's fake tensor tracing failed on the shape transformation through `project_in`.

**Fix**: Use **eager** function for CUDA graph capture. CUDA graphs and torch.compile serve overlapping purposes — using both is redundant and problematic.

```python
# Before (crashes)
self._fast_graph_runner = S2ProFastGraphRunner(
    codebook_fn=self._codebook_fn,  # compiled
    ...
)

# After
self._fast_graph_runner = S2ProFastGraphRunner(
    codebook_fn=self._codebook_fn_eager,  # eager, no Dynamo tracing
    ...
)
```

### Problem 2: Static buffer has wrong hidden dimension

**Error** (second attempt, after fixing Problem 1):
```
RuntimeError: Given normalized_shape=[2560], expected input with shape [*, 2560],
  but got input of size[1, 1, 1024]
```

**Root cause**: The hidden_dim probing code inspected `audio_decoder.project_in.parameters()` to get the input dimension. But for S2-Pro, **`config.text_dim == config.dim` (both 2560)**, so `project_in = nn.Identity()` which has **zero parameters**. The probing loop iterated zero times and fell back to `hidden_dim = 1024` (wrong).

The text model (Qwen3-4B) outputs hidden states with dim=2560. The static buffer must match.

```python
# Before (wrong: Identity has no params, fallback to 1024)
hidden_dim = None
for p in audio_decoder.project_in.parameters():
    hidden_dim = p.shape[1]
    break
if hidden_dim is None:
    hidden_dim = 1024  # <-- wrong for Qwen3-4B

# After (correct: read from audio decoder config)
hidden_dim = getattr(
    getattr(audio_decoder, "config", None), "text_dim", None
)
if hidden_dim is None:
    for p in audio_decoder.project_in.parameters():
        hidden_dim = p.shape[1]
        break
if hidden_dim is None:
    hidden_dim = 2560
```

### Fix 3: `top_k` passed as tensor but `torch.topk` needs Python int

```python
# Before: top_k was a tensor buffer in the graph runner
self._top_k_buf = torch.full((max_bs, 1), 30, dtype=torch.int64)
# ... passed as tensor to codebook_fn

# After: top_k is a Python int, baked into the captured graph
self._top_k = top_k  # int, e.g. 30
# ... passed as int constant during capture
def run():
    return self._codebook_fn(h, s, t, p, self._top_k)
```

The `top_k` value (30) is constant across all requests and all replay calls, so baking it as a compile-time constant is correct and simplifies the interface.

### Audio Decoder CUDA Graph Compatibility

Verified that the audio decoder (`fish_speech/models/text2semantic/modeling.py`) is CUDA-graph-safe:

- `reset_caches()`: uses `.zero_()` (in-place tensor op, captured by graph)
- `forward_kvcached()`: uses `self.input_pos.fill_(codebook_idx)` (in-place tensor op)
- KV cache updates: all in-place tensor operations
- No Python-level state changes that would be invisible to CUDA graphs

### Files changed

| File | Change |
|------|--------|
| `runtime/s2pro_sglang_ar.py` | `S2ProFastGraphRunner`: use eager fn, `top_k` as int (removed `_top_k_buf`), `replay()` drops `top_k` param |
| `runtime/s2pro_sglang_ar.py` | `_batched_two_stage_decode`: pass int `top_k` to codebook fn, update `replay()` call |
| `runtime/s2pro_ar.py` | `_sample_with_topk` signature: `top_k: int` (was `Union[int, Tensor]`) |

---

## P2: Benchmark Updates

Added CLI flags to `benchmarks/profile_s2pro_sglang.py`:

- `--enable-cuda-graph`: enables CUDA graph capture for codebook loop
- `--max-batch-size`: controls max batch size for engine and graph capture (default 64)

---

## Test Commands

```bash
export S2PRO_CKPT=/root/.cache/huggingface/s2-pro/s2-pro
export SEED_TTS=/tmp

# --- P0: compile + topk optimization ---
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_p0_v2 \
    --max-samples 50 --batch-sizes 1 --save-audio

python benchmarks/eval_wer.py \
    --meta $SEED_TTS/seedtts_testset/en/meta.lst \
    --audio-dir results/s2pro_p0_v2/audio --lang en

# --- P1: CUDA graph + batched ---
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_p1_v2 \
    --max-samples 50 --batch-sizes 1,2,4,8 --enable-cuda-graph --save-audio

# --- No compile baseline (for comparison) ---
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_nocompile \
    --max-samples 50 --batch-sizes 1 --no-compile --save-audio
```

## Expected Results

| Config | tok/s (BS=1) | WER | Status |
|--------|-------------|-----|--------|
| compile + topk (P0 v2) | ~60 | ~1% | To verify |
| compile + CUDA graph (P1 v2) | ~60 (BS=1), near-linear batch scaling | ~1% | To verify |
| No compile (baseline) | ~26 | ~1% | Reference |

## Fish-style 2-D `input_ids` Architecture

> Date: 2026-03-08
> Commit: feat(s2pro): fish-style 2-D input_ids with inline VQ embedding

### Problem: `input_embeds` injection blocks text model CUDA graph

The previous design injected precomputed **float** `input_embeds` into the model's forward pass:
- **Prefill**: Looked up VQ-masked positions, called `audio_decoder.embed_text_dim()`, scattered results into `text_embeds`
- **Decode**: Called `audio_decoder.embed_one_token()` per semantic token, scattered into `text_embeds`

This required `model_worker_batch.input_embeds` (a dynamic float tensor) which is **incompatible with CUDA graph capture** — CUDA graphs need static tensor shapes and integer inputs for embedding lookups.

### Solution: Move VQ embedding inside the model

Adopted the "fish-style" architecture from FishAudio's official implementation:

1. **2-D integer `input_ids [N, K+1]`**: Column 0 = semantic token ID, columns 1..K = raw codebook indices
2. **Inline `_embed_with_codebooks()`** in the text model's forward:
   - `text_embeds = embed_tokens(input_ids[:, 0])`
   - At VQ positions (semantic ID in `[begin, end]`): sum K codebook embeddings + text embedding, scale by `1/sqrt(K+1)`
3. **`codebook_embeddings`** weight (`nn.Embedding(codebook_size * K, hidden_size)`) loaded from `audio_decoder.codebook_embeddings.weight`

### Data flow

```
Tokenizer (build_prompt)
  → input_values [K+1, seq_len]    # new: multi-codebook tensor
  → input_ids [seq_len]            # kept: 1-D semantic IDs for SGLang

S2ProState / engine_io
  → transpose to [seq_len, K+1]

S2ProSGLangModelRunner._build_2d_input_ids()
  Prefill: slice each request's input_values by prefix_len/extend_input_len
  Decode:  semantic_token || _last_codebook_values  →  [bs, K+1]

Text model forward()
  input_ids.dim() == 2  →  _embed_with_codebooks()
  input_ids.dim() == 1  →  embed_tokens() (fallback)
```

### Files changed

| File | Change |
|------|--------|
| `sglang_model.py` | Added `codebook_embeddings`, `_embed_with_codebooks()`, 2-D `input_ids` handling in `forward()`, weight loading for `audio_decoder.codebook_embeddings` |
| `tokenizer.py` | `build_prompt()` now returns `input_values [K+1, seq_len]` |
| `io.py` | Added `input_values` field to `S2ProState`, serialization support |
| `pipeline/engine_io.py` | Transpose `input_values` from `[K+1, seq]` to `[seq, K+1]` for request data |
| `pipeline/stages.py` | Pass `input_values` from prompt to `S2ProState` |
| `runtime/s2pro_sglang_ar.py` | Added `input_values` to `S2ProSGLangRequestData`, replaced `_inject_vq_embeds()` with `_build_2d_input_ids()` |
| `benchmarks/profile_s2pro_sglang.py` | Pass `input_values` to request data |

### Results

**Quality** (50 samples, seed-tts-eval EN):

| Mode | WER (mean) | WER (median) |
|------|-----------|-------------|
| no-compile (fish-style) | 0.99% | 0.0% |
| compile (fish-style) | 1.10% | 0.0% |

No quality regression — identical to pre-fish-style results.

**Performance** (compile mode, fish-style):

| Metric | BS=1 | BS=2 | BS=4 | BS=8 |
|--------|------|------|------|------|
| tok/s (per-req) | 52.6 | 28.3 | 27.0 | 25.6 |
| tok/s (agg) | 52.6 | 28.5 | 27.2 | 25.8 |
| RTF | 0.409 | 0.771 | 0.802 | 0.841 |
| Latency | 1.58s | 2.95s | 3.11s | 3.28s |
| TTFT | 25.6ms | 41.8ms | 42.5ms | 43.9ms |
| TTFB | 207.7ms | 395.5ms | 395.0ms | 399.2ms |

Comparison with previous no-compile (fish-style) at BS=4/8:

| | BS=4 no-compile | BS=4 compile | BS=8 no-compile | BS=8 compile |
|---|---|---|---|---|
| tok/s | 26.0 | 27.0 (+4%) | 24.3 | 25.6 (+5%) |
| RTF | 0.830 | 0.802 | 0.889 | 0.841 |

Performance is equivalent or slightly better. The core value is architectural: all model inputs are now integer tensors, unblocking text model CUDA graph capture.

---

## Key Design Decisions

1. **topk vs sort for sampling**: `topk` is O(V + k log k) vs sort's O(V log V). For codebook V=4096, k=30, topk is ~5x fewer FLOPs. Both are CUDA-graph-safe; the old sort approach was chosen for "safety" but was unnecessarily conservative.

2. **Eager fn for CUDA graph capture**: torch.compile + CUDA graph is redundant. CUDA graphs already eliminate kernel launch overhead. Using eager avoids Dynamo's fake tensor tracing issues with complex model internals (e.g., `project_in` shape changes).

3. **top_k as int constant**: Since all requests use the same top_k value and `torch.topk` requires a Python int, baking it as a constant simplifies the code and avoids `.item()` sync issues in compiled/CUDA-graph paths.

4. **Fish-style 2-D `input_ids`**: Moving VQ embedding logic inside the model's forward pass keeps all inputs as integer tensors. This is the prerequisite for text model CUDA graph capture — the biggest remaining performance gap vs FishAudio's official ~120 tok/s at BS=1.

## Text Model CUDA Graph

> Date: 2026-03-08
> Commit: feat(s2pro): CUDA graph capture/replay for text model decode

### Problem: High CPU-side kernel launch overhead during decode

Each text model decode step (36 transformer layers with RadixAttention) launches hundreds of CUDA kernels individually. CPU-side launch overhead dominates at small batch sizes, capping throughput at ~52 tok/s (BS=1).

### Solution: Custom `S2ProTextCudaGraphRunner`

SGLang's built-in `CudaGraphRunner` expects 1-D `input_ids` (`GraphInputBuffers.input_ids` is `[max_num_token]`). Our fish-style architecture uses 2-D `input_ids [bs, K+1]`. A custom graph runner was implemented:

1. **Static buffers**: `input_ids [max_bs, K+1]`, `req_pool_indices [max_bs]`, `seq_lens [max_bs]`, `out_cache_loc [max_bs]`, `seq_lens_cpu [max_bs]`
2. **Capture**: For each batch size, build `ForwardBatch` with static buffers, run warmup, capture CUDA graph
3. **Replay**: Copy live data into static buffers, init attention metadata, `graph.replay()`, slice output to actual batch size

Key implementation details:
- **Positions computed inside graph**: `positions = (seq_lens - 1).clamp(min=0).to(int64)` is part of the captured computation so it updates when `seq_lens` changes during replay
- **Branchless `_embed_with_codebooks`**: Removed `vq_masks.any()` (CPU-GPU sync) and float-typed scale computation that caused `rmsnorm` dtype errors; uses `vq_float * (inv_scale - 1.0) + 1.0` to stay in bf16
- **FlashInfer compatibility**: Passes `seq_lens_cpu` to `init_forward_metadata_replay_cuda_graph` (required by FlashInfer backend, which SGLang auto-selects on H100)
- **Memory**: `mem_fraction_static` reduced from 0.85 to 0.75 when text CUDA graph is enabled to leave headroom for graph memory pools

### Bugs fixed during implementation

1. **`vq_masks.any()` → CPU-GPU sync during capture**: Removed branch; always compute VQ embeddings with masking
2. **Float32 dtype contamination**: `bool_tensor * python_float` produces float32, breaking `rmsnorm`. Fixed by casting mask to `text_embeds.dtype` first
3. **Static positions**: Positions computed outside `run_once()` were baked at capture time, causing all decode steps to use the same position → model never generated EOS (2048 tokens every request). Fixed by computing positions inside `run_once()` from the static `seq_lens` buffer
4. **Missing `seq_lens_cpu`**: FlashInfer backend (auto-selected on H100) requires `seq_lens_cpu` in `init_forward_metadata_replay_cuda_graph`. Added pre-allocated CPU buffer

### Files changed

| File | Change |
|------|--------|
| `runtime/s2pro_sglang_ar.py` | New `S2ProTextCudaGraphRunner` class (capture/replay), `S2ProSGLangModelRunner` uses it for decode steps |
| `sglang_model.py` | `_embed_with_codebooks` made branchless and bf16-safe |
| `factory.py` | `use_text_cuda_graph` flag, instantiates and captures graph runner |
| `benchmarks/profile_s2pro_sglang.py` | `--enable-text-cuda-graph` CLI flag, reduced `mem_fraction_static` for graph mode |

### Results

**Quality** (50 samples, seed-tts-eval EN):

| Mode | WER (mean) | WER (median) |
|------|-----------|-------------|
| text CUDA graph | 1.61% | 0.0% |

Quality maintained — no samples >50% WER.

**Performance** (text CUDA graph, no compile for audio decoder):

| Metric | BS=1 | BS=2 | BS=4 | BS=8 |
|--------|------|------|------|------|
| tok/s (per-req) | **79.5** | 35.1 | 33.0 | 31.0 |
| tok/s (agg) | 73.4 | 35.4 | 33.4 | 31.2 |
| RTF | **0.301** | 0.624 | 0.659 | 0.696 |
| Latency | 1.16s | 2.37s | 2.55s | 2.74s |
| TTFT | 136ms | 42.8ms | 44.2ms | 45.9ms |
| TTFB | 258ms | 336ms | 338ms | 342ms |

**Comparison** (fish-style compile baseline → text CUDA graph):

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| BS=1 tok/s | 52.6 | **79.5** | **+51%** |
| BS=2 tok/s | 28.5 | 35.1 | +23% |
| BS=4 tok/s | 27.2 | 33.0 | +21% |
| BS=8 tok/s | 25.8 | 31.0 | +20% |
| BS=1 RTF | 0.409 | **0.301** | **-26%** |
| BS=8 RTF | 0.841 | 0.696 | -17% |

### Test command

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_text_cuda_graph \
    --max-samples 50 --batch-sizes 1,2,4,8 \
    --enable-text-cuda-graph --max-batch-size 16 --save-audio
```

---

## Gap Analysis: vs FishAudio Official (H200)

> FishAudio S2 official numbers (single H200 GPU):
> - RTF: 0.195
> - TTFA: ~100 ms
> - Throughput: 3,000+ acoustic tokens/s (RTF < 0.5)
> - WER (EN): 0.99%

### Current comparison

| Metric | FishAudio Official (H200) | Ours (H100, text CUDA graph) | Gap |
|--------|--------------------------|------------------------------|-----|
| RTF (BS=1) | **0.195** | 0.301 | 1.54x slower |
| TTFA / TTFB | **~100 ms** | 258 ms | 2.6x slower |
| Throughput (agg) | **3,000+ tok/s** | 73.4 (BS=1) / ~248 (BS=8) | 12x gap |
| WER (EN) | **0.99%** | 1.61% | Slightly higher |

### Optimization progress

| Stage | BS=1 tok/s | BS=1 RTF | Δ vs baseline |
|-------|-----------|----------|---------------|
| Baseline (eager, no opt) | ~26 | ~0.83 | - |
| P0: compile + topk sampling | ~52 | ~0.41 | +100% |
| Fish-style 2-D input_ids | ~52.6 | ~0.41 | Architecture prep |
| **Text model CUDA graph** | **79.5** | **0.301** | **+206%** |
| FishAudio official target | ~120+ | 0.195 | ~50% gap remaining |

### Root causes of remaining gaps

**1. Batch scaling degradation (biggest problem)**

BS=1 → BS=2: per-request tok/s drops from 79.5 to 35.1 (-56%). This means BS=2 aggregate (70.2) is actually *lower* than BS=1 (73.4). The audio decoder codebook loop runs serially across all 10 codebooks per batch, and its cost grows linearly with batch size. FishAudio achieves 3000+ tok/s at RTF < 0.5 under high concurrency, implying far better batch scaling.

**2. torch.compile disabled inside text CUDA graph**

`patch_model` calls `_to_torch()` which converts weights to fp32. The custom `fused_inplace_qknorm` kernel only supports bf16/fp16, causing a crash. FishAudio likely uses a compatible qk_norm or skips `_to_torch()`. Without compile, we miss kernel fusion optimizations (estimated ~20-30% gap).

**3. TTFB/TTFA gap (258ms vs 100ms)**

The first audio chunk requires: prefill → semantic token → 10 codebook tokens → DAC decode. The codebook loop alone takes ~120ms. FishAudio may overlap prefill with codebook generation, or use a faster codebook decoder path.

**4. Hardware difference (H200 vs H100)**

H200 memory bandwidth: 4.8 TB/s vs H100: 3.35 TB/s (+43%). For memory-bound decode, this accounts for ~30% of the RTF gap. Adjusting for hardware, our effective RTF would be ~0.21 on H200, closer to their 0.195.

---

## TODO: Next Optimizations

| Priority | Task | Difficulty | Expected Impact | Details |
|----------|------|-----------|----------------|---------|
| **P0** | torch.compile inside text CUDA graph | Medium | BS=1: 79 → ~100+ tok/s | Fix `fused_inplace_qknorm` fp32 issue or bypass `_to_torch()` dtype conversion |
| **P1** | Combined text+audio CUDA graph | Medium | Reduce TTFB ~50ms, improve batch scaling | Capture codebook loop inside same graph as text model decode |
| **P2** | Audio decoder batch parallelism | Hard | Better batch scaling (BS=8 agg: 248 → ~500+ tok/s) | Current serial codebook loop is the batch scaling bottleneck |
| **P3** | Text/audio overlap scheduling | Hard | Hide codebook latency behind next text step | Pipeline text model step N+1 while audio decoder runs step N |
| **P4** | Profile-guided compile tuning | Medium | ~10-15% kernel fusion gains | Tune `SGLANG_TORCH_COMPILE_MODE`, investigate `fullgraph=True` |
| **P5** | Benchmark on H200 | Easy | Fair comparison with official numbers | H200 bandwidth advantage is ~43%, expected RTF ~0.21 |
