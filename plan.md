# Talker Integration Fix Plan

## 1. RFC #16546 — SGLang-Omni Design Summary

Source: https://github.com/sgl-project/sglang/issues/16546

### 1.1 Core Architecture: Stage-Based Pipeline

The RFC defines a stage-based pipeline architecture where each omni model is
decomposed into discrete processing stages. Each stage is independently
scheduled and executed.

**Class Diagram** (Diagram 1 from RFC):

```
Coordinator
  -stages: Dict<str, StageEndpoint>
  -active_requests: Set<str>
  -entry_stage: str
  +register_stage(name, endpoint)
  +submit(req_id, request)
  +on_request_start(req_id)
  +on_request_complete(req_id)
  +on_request_fail(req_id, error)
  +abort(req_id)
       │ 1
       │ registers
       ▼ *
Stage
  -name: str
  -scheduler: StageScheduler
  -get_next: Callable               ← decides where output goes next
  -data_plane: DataPlaneClient
  -control_plane: ControlPlaneClient
  +run()
  +receive() : Input
  +send(next_stage, req_id, output)
  +on_abort(req_id)
       │
       ├── has 1 ──► StageScheduler
       │               -workers: List<Worker>
       │               -policy: SchedulingPolicy
       │               -pending_queue: Queue
       │               +execute(inputs) : List<Output>
       │               +add_worker(worker)
       │               +remove_pending(req_id)
       │                    │              │
       │                uses 1          manages *
       │                    │              │
       │               SchedulingPolicy   Worker (abstract)
       │               (abstract)         +execute(batch) : List<Output>
       │               +select_batch(queue, workers)
       │                    │
       │               ┌────┴────┐
       │         RoundRobinPolicy  CacheAwarePolicy
       │         -max_batch_size   -radix_tree
       │                           -max_batch_size
       │
       ├── uses 1 ──► ControlPlaneClient
       │               +send_data_ready(msg)
       │               +recv_data_ready() : DataReadyMessage
       │               +send_abort(msg)
       │
       └── uses 1 ──► DataPlaneClient
                       +put(key, tensor) : bool
                       +get(key) : Tensor
                       +cleanup(req_id)
```

Key design points:
- **Worker is abstract**: each stage provides its own Worker implementation.
  For AR stages → SGLang ModelWorker. For encoders → batch encoder. Etc.
- **SchedulingPolicy is pluggable**: RoundRobin for simple stages, CacheAware
  for stages with KV cache (like AR decoders).
- **get_next** is a simple callable that decides routing: next stage, END,
  or cyclic back to a previous stage.

### 1.2 Component Diagram (Diagram 2 from RFC)

```
                    ┌─────────────────┐
                    │   API Layer      │
                    │   API Server     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
        Control     │   Coordinator    │
        Plane       └──┬──────┬──────┬┘
        (ZMQ)          │      │      │
              ┌────────▼──┐ ┌─▼──────▼────┐ ┌────────────┐
              │  Stage 1   │ │  Stage 2     │ │  Stage 3   │
              │ Scheduler  │ │ Scheduler    │ │ Scheduler  │
              │ get_next   │ │ get_next     │ │ get_next   │
              │ Workers    │ │ Workers      │ │ Workers    │
              └──────┬─────┘ └──────┬──────┘ └─────┬──────┘
                     │              │               │
              ◄──────┴──────────────┴───────────────┴──────►
                    Data Plane (Mooncake / NIXL)
```

- API Server only talks to Coordinator.
- Coordinator dispatches to stages via Control Plane (ZMQ).
- Stages exchange tensor data via Data Plane (Mooncake / shared memory).
- Each stage has its own Scheduler + Workers.

### 1.3 Sequence Diagrams (Diagrams 3-5 from RFC)

**Normal Flow** (Stage 1 → Stage 2 → Stage 3 → END):
1. API Server → Coordinator: request req_123
2. Coordinator: on_request_start → submit to Stage 1
3. Stage 1: execute → PUT output to Data Plane → get_next returns "Stage 2"
   → send DataReadyMessage to Stage 2
4. Stage 2: GET input from Data Plane → execute → PUT output → get_next
   returns "Stage 3" → send DataReadyMessage to Stage 3
5. Stage 3: GET input → execute → PUT output → get_next returns "END"
   → on_request_complete → result back to API Server

**Early End** (Stage 2 → END, Stage 3 not involved):
- Same as normal but Stage 2's get_next returns "END" directly.
  (e.g., text-only output skips talker/codec stages)

**Abort**: Coordinator broadcasts AbortMessage to all stages in parallel.
Each stage: remove_pending → cleanup Data Plane.

### 1.4 Data Flow Patterns (Diagram 6 from RFC)

All patterns supported via get_next() logic:

```
Cyclic:     Stage 1 ⟲ Stage 2 → END
Fan-out:    Stage 1 → Stage 2a + Stage 2b → END
Early End:  Stage 1 → Stage 2 → (Stage 3 | END)
Linear:     Stage 1 → Stage 2 → Stage 3 → END
```

### 1.5 Multi-Instance Scaling

Each stage can become a bottleneck independently. Solution: horizontally
scale individual stages using SGLang Router with load balancing across
multiple instances of the same stage.

### 1.6 Flexibility Principle

> "Since the model architecture paradigm has not converged yet, and it might
> not even converge in 2026, we must uphold flexibility as the first-class
> principle when designing this system."

Stages are defined via YAML config. Each stage independently chooses its
scheduling policy, device placement, and Worker implementation.


## 2. Qwen3-Omni Model Architecture

Source: [Technical Report](https://arxiv.org/abs/2509.17765),
[HF docs](https://huggingface.co/docs/transformers/main/model_doc/qwen3_omni_moe),
[vLLM-Omni config](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml)

### 2.1 Overall Pipeline

```
Vision Encoder ─┐
                 ├──► Thinker (28-layer MoE, AR text decoder)
Audio Encoder  ─┘         │
                           │ thinker_embed + thinker_hidden
                           ▼
                      Talker Backbone (20-layer MoE, AR codec decoder)
                           │ per decode step:
                           │   codec_head → sample → layer-0 code
                           │   Code Predictor → layers 1..15
                           ▼
                      Code2Wav (causal ConvNet) → waveform
```

### 2.2 Thinker

- 28-layer MoE transformer (128 experts, top-8), hidden_size=2048.
- Autoregressively generates text tokens (continuous batching, KV cache).
- Exports two tensors for the Talker:
  - **thinker_embed**: embedding layer output (layer -1)
  - **thinker_hidden**: hidden states from decoder layer 24

### 2.3 Talker Backbone

- 20-layer MoE transformer (128 experts, top-6, **with shared expert**).
- hidden_size=1024, head_dim=128, 16 attn heads, 2 KV heads.
- Uses `codec_embedding` (vocab=3072) instead of text embeddings.
- MoE block = routed experts + shared expert (gated).
- Input: projected thinker outputs via ResizeMLP (2048 → 1024):
  - Text positions: `text_projection(thinker_embed)`
  - Multimodal positions: `hidden_projection(thinker_hidden)`

**CRITICAL: The Talker Backbone is an autoregressive decoder.**

Evidence from vLLM-Omni's stage config:
- `worker_type: "ar"` (autoregressive)
- `scheduler_cls: "OmniARScheduler"` (AR scheduler with continuous batching)
- `max_tokens: 4096` (generates up to 4096 codec tokens)
- `stop_token_ids: [2150]` (codec EOS)
- `temperature: 0.9, top_k: 50` (sampling, not argmax)
- `gpu_memory_utilization: 0.6` (needs KV cache memory)

The Talker generates codec tokens **one at a time**, exactly like the
Thinker generates text tokens. At each decode step t:
1. Backbone forward: input_id = previous codec token → hidden state h_t
   (using KV cache from steps 0..t-1)
2. `codec_head(h_t)` → sample with temperature/top_k → layer-0 code
3. Code Predictor: `[h_t, embed(layer0_code)]` → residual codes layers 1..15
4. All 16 codes for frame t → Code2Wav for streaming audio synthesis
5. Repeat until codec EOS token or max_tokens

### 2.4 Code Predictor (MTP Module)

- 5-layer dense transformer (~80M params).
- hidden_size=1024, intermediate_size=3072, 16 heads, 8 KV heads.
- Called **once per Talker decode step** — NOT a separate pipeline stage.
- Fixed-step autoregressive loop (15 iterations, sequence length 2→17):
  1. Input: `[talker_hidden_at_step, embed(layer0_code)]` (length 2)
  2. For layer_idx in 1..15: forward → lm_head[layer_idx] → sample → embed → append
  3. Output: 15 residual codes + summed embeddings

### 2.5 Mapping to RFC Stages

```
RFC Stage          │ Qwen3-Omni Component     │ Worker Type    │ Scheduling
───────────────────┼──────────────────────────┼────────────────┼──────────────
Stage 0 (Encode)   │ Vision + Audio Encoders   │ Batch encoder  │ RoundRobin
Stage 1 (Thinker)  │ Thinker (28L MoE)        │ SGLang AR      │ CacheAware
Stage 2 (Talker)   │ Talker Backbone (20L MoE) │ SGLang AR      │ CacheAware
                   │ + Code Predictor (5L)     │ (per-step hook)│
Stage 3 (Code2Wav) │ Code2Wav ConvNet          │ Custom         │ RoundRobin
```

The Code Predictor is NOT a separate stage. It runs within the Talker stage's
Worker, invoked at each decode step after the backbone produces a layer-0 code.
This is how vLLM-Omni implements it: `talker_mtp()` is called per-step
inside the Talker's generation loop.


## 3. Current Class Hierarchy in This Codebase

```
Qwen3OmniTalker (talker.py — top-level nn.Module)
├── text_projection: ResizeMLP        (thinker 2048 → talker 1024)
├── hidden_projection: ResizeMLP      (thinker 2048 → talker 1024)
├── model: Qwen3OmniMoeTalkerTextModel         ← TALKER BACKBONE
│   ├── codec_embedding: nn.Embedding(3072, 1024)
│   ├── layers[0..19]: Qwen3OmniMoeTalkerDecoderLayer
│   │   └── inherits Qwen3OmniMoeThinkerTextDecoderLayer
│   │       ├── self_attn: Qwen3OmniMoeThinkerTextAttention
│   │       │     [SGLang: RadixAttention, QKVParallelLinear, RoPE]
│   │       └── mlp: Qwen3OmniMoeTalkerSparseMoeBlock
│   │             [SGLang: FusedMoE + SharedExpert]
│   └── norm: RMSNorm
├── codec_head: ReplicatedLinear(1024, 3072)    ← layer-0 logits
└── code_predictor: Qwen3OmniMoeTalkerCodePredictor  ← CODE PREDICTOR
    ├── model.codec_embedding[0..14]: nn.Embedding(2048, 1024)
    ├── model.layers[0..4]: nn.Module            ← hand-assembled layers
    │   ├── self_attn: _CausalSelfAttention      ← pseudo-implementation
    │   ├── mlp: _CodePredictorMLP               ← pseudo-implementation
    │   ├── input_layernorm: RMSNorm
    │   └── post_attention_layernorm: RMSNorm
    ├── model.norm: RMSNorm
    └── lm_head[0..14]: ReplicatedLinear(1024, 2048)
```

Runtime layer (factory.py):
```
create_talker_codec_engine()
└── OmniEngine
    ├── Scheduler
    │   ├── _SingleRequestBatchPlanner      ← custom, no batching
    │   ├── SimpleResourceManager
    │   └── SinglePassIterationController   ← single-pass, no AR loop
    └── TalkerCodecRunner                   ← custom runner
        └── calls Qwen3OmniTalker with:
            ├── _MockForwardBatch (30+ stubbed fields)
            ├── patch_talker_attention() (monkey-patches RadixAttention → _SDPAWrapper)
            └── _load_checkpoint_weights() (manual safetensors loading)
```


## 4. What Is Actually Problematic

### 4.1 FUNDAMENTAL: Talker Backbone execution model is wrong (CRITICAL)

**The Talker Backbone is treated as a single-pass model, but it is actually
an autoregressive decoder.**

Current behavior in `TalkerCodecRunner._generate_codec()`:
1. `_run_backbone()` — runs the FULL backbone forward on the entire input
   sequence in ONE pass (no KV cache, no AR loop)
2. `logits.argmax(dim=-1)` — takes argmax for ALL positions at once
   (no sampling, no temperature)
3. `code_predictor_forward()` — generates residual codes for all positions

Correct behavior (per the actual Qwen3-Omni architecture):
1. Prefill: process projected thinker embeddings as context (with KV cache)
2. Decode loop (until codec EOS or max_tokens):
   a. Backbone forward: single token → h_t (using KV cache)
   b. `codec_head(h_t)` → sample with temperature=0.9, top_k=50 → layer-0 code
   c. Code Predictor: `[h_t, embed(layer0_code)]` → residual codes for frame t
   d. Feed layer-0 code back as next input token
   e. Stream all 16 codes for frame t to Code2Wav

This is not a cosmetic issue — it produces fundamentally different outputs:
- Single-pass = processes the input but doesn't GENERATE anything new
- Autoregressive = generates a variable-length sequence of codec tokens
- argmax ≠ sampling with temperature=0.9, top_k=50

### 4.2 Code Predictor is a separate engine, but should be per-step (CRITICAL)

Current: `create_talker_codec_engine()` creates a separate `OmniEngine` with
its own `Scheduler` for the Code Predictor + Talker Backbone combined.

Per the RFC and vLLM-Omni: the Code Predictor is NOT a separate pipeline stage.
It runs WITHIN the Talker stage's Worker, called at each decode step after
the backbone produces a layer-0 code. The Talker Backbone + Code Predictor
together form a single Stage with a single Worker.

In the RFC's terms:
- Talker = one **Stage** with one **StageScheduler**
- The backbone is the AR Worker (like ModelWorker)
- The Code Predictor is invoked per-step within that Worker's execute() logic
- They share the same scheduling policy (CacheAware for the backbone's KV cache)

### 4.3 Code Predictor internals: pseudo-implementation (HIGH)

The `Qwen3OmniMoeTalkerCodePredictor` layers use hand-written components:

| Component | Current (pseudo) | Should Use |
|-----------|-----------------|------------|
| Attention | `_CausalSelfAttention` — hand-written RoPE, `nn.Linear` QKV, raw SDPA | `QKVParallelLinear`, `get_rope()`, SGLang attention backend |
| MLP | `_CodePredictorMLP` — 3 separate `ReplicatedLinear` | `Qwen3OmniMoeTalkerDenseMLP` (already exists: `MergedColumnParallelLinear` + `RowParallelLinear` + `SiluAndMul`) |
| Layer | Ad-hoc `nn.Module()` + attribute assignment | Proper `DecoderLayer` class |
| Weight loading | Custom `cp_qkv_buffer` dict for manual QKV concat | Standard `stacked_params` path |

~160 lines of custom code (talker.py:344-601) reimplements what SGLang provides.

### 4.4 Runtime workaround code in factory.py (HIGH)

~200 lines of workaround code exist because the Talker isn't run through
ModelWorker:

- `TalkerCodecRunner` — custom model runner
- `_MockForwardBatch` — 30+ stubbed fields to fake a ForwardBatch
- `_SDPAWrapper` — replaces RadixAttention with plain SDPA
- `_ExtendForwardMode` — stub forward mode
- `patch_talker_attention()` — monkey-patches attention modules
- `_SingleRequestBatchPlanner` — one-request-at-a-time planner
- `_load_checkpoint_weights()` — manual safetensors loading
- `_load_talker_config()` — manual config loading

All of these should be deleted once the Talker runs through ModelWorker.

### 4.5 What is NOT problematic

- **`Qwen3OmniMoeTalkerTextModel`** — the backbone model class uses proper
  SGLang components. No changes needed to the model itself.
- **`Qwen3OmniMoeTalkerDecoderLayer`** — correctly inherits from Thinker's
  decoder layer, only overrides MLP with shared-expert MoE.
- **`Qwen3OmniMoeTalkerSparseMoeBlock`** — correctly extends Thinker's MoE.
- **`ResizeMLP`**, **`DenseMLP`**, **`SharedExpertMLP`** — all properly use
  SGLang's parallel linear layers.
- **Config classes** — all correct.
- **`Qwen3OmniTalker.prepare_input_embeds()`** — correct projection logic.
- **`Qwen3OmniTalker.forward()`** — correct signature, delegates to TextModel.
- **`Qwen3OmniTalker.compute_logits()`** — correct, uses codec_head.
- **`Qwen3OmniTalker.load_weights()`** — mostly correct for the backbone;
  only the `cp_qkv_buffer` section for Code Predictor is problematic.


## 5. Resolution Goals

### Goal 1: Run the Talker as an AR stage through SGLang's ModelWorker

The Talker Backbone is an autoregressive decoder. It must run through
`ModelWorker` with KV cache, continuous batching, and proper sampling,
just like the Thinker runs via `create_sglang_ar_engine()`.

Specifically:
- Register `Qwen3OmniTalker` as a model that `ModelWorker` can load and run.
- Use `create_sglang_ar_engine()` (or a close variant) for the Talker stage.
- Configure AR generation: temperature=0.9, top_k=50, stop_token=codec_eos.
- This provides: KV cache, batched decode, CUDA graphs, `ForwardBatch`.
- Let `ModelWorker` handle weight loading via `load_weights()`.

### Goal 2: Integrate Code Predictor as synchronous post-step within Talker Worker

The Code Predictor is a separate ~80M-param model that runs once per Talker
decode step. It CANNOT be part of the backbone's CUDA graph (different model,
different shapes). It runs synchronously after each decode step in eager mode.

#### Data-flow property that makes this safe

The Code Predictor's output (residual codes) flows DOWNSTREAM to Code2Wav.
It does NOT feed back into the backbone. The backbone only needs layer0_code:

```
backbone: token_in → forward → logits → sample → layer0_code → feed back
                                              ↓
code predictor:              (h_t, layer0_code) → residual codes → Code2Wav
```

#### Per-step execution sequence

```
for each decode step:
    1. backbone forward (CUDA-graphed)  →  logits, h_t
    2. sample  →  layer0_code
    3. h_t = hidden_states.clone()      ← copy out of graph buffer
    4. code_predictor(h_t, layer0_code) ← eager mode, synchronous
    5. emit (layer0_code, residual_codes) to downstream
    6. feed layer0_code back as next input
```

#### Where to hook into SGLang

The integration point is `SGLangModelRunner.execute()` in sglang_ar.py.
Currently the method does:

```python
def execute(self, scheduler_output):
    schedule_batch = self.batch_planner._cached_schedule_batch
    model_worker_batch = schedule_batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, ...)
    batch_result = self.model_worker.forward_batch_generation(forward_batch)
    # ... hidden hook, sampling, output processing ...
```

After `forward_batch_generation()` returns:
- `batch_result.logits_output.hidden_states` contains h_t (if
  `capture_hidden_mode` is enabled — this is how the Thinker already
  captures `thinker_embed` and `thinker_hidden`).
- `batch_result.next_token_ids` contains the sampled layer0_codes.

The Code Predictor call is inserted AFTER sampling and BEFORE the output
is returned to the scheduler. This is analogous to how the existing
`_hidden_hook.pop_captured()` + `_store_captured_hidden()` already runs
between forward and output — just with an additional model forward.

#### CUDA graph compatibility

1. **h_t extraction**: Use SGLang's `capture_hidden_mode` to get
   `hidden_states` from `LogitsProcessorOutput`. This works for both
   eager forward (prefill) and CUDA graph replay (decode) — the graph
   runner already returns `output.hidden_states[:raw_num_token]`.
   The existing `HiddenStateCaptureHook` uses `register_forward_hook`
   which does NOT fire during graph replay — for the Talker, we must
   use `capture_hidden_mode` instead.

2. **Buffer lifetime**: After graph replay, `hidden_states` lives in
   graph-owned memory. Must `.clone()` before the Code Predictor runs
   (Code Predictor's eager forward may trigger allocations that invalidate
   graph memory). The existing `_store_captured_hidden()` already does
   `tensor[start:end].clone()`.

3. **Code Predictor memory**: Parameters (~160MB in bf16) are allocated
   in regular GPU memory before graph capture. They are not part of the
   graph memory pool. Eager-mode activations are allocated/freed normally.

4. **Batch dimension**: In decode mode, each request = 1 token, so h_t
   per request is `[1, 1024]`. The Code Predictor can batch across all B
   requests in the batch: stack h_t vectors into `[B, 1, 1024]` and run
   one batched `code_predictor_forward()`.

5. **Skip conditions**: Code Predictor must NOT run when:
   - `schedule_batch.is_prefill_only` is True (no tokens sampled yet)
   - A request just finished (codec EOS) — its layer0_code is meaningless
   - `batch_result.next_token_ids` is zeroed out (prefill placeholder)

### Goal 3: Rewrite Code Predictor layers with proper building blocks

Replace pseudo-implementations with SGLang components:
- **Attention**: proper class using `QKVParallelLinear`, `get_rope()`, and
  QK-norm (per vLLM-Omni reference). The Code Predictor runs in eager mode
  with max sequence length 17 — plain SDPA is acceptable as the attention
  backend. But the linear layers and RoPE should use SGLang's implementations
  for weight loading compatibility and future TP support.
- **MLP**: reuse `Qwen3OmniMoeTalkerDenseMLP` (already exists in talker.py,
  uses `MergedColumnParallelLinear` + `RowParallelLinear` + `SiluAndMul`).
- **Decoder Layer**: create a proper `Qwen3OmniMoeTalkerCodePredictorDecoderLayer`
  class instead of ad-hoc `nn.Module()` assembly.
- **Weight loading**: remove `cp_qkv_buffer` special case — use standard
  `stacked_params` path. With `QKVParallelLinear`, the existing shard_id
  logic handles the separate q/k/v checkpoint weights automatically.
- Delete ~160 lines of `_CausalSelfAttention`, `_CodePredictorMLP`,
  `_build_rope_cache`, `_apply_rotary_emb`.

### Goal 4: Delete all runtime workaround code

Once Goals 1-3 are achieved, delete from factory.py:
- `TalkerCodecRunner`
- `_MockForwardBatch`
- `_SDPAWrapper`
- `_ExtendForwardMode`
- `patch_talker_attention()`
- `_SingleRequestBatchPlanner`
- `_load_checkpoint_weights()`
- `_load_talker_config()`
- `create_talker_codec_engine()`

### Non-goals (out of scope)

- KV cache for the Code Predictor's internal loop (max 17 tokens).
- Async/stream-overlapped Code Predictor execution (future optimization).
- Separate RFC stage for Code Predictor (future architecture).
- Tensor parallelism for the Talker.
- Code2Wav integration.
- Data Plane (Mooncake/NIXL) for inter-stage tensor transfer.
