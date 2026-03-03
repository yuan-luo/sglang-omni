# SGLang Omni - Talker Integration Development Plan

> Generated: 2026-03-03
> RFC: https://github.com/sgl-project/sglang/issues/16546
> Branch: `talker`

---

## 0. Code Style Requirements (MANDATORY)

**所有代码编写、修改、重构都必须严格遵守 `code-style-agent` 定义的代码风格规范。**
**优先级层次：Correctness (P0) > Performance (P1) > Maintainability (P2) > Style (P3) > Process (P4)。**

Agent 配置位于 `/data/chenyang/.claude/agents/code-style-agent.md`，执行任何编码任务时必须调用该 agent 进行审查。

### P0 — Correctness（最高优先级）

- **Fail Fast**: 用 `assert` 检查程序员不变量，用 `raise ValueError/TypeError` 检查用户输入。在 public API 入口和系统边界做验证，internal helpers 可以假设输入有效。
- **Early return / guard clause**: 函数顶部用 guard clause，禁止深层嵌套 if-else。
- **禁止 over-protect**: 如果操作 99% 正确，不要加防御性处理。LLM 倾向于 over-protect，必须抵制。
- **禁止 over-catch**: `try/except` 必须有明确、窄小的保护区域。禁止用一个 `try/except` 包裹大段代码。
- **线程安全**: 显式标注线程安全保证，最小化锁范围，持锁时禁止 I/O 或 GPU 操作。优先用 queue 而非共享可变状态。
- **资源管理**: 文件句柄、socket、CUDA stream 必须用 `with`。cache/buffer 必须有大小上限或淘汰策略。

### P1 — Performance

- **禁止在 model inference 热路径中频繁调用** `.item()`, `.cpu()`, `.tolist()`。
- **数据处理尽量在 GPU 上向量化**。
- **热路径使用低开销实现**。

### P2 — Maintainability

- **DRY**: 超过 5 行的重复代码必须提取为共享函数。
- **文件不超过 2000 行**，超出需拆分（Mixin 或子模块）。
- **函数不超过 50 行**（不含 docstring），逻辑子步骤提取为 private helper。
- **命名清晰**: public API 禁止缩写（`request_count` 而非 `req_cnt`），允许 `num`, `idx`, `cfg`, `bs` 等通用缩写。
- **布尔变量前缀**: `is_`, `has_`, `should_`, `can_`。
- **对称命名**: `start/stop`, `begin/end`, `open/close`, `send/recv`，禁止混用（如 `start/finish`）。
- **禁止泛型命名**: 不用 `data`, `result`, `info`, `tmp`, `manager`, `handler`，用 `token_ids`, `decode_result` 等具体名称。
- **Import 顺序**: stdlib → third-party → local，用空行分隔。禁止 `from module import *`。重依赖用 lazy import。
- **常量**: 禁止魔法数字，提取为命名常量。

### P3 — Style

- **函数纯度**: 优先纯函数，避免原地修改输入参数。
- **精简构造函数**: `__init__` 参数精简，不传庞大 config 对象。
- **避免动态属性** (`getattr`/`setattr`)，代码应显式。
- **完整分支**: 如果 `if` 赋值或返回值，必须包含 `else`。Guard clause 不需要 `else`。分支超过 3 层需重构为 `if/elif/.../else` 或 dispatch dict。
- **类型注解**: 所有 public API 和函数签名必须有 type hints。
- **私有前缀**: 类/文件内部函数用 `_private` 前缀。
- **清理**: 删除调试注释和日志，删除中文注释。

### P4 — Process

- **测试**: PR 描述中提供可复制粘贴的验证脚本。重要功能添加 CI 单元测试。
- **测试合约**: 测试输入→输出，不测内部状态。测试中固定随机种子。每个 test 函数一个断言概念。

### Self-Verification Checklist（每次提交代码前必须过一遍）

```
- [ ] No broad try/except blocks
- [ ] Validation only at boundaries, not deep inside helpers
- [ ] Early returns used where appropriate
- [ ] No .item() / .cpu() / .tolist() in hot paths
- [ ] No duplicate code blocks > 5 lines
- [ ] All functions < 50 lines
- [ ] All public APIs have type hints
- [ ] Boolean variables prefixed with is_/has_/should_/can_
- [ ] No magic numbers
- [ ] No wildcard imports
- [ ] No generic names (data, result, tmp)
- [ ] Context managers for all resources
- [ ] Complete branching (else clauses where required)
- [ ] No debug comments or logs left in
- [ ] No Chinese comments in code
```

**违反以上任何规则的代码不予接受。**

---

## 1. Task Background

### What is SGLang Omni?

SGLang Omni is a multi-stage pipeline system for omni models (Qwen3-Omni-7B) that support multi-modal input (text, audio, video, image) and multi-modal output (text + audio). The architecture disaggregates the model into independent stages, each with its own scheduler and execution engine, connected via relay (shared memory / NCCL / NixL / Mooncake).

### Omni Model Architecture (Qwen3-Omni)

The model has the following stages:

```
Input (text/audio/video/image)
  -> Preprocessing (tokenization)
  -> Image Encoder (ViT)
  -> Audio Encoder (Whisper)
  -> MM Aggregate (merge embeddings)
  -> Thinker (MoE AR text generation, 32 layers)
  -> Talker (MoE AR codec generation, 20 layers + CodePredictor)
  -> Code2Wav (vocoder, codec codes -> audio waveform)
  -> Decode (text output)
```

### PR Goal

End-to-end pipeline using SGLang model runner for both thinker and talker with:
- Chunked prefill and batching + prefix cache
- Relay connecting thinker output to talker input
- Concurrent text output (from thinker) and audio output (from talker)
- Support for video, audio, and text inputs

---

## 1.5. Known Bugs & Pitfalls in Existing Code (MUST FIX)

### Bug 1: `forward_batch=None` in code_predictor_forward (P0-BLOCKER)

**File**: `sglang_omni/models/qwen3_omni/talker.py:638`

```python
# CURRENT CODE (BROKEN):
predictor_hidden = self.code_predictor(
    inputs_embeds=current_input,
    positions=torch.arange(current_input.shape[1], device=current_input.device),
    forward_batch=None,  # <-- BUG: RadixAttention REQUIRES ForwardBatch
)
```

`Qwen3OmniMoeTalkerCodePredictor` 的 decoder layer 使用了 `Qwen3OmniMoeThinkerTextAttention`，后者内部用 `RadixAttention`。`RadixAttention` 需要 `ForwardBatch` 来管理 KV cache。传 `None` 会崩溃。

**Fix options**:
- **Option A (推荐)**: Code predictor 的序列非常短（2-17 tokens），不需要 RadixAttention。把 code predictor 的 attention 层替换为标准 `torch.nn.MultiheadAttention` 或手写的 causal attention。
- **Option B**: 为 code predictor 构造一个专用的 mini ForwardBatch，但这引入了不必要的复杂度。

### Bug 2: `top_k_top_p_sampling_from_probs` 接口可能不匹配

**File**: `sglang_omni/models/qwen3_omni/talker.py:646-648`

```python
code = top_k_top_p_sampling_from_probs(
    probs, top_k=50, top_p=0.8
)  # expects [batch, vocab] -> [batch, 1]
```

需要验证 SGLang 的 `top_k_top_p_sampling_from_probs` 的签名是否匹配。如果不匹配，用 `torch.multinomial` 做 fallback。

### Pitfall 1: SGLang model runner 不暴露中间层 hidden states

**File**: `sglang_omni/engines/omni/runtime/sglang_ar.py:410`

当前 `SGLangModelRunner.execute()` 调用 `model_worker.forward_batch_generation(forward_batch)`，只返回 `GenerationBatchResult(logits_output=..., can_run_cuda_graph=...)`。**没有返回中间层 hidden states**。

但 talker 需要 thinker 的 layer 0 (embedding) 和 layer 24 (`accept_hidden_layer`) 的输出。这是最复杂的改造点之一。详见 Task 9。

### Pitfall 2: Thinker SGLang 模型的 forward 可能不直接输出分层 hidden states

SGLang 的 `Qwen3OmniMoeThinkerTextModel.forward()` 在 `thinker.py` 中只返回最终的 `hidden_states`，不像 HF 的 `output_hidden_states=True` 那样返回所有层。需要在 thinker model 中添加 hook 或修改 forward 来捕获 layer 0 和 layer 24。

---

## 1.6. Thinker → Talker Exact Data Contract

> 来源：HF 参考实现 `transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py:3955-3985`

### Thinker 必须输出的 3 个 Tensor

| Tensor | Shape | Source | Description |
|--------|-------|--------|-------------|
| `thinker_embed` | `[batch, seq_len, 2048]` | thinker layer 0 output（embedding 层） | Text token 的原始表示 |
| `thinker_hidden` | `[batch, seq_len, 2048]` | thinker layer 24 output（`accept_hidden_layer`） | 融合了 multimodal 信息的深层表示 |
| `is_multimodal_mask` | `[batch, seq_len]` bool | `input_ids == audio/image/video_token_id` | 标记哪些位置是多模态 token |

### Config 关键参数

| Parameter | Value | Location |
|-----------|-------|----------|
| `accept_hidden_layer` | 24 | `Qwen3OmniMoeTalkerConfig` |
| `thinker_hidden_size` | 2048 | `Qwen3OmniMoeTalkerConfig` |
| `talker.text_config.hidden_size` | 1024 | `Qwen3OmniMoeTalkerTextConfig` |
| `talker.text_config.intermediate_size` | 2048 | `Qwen3OmniMoeTalkerTextConfig` |
| `num_code_groups` | 16 | `Qwen3OmniMoeTalkerConfig` |
| `audio_token_id` / `image_token_id` / `video_token_id` | see config | `Qwen3OmniMoeThinkerConfig` |

### Talker 内部处理流程

```
Step 1: prepare_input_embeds()
  text_positions:        text_projection(thinker_embed)       -> [batch, *, 1024]
  multimodal_positions:  hidden_projection(thinker_hidden)    -> [batch, *, 1024]
  merged by is_multimodal_mask -> talker_input_embeds          [batch, seq_len, 1024]

Step 2: talker.forward(inputs_embeds=talker_input_embeds)
  -> talker_hidden_states                                      [batch, seq_len, 1024]

Step 3: talker.compute_logits(talker_hidden_states)
  -> layer0_logits                                             [batch, seq_len, codec_vocab]
  -> layer0_codes = argmax/sample(layer0_logits)               [batch, seq_len]

Step 4: talker.code_predictor_forward(layer0_codes, talker_hidden_states)
  -> result_codes                                              [batch, 16, seq_len]
  -> summed_embeddings                                         [batch, seq_len, 1024]

Step 5 (future): Code2Wav(result_codes) -> audio waveform
```

### Relay 传输的 Payload

Thinker 完成后，通过 relay 传给 talker 的数据：

```python
{
    "thinker_embed": Tensor[batch, seq_len, 2048],        # GPU tensor
    "thinker_hidden": Tensor[batch, seq_len, 2048],       # GPU tensor
    "is_multimodal_mask": Tensor[batch, seq_len],          # bool, GPU
    "input_ids": Tensor[batch, seq_len],                   # int64, for reference
    "output_ids": list[int],                               # thinker generated token IDs
}
```

---

## 1.7. Template-to-Follow References (每个 Task 的模板)

每个 Task 有一个明确的"照着抄然后改"的模板文件：

| Task | 模板文件 | 照抄范围 | 需要改什么 |
|------|---------|---------|-----------|
| Task 1 (routing) | `pipeline/next_stage.py` | `thinker_next()` 的结构 | 加 `TALKER_STAGE`，改 `thinker_next` 返回值 |
| Task 2 (PipelineState) | `io.py` 中 `ThinkerOutput` 和 `thinker_out` 字段 | 整个 pattern | 加 `TalkerOutput` TypedDict + `talker_inputs`/`talker_out` 字段 |
| Task 3 (engine_io) | `engine_io.py` 中 `build_sglang_thinker_request()` | 整个函数结构 | 改为从 `state.thinker_out` 提取 hidden states 给 talker |
| Task 4 (talker engine) | `runtime/sglang_ar.py` 中 `SGLangModelRunner` | 类结构和 `execute()` 方法 | 替换 forward 逻辑为 talker 多步 codec 生成 |
| Task 5 (engine factory) | `factory.py` 中 `create_sglang_ar_engine()` | 整个工厂函数 | 换成 TalkerModelWorker + TalkerModelRunner |
| Task 6 (executor) | `stages.py` 中 `create_sglang_thinker_executor()` | 整个函数 | 换 request/result/stream builder 为 talker 版本 |
| Task 7 (config) | `config.py` 中 THINKER_STAGE 的 StageConfig | 一行配置 | 改名、改 factory path、改 get_next |
| Task 8 (merge) | `merge.py` 中 `merge_for_thinker()` | 合并逻辑结构 | 改为从 thinker output 提取 hidden states |
| Task 9 (hidden states) | `runtime/ar.py` 中 `_capture_prefill_outputs()` | capture 机制 | 在 SGLang forward 路径中截取 layer 0 和 layer 24 |

---

## 1.8. Difficulty Assessment & Fallback Strategies

| Task | 难度 | 风险 | Fallback |
|------|------|------|----------|
| Task 1 (routing) | 低 | 无 | — |
| Task 2 (PipelineState) | 低 | 无 | — |
| Task 3 (engine_io) | 中 | tensor shape 不匹配 | 加 shape assert，打印 debug info |
| Task 4 (talker engine) | **高** | code_predictor attention 崩溃 | 先用标准 attention 替代 RadixAttention（见 Bug 1） |
| Task 5 (engine factory) | 中 | ModelWorker 加载失败 | 先不走 SGLang ModelWorker，直接 `torch.load` talker 权重 |
| Task 6 (executor) | 中 | — | — |
| Task 7 (config) | 低 | — | — |
| Task 8 (merge) | 中 | — | — |
| Task 9 (hidden states) | **高** | SGLang forward 不暴露中间层 | **Fallback: 在 thinker model 的 decoder layer 上注册 `register_forward_hook` 捕获 layer 0 和 layer 24 的输出** |
| Task 10 (Code2Wav) | 高，但 MVP 可跳过 | — | MVP 只输出 codec codes，不转音频 |
| Task 11 (streaming) | 中 | — | — |
| Task 12 (E2E test) | 低 | 依赖前面所有 Task | 先用 mock model 跑通流程，再换真模型 |

### Task 9 的三种实现方案（最关键的硬骨头）

**方案 A — `register_forward_hook`（推荐，最小改动）**:
```python
# 在 thinker model 上注册 hook 捕获中间层
captured_hidden = {}

def hook_fn(layer_idx):
    def _hook(module, input, output):
        captured_hidden[layer_idx] = output[0].detach()
    return _hook

# 注册 layer 0 和 layer 24
thinker.model.layers[0].register_forward_hook(hook_fn(0))
thinker.model.layers[24].register_forward_hook(hook_fn(24))
```
优点：不改 SGLang 内部代码。缺点：hook 在每步都触发，需要手动 accumulate。

**方案 B — 修改 thinker forward 返回值**:
直接改 `Qwen3OmniMoeThinkerTextModel.forward()` 让它返回 `(hidden_states, layer_outputs_dict)`。
优点：干净。缺点：改动 SGLang vendor 代码，升级困难。

**方案 C — 在 SGLangModelRunner 中截取**:
在 `SGLangModelRunner.execute()` 调 `model_runner.forward()` 前后，从 model 上读取缓存的中间层。
优点：改动集中在 runtime 层。缺点：依赖 model 内部状态。

---

## 2. Current Status (What's Done)

### Fully Implemented

| Component | Location | Status |
|-----------|----------|--------|
| **Thinker SGLang model** | `sglang_omni/models/qwen3_omni/thinker.py` | Done - Uses RadixAttention, ForwardBatch, FusedMoE |
| **Talker SGLang model** | `sglang_omni/models/qwen3_omni/talker.py` | Done - Shared Expert MoE, CodePredictor, weight loading |
| **Talker configs** | `sglang_omni/config/qwen3_omni.py` | Done - TalkerTextConfig, TalkerCodePredictorConfig, TalkerConfig |
| **Thinker pipeline executor** | `sglang_omni/models/qwen3_omni/pipeline/stages.py:create_sglang_thinker_executor_from_config()` | Done |
| **Thinker pipeline stage** | `sglang_omni/models/qwen3_omni/config.py` (StageConfig for THINKER_STAGE) | Done |
| **SGLang AR engine** | `sglang_omni/engines/omni/factory.py:create_sglang_ar_engine()` | Done - ModelWorker, PrefillManager, DecodeManager |
| **SGLang batch planner** | `sglang_omni/engines/omni/runtime/sglang_ar.py:SGLangBatchPlanner` | Done - Chunked prefill + decode scheduling |
| **SGLang model runner** | `sglang_omni/engines/omni/runtime/sglang_ar.py:SGLangModelRunner` | Done |
| **SGLang iteration controller** | `sglang_omni/engines/omni/runtime/sglang_ar.py:SGLangIterationController` | Done |
| **Tree cache / RadixCache** | `sglang_omni/engines/ar/sglang_backend/scheduler/cache.py` | Done |
| **Prefill Manager** | `sglang_omni/engines/ar/sglang_backend/scheduler/prefill.py` | Done |
| **Decode Manager** | `sglang_omni/engines/ar/sglang_backend/scheduler/decode.py` | Done |
| **Generic scheduler** | `sglang_omni/engines/omni/scheduler.py` | Done - Request lifecycle, batch planning delegation |
| **Relay system** | `sglang_omni/relay/` | Done - SHM, NCCL, NixL, Mooncake backends |
| **Data plane adapter** | `sglang_omni/pipeline/worker/data_plane.py` | Done - Tensor extraction & relay transfer |
| **Pipeline coordinator** | `sglang_omni/pipeline/coordinator.py` | Done - Stage routing, request tracking |
| **Preprocessing / Encoder stages** | `sglang_omni/models/qwen3_omni/pipeline/stages.py` | Done |
| **Pipeline state** | `sglang_omni/models/qwen3_omni/io.py:PipelineState` | Done (thinker fields only) |
| **OpenAI-compatible server** | `sglang_omni/serve/` | Done |

### Not Done (Talker Integration)

| Component | Status | Notes |
|-----------|--------|-------|
| **TALKER_STAGE constant & routing** | Not started | `next_stage.py` has no TALKER_STAGE |
| **Talker pipeline executor** | Not started | No `create_sglang_talker_executor` |
| **Talker StageConfig in pipeline** | Not started | `config.py` has no talker stage entry |
| **Thinker->Talker relay connection** | Not started | `thinker_next()` returns DECODE, not TALKER |
| **Talker engine factory** | Not started | No `create_sglang_talker_engine()` in factory.py |
| **Talker engine_io helpers** | Not started | No `build_talker_request()`, `apply_talker_result()` |
| **PipelineState talker fields** | Not started | No `talker_inputs`, `talker_out` fields |
| **Talker batch result processing** | Not started | Talker generates multi-layer codec codes, not text tokens |
| **Talker-specific IterationController** | Not started | Talker finish condition differs from thinker |
| **Code2Wav / vocoder stage** | Not started | Codec codes -> audio waveform conversion |
| **Audio output streaming** | Not started | No audio_chunk/audio_final event emission |
| **End-to-end test** | Not started | No test for thinker+talker pipeline |
| **Merge function for talker inputs** | Not started | Need `merge_for_talker` to pack thinker outputs for talker |

---

## 3. Remaining Work (Detailed Tasks)

### Task 1: Add TALKER_STAGE to Pipeline Routing

**File**: `sglang_omni/models/qwen3_omni/pipeline/next_stage.py`

**What to do**:
1. Add `TALKER_STAGE = "talker_ar"` constant (name follows SGLang convention)
2. Add `CODE2WAV_STAGE = "code2wav"` constant (for future vocoder)
3. Modify `thinker_next()` to route to TALKER_STAGE (when audio output is needed) or DECODE_STAGE (text-only)
4. Add `talker_next()` function that routes to DECODE_STAGE (or CODE2WAV_STAGE)

**Routing logic**:
```
thinker_next(request_id, output) -> [TALKER_STAGE, DECODE_STAGE]
  # Both in parallel: talker for audio, decode for text
  # OR just TALKER_STAGE, then talker routes to decode

talker_next(request_id, output) -> DECODE_STAGE
  # After talker finishes, go to decode for final output assembly
```

**Key decision**: Should thinker fan out to both talker AND decode simultaneously (parallel text+audio), or should it be sequential (thinker -> talker -> decode)?

**Recommendation**: Use parallel fan-out. Thinker text output can stream to decode immediately while talker processes audio in parallel. The coordinator already supports multi-destination routing (see `preprocessing_next` which returns a list).

```python
def thinker_next(request_id: str, output: Any) -> list[str]:
    # Check if audio output is requested (via request params or state)
    # If so: route to both TALKER_STAGE and DECODE_STAGE
    # If text-only: route to DECODE_STAGE only
    return [TALKER_STAGE, DECODE_STAGE]  # or just [DECODE_STAGE]
```

### Task 2: Extend PipelineState for Talker

**File**: `sglang_omni/models/qwen3_omni/io.py`

**What to do**:
1. Add `TalkerOutput` TypedDict:
   ```python
   class TalkerOutput(TypedDict, total=False):
       codec_codes: list[list[int]]  # [num_code_groups, seq_len]
       audio_hidden_states: Any      # Hidden states for Code2Wav
       step: int
       is_final: bool
   ```
2. Add fields to `PipelineState`:
   ```python
   talker_inputs: dict[str, Any] = field(default_factory=dict)
   talker_out: TalkerOutput | None = None
   ```
3. Update `from_dict()` and `to_dict()` to include the new fields
4. Add `OmniEventType` entries if needed (e.g., `"codec_chunk"`)

### Task 3: Create Talker Engine I/O Helpers

**File**: `sglang_omni/models/qwen3_omni/pipeline/engine_io.py`

**What to do**:
1. Add `build_talker_request()`:
   - Takes PipelineState (containing thinker_out with hidden states)
   - Extracts thinker hidden states and embeddings
   - Calls `talker.prepare_input_embeds()` to project to talker space
   - Returns request data suitable for talker's SGLang engine

2. Add `build_sglang_talker_request()`:
   - Like `build_sglang_thinker_request()` but for talker
   - Creates SGLang Req with talker-specific SamplingParams
   - Wraps in SGLangARRequestData

3. Add `apply_talker_result()`:
   - Takes talker output (codec codes, hidden states)
   - Stores in state.talker_out
   - Stores in state.engine_outputs[TALKER_STAGE]

**Critical**: The thinker's SGLang model runner currently only outputs `next_token_ids` (one token per step). For the talker integration, the thinker must also output its hidden states (the intermediate representations needed by `talker.prepare_input_embeds`). This requires:
- Capturing thinker's `hidden_states` during forward pass (see `capture_model_output_keys` in existing ARRequestData)
- Relaying these tensors to the talker stage

### Task 4: Create Talker-Specific SGLang Engine Components

**Overview**: The talker's forward pass is fundamentally different from the thinker's standard AR generation. The talker:
1. Takes projected thinker outputs as input embeddings (not token IDs)
2. Runs through MoE backbone to generate hidden states
3. Computes layer-0 codec logits via `codec_head`
4. Runs `code_predictor_forward()` autoregressively for layers 1-15

This multi-step process needs a custom model runner and iteration controller.

#### 4a. Talker Model Runner

**File**: New file `sglang_omni/engines/omni/runtime/sglang_talker.py`

**What to do**:
1. Create `TalkerModelRunner` that wraps the talker model execution:
   ```python
   class TalkerModelRunner:
       def execute(self, scheduler_output):
           # 1. Get ForwardBatch from ScheduleBatch
           # 2. Prepare input embeddings (project thinker outputs)
           # 3. Run talker backbone forward
           # 4. Compute layer-0 codec logits + sample
           # 5. Run code_predictor_forward for layers 1-15
           # 6. Return codec codes per request
   ```

2. Key difference from SGLangModelRunner: talker must handle the multi-step codec generation within a single `execute()` call, or alternatively, treat each codec step as a separate decoding step.

**Design choice**:
- **Option A**: Treat talker as a standard AR model where each step generates one codec token for layer-0, then batch the code_predictor_forward separately. This fits the existing scheduler loop but requires careful state management.
- **Option B (Recommended)**: Wrap the entire talker forward pass (backbone + codec_head + code_predictor) as a single "step" in the model runner. Each step produces a complete set of codec codes for one audio frame position.

#### 4b. Talker Iteration Controller

**File**: Same as 4a or in `sglang_omni/engines/omni/runtime/sglang_talker.py`

**What to do**:
1. Create `TalkerIterationController` that determines when talker generation is finished:
   - Talker generates codec codes for each position corresponding to thinker's output tokens
   - Finished when all positions have been processed (or a special end-of-audio token is generated)

2. The `update_request()` method must:
   - Accumulate codec codes across steps
   - Track which positions have been processed
   - Update the request's output data

#### 4c. Talker Output Processor

1. Create `TalkerOutputProcessor` that converts the talker's multi-layer codec codes into per-request outputs suitable for streaming.

### Task 5: Create Talker Engine Factory

**File**: `sglang_omni/engines/omni/factory.py`

**What to do**:
1. Add `create_sglang_talker_engine()`:
   ```python
   def create_sglang_talker_engine(
       server_args: Any,
       talker_model: Qwen3OmniTalker,  # Pre-loaded talker model
       gpu_id: int = 0,
   ) -> OmniEngine:
       # Similar structure to create_sglang_ar_engine but with:
       # - TalkerModelRunner instead of SGLangModelRunner
       # - TalkerIterationController instead of SGLangIterationController
       # - Shared PrefillManager/DecodeManager for KV cache
   ```

2. **Key issue**: The current `ModelWorker` loads the model via `SGLModelRunner` which uses `ModelConfig.from_server_args()`. This assumes a standard HF model architecture. The talker needs a custom model loading path since it has a different structure (codec_embedding, shared expert MoE, code_predictor).

**Options**:
- **Option A**: Register the talker model class with SGLang's model registry so `ModelConfig` can find it. This requires adding model registration code.
- **Option B (Recommended)**: Create a `TalkerModelWorker` that directly loads the `Qwen3OmniTalker` model and provides the same interface as `ModelWorker` (forward_batch_generation, get_memory_pool). This avoids modifying SGLang's model registry.

### Task 6: Create Talker Pipeline Executor

**File**: `sglang_omni/models/qwen3_omni/pipeline/stages.py`

**What to do**:
1. Add `create_sglang_talker_executor()`:
   ```python
   def create_sglang_talker_executor(
       server_args: Any,
       model_path: str,
       *,
       gpu_id: int = 0,
   ) -> EngineExecutor:
       # 1. Load Qwen3OmniTalker model
       # 2. Create talker engine via create_sglang_talker_engine()
       # 3. Define _request_builder: PipelineState -> TalkerRequestData
       # 4. Define _result_builder: TalkerOutput -> PipelineState update
       # 5. Define _stream_builder: per-step codec output -> audio events
       # 6. Return EngineExecutor(engine, request_builder, result_builder, stream_builder)
   ```

2. Add `create_sglang_talker_executor_from_config()`:
   ```python
   def create_sglang_talker_executor_from_config(
       model_path: str,
       *,
       gpu_id: int = 0,
       talker_max_seq_len: int = 4096,
       server_args_overrides: dict | None = None,
   ) -> EngineExecutor:
       # Build ServerArgs for talker
       # Call create_sglang_talker_executor()
   ```

### Task 7: Add Talker StageConfig to Pipeline

**File**: `sglang_omni/models/qwen3_omni/config.py`

**What to do**:
1. Import `TALKER_STAGE` from `next_stage.py`
2. Add talker StageConfig between THINKER_STAGE and DECODE_STAGE:
   ```python
   StageConfig(
       name=TALKER_STAGE,
       executor=ExecutorConfig(
           factory="sglang_omni.models.qwen3_omni.pipeline.stages.create_sglang_talker_executor_from_config",
           args={
               "talker_max_seq_len": 4096,
           },
       ),
       get_next="sglang_omni.models.qwen3_omni.pipeline.next_stage.talker_next",
       relay=RelayConfig(device="cuda"),
   )
   ```

### Task 8: Create Merge Function for Talker Inputs

**File**: `sglang_omni/models/qwen3_omni/pipeline/merge.py`

**What to do**:
1. Add `merge_for_talker()` (if talker receives aggregated inputs from multiple sources)
2. Or, if talker only receives from thinker, the merge is simpler - just extract thinker's output hidden states

**Key data flow**:
```
Thinker output:
  - output_ids: [int]           # Generated text token IDs
  - hidden_states: Tensor       # Last layer hidden states (needed by talker)
  - embeddings: Tensor          # Input embeddings (needed by talker for text_projection)

Talker input:
  - thinker_hidden_states: Tensor  # For hidden_projection
  - thinker_embeds: Tensor         # For text_projection
  - is_multimodal_mask: Tensor     # Which positions are multimodal vs text
  - codec_input_ids: Tensor        # Initial codec tokens (BOS or continuation)
```

### Task 9: Handle Thinker Hidden State Capture

**File**: Multiple files

**What to do**:
The thinker's SGLang model runner currently only outputs `next_token_ids`. For talker input, we need the thinker's hidden states.

1. **In thinker model** (`thinker.py`): The model already outputs hidden states (it's a standard transformer). The SGLang ForwardBatch captures the final hidden states before the lm_head.

2. **In SGLangModelRunner** (`runtime/sglang_ar.py`): Modify `execute()` to capture and store hidden states when `capture_model_output_keys` includes "hidden_states". Currently the GenerationBatchResult only has `logits_output`. Need to also store the pre-logit hidden states.

3. **In SGLangIterationController**: When thinker finishes, the hidden states should be accumulated and stored on the request data for relay to talker.

**This is one of the most critical and complex pieces** because:
- SGLang's paged KV cache stores attention KV but not the hidden states
- We need to store per-token hidden states for ALL generated tokens (not just the last one)
- These need to be relayed to the talker via the data plane

**Approach**:
- Add a `hidden_states_buffer` to `SGLangARRequestData` that accumulates hidden states per step
- In `SGLangModelRunner.execute()`, after forward pass, store the last-layer hidden state
- When thinker finishes, the full hidden_states tensor is available on the request data
- The relay (DataPlaneAdapter) then transfers this tensor to the talker stage

### Task 10: Code2Wav / Audio Vocoder Stage (Optional for MVP)

**Note**: For the initial MVP, the talker output (codec codes) can be converted to audio using a simple offline vocoder. The Code2Wav stage can be added later as a separate pipeline stage.

**For MVP**: Output codec codes directly and convert to audio in the decode stage or post-processing.

**For full implementation**:
1. Add `CODE2WAV_STAGE` to pipeline
2. Implement vocoder wrapper (DiT-based or SoundStream decoder)
3. Add streaming audio output

### Task 11: Audio Output Streaming and Decode

**File**: `sglang_omni/models/qwen3_omni/pipeline/merge.py` and `stages.py`

**What to do**:
1. Update `decode_events()` to handle talker output:
   - Emit `audio_chunk` events with codec codes or PCM audio data
   - Emit `audio_final` event when talker finishes

2. Update `create_decode_executor()` to process both thinker and talker outputs:
   - If `state.talker_out` exists, include audio in the final response
   - If only `state.thinker_out`, text-only response (current behavior)

3. Update the stream builder for talker to emit audio streaming events

### Task 12: End-to-End Integration Test

**File**: New file `tests/test_thinker_talker_e2e.py`

**What to do**:
1. Test thinker -> talker pipeline with mock models:
   ```python
   async def test_thinker_talker_e2e():
       # 1. Create pipeline with all stages including talker
       # 2. Submit a request with text input
       # 3. Verify thinker produces text output
       # 4. Verify talker produces codec codes
       # 5. Verify decode stage assembles both
   ```

2. Test with real Qwen3-Omni model (GPU required):
   ```python
   async def test_qwen3_omni_full_pipeline():
       # 1. Load model weights
       # 2. Create pipeline
       # 3. Submit multimodal request
       # 4. Verify text + audio output
   ```

---

## 4. Task Dependency Graph

```
Task 1 (routing)     ─────────────────────────────────────┐
Task 2 (PipelineState)────────────────────────────────────┤
Task 3 (engine_io)   ──┬──────────────────────────────────┤
Task 9 (hidden states)─┘                                  │
Task 4 (talker engine components) ────┐                   │
Task 5 (talker engine factory)   ─────┤                   │
Task 6 (talker executor)         ─────┼───> Task 7 (config) ──> Task 11 (streaming) ──> Task 12 (test)
Task 8 (merge function)         ──────┘                   │
Task 10 (Code2Wav)              ──────────────────────────┘
```

**Suggested execution order** (critical path):

1. **Phase 1 - Data Flow** (Tasks 1, 2, 3, 8): Define the data structures and routing
2. **Phase 2 - Hidden State Capture** (Task 9): Enable thinker to output hidden states
3. **Phase 3 - Talker Engine** (Tasks 4, 5): Build talker-specific engine components
4. **Phase 4 - Pipeline Wiring** (Tasks 6, 7): Wire talker into the pipeline
5. **Phase 5 - Output** (Tasks 10, 11): Audio output and streaming
6. **Phase 6 - Validation** (Task 12): End-to-end testing

---

## 5. Implementation Details & Key Decisions

### 5.1 Talker Model Loading

The talker model (`Qwen3OmniTalker`) has `load_weights()` that expects HF checkpoint format with `talker.` prefix. Two approaches:

**Option A - Register with SGLang** (not recommended for MVP):
- Add `Qwen3OmniTalker` to SGLang's model registry
- Requires upstream changes or monkey-patching

**Option B - Custom TalkerModelWorker** (recommended):
```python
class TalkerModelWorker:
    def __init__(self, model_path, server_args, gpu_id):
        # 1. Load Qwen3OmniTalker directly
        config = load_talker_config(model_path)
        self.model = Qwen3OmniTalker(config)
        self.model.load_weights(load_checkpoint_weights(model_path))
        self.model = self.model.to(f"cuda:{gpu_id}")

        # 2. Initialize KV cache pools (reuse SGLang's allocator)
        # 3. Provide same interface as ModelWorker
```

### 5.2 Talker Scheduler Design

The talker's generation pattern differs from standard AR:
- **Thinker**: One token per step, runs until EOS
- **Talker**: For each thinker output position, generates 16 codec codes (1 from backbone + 15 from code_predictor)

**Recommended approach**: Treat the talker as a "position-level AR model":
- Each scheduler step processes one position
- Within that step, the model runner handles the full 16-layer codec generation
- The scheduler tracks positions completed vs total positions

### 5.3 Thinker Hidden State Relay

The relay must transfer thinker's hidden states to talker. Key considerations:
- Hidden states are large tensors: `[seq_len, hidden_size]` where hidden_size=3584 for Qwen3-Omni
- For streaming: relay hidden states incrementally as thinker generates tokens
- For batch: accumulate all hidden states, relay once when thinker finishes

**Recommended**: Batch relay after thinker finishes (simpler, sufficient for MVP). Stream relay can be added later for latency optimization.

### 5.4 Naming Conventions

Follow SGLang main repo naming:
- Stage names: `"thinker"`, `"talker"` (already in use)
- Constants: `THINKER_STAGE`, `TALKER_STAGE`
- Engine: `create_sglang_talker_engine()`
- Executor: `create_sglang_talker_executor_from_config()`
- Model runner: `TalkerModelRunner`
- Request data: `TalkerARRequestData(SGLangARRequestData)`

---

## 6. Verification Checklist

### 核心验收标准 / Core Acceptance Criteria

**唯一的验收标准：能 end2end 跑通 thinker 出字 + talker 出音频。**

如果 talker 接入了 SGLang model runner，那就必须能跑起来一个完整的 thinker + talker 的例子。不能跑起来就是没完成。

#### E2E Smoke Test Script（最终验证用）

实现完成后，应该能运行类似以下脚本，并得到文字 + codec codes 输出：

```python
# tests/test_thinker_talker_e2e.py
"""
End-to-end smoke test: thinker generates text, talker generates codec codes.
This is THE acceptance test — if this doesn't pass, the integration is not done.
"""
import asyncio
import torch

async def test_thinker_talker_e2e():
    """
    1. Build full pipeline: preprocessing -> encoders -> aggregate -> thinker -> talker -> decode
    2. Submit a text-only request: "Hello, how are you?"
    3. Assert thinker produces text tokens (output_ids non-empty)
    4. Assert thinker hidden states are relayed to talker
    5. Assert talker produces codec codes (shape [num_code_groups, seq_len])
    6. Assert decode stage returns both text and audio data
    """
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig

    MODEL_PATH = "/path/to/Qwen3-Omni-7B"  # or env var

    # Build the pipeline (contains thinker + talker stages)
    config = Qwen3OmniPipelineConfig(model_path=MODEL_PATH)

    # Verify talker stage exists in config
    stage_names = [s.name for s in config.stages]
    assert "talker" in stage_names or "talker_ar" in stage_names, \
        f"Talker stage missing from pipeline config! Got: {stage_names}"

    # ... (launch pipeline, submit request, collect results)

    # Key assertions:
    # 1. Text output exists and is non-empty
    assert result.get("text"), "Thinker must produce text output"

    # 2. Codec codes exist (talker output)
    codec_codes = result.get("codec_codes")
    assert codec_codes is not None, "Talker must produce codec codes"
    # codec_codes shape: [num_code_groups=16, generated_audio_frames]
    assert len(codec_codes) == 16, f"Expected 16 code groups, got {len(codec_codes)}"

    print(f"[PASS] Text: {result['text'][:100]}...")
    print(f"[PASS] Codec codes shape: {[len(c) for c in codec_codes]}")
```

**如何运行**：
```bash
# 需要 GPU + 模型权重
python -m pytest tests/test_thinker_talker_e2e.py -v -s

# 或者用 server 模式验证：
python -m sglang_omni.serve --model-path /path/to/Qwen3-Omni-7B
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "audio": {"voice": "default", "format": "wav"}}'
# Response 应该包含 text + audio 两个字段
```

#### 分步检查点 / Stepwise Checkpoints

在最终 E2E 之前，每完成一个 Task 应该能通过对应的检查点：

| 完成 Task | 检查方式 | 预期结果 |
|-----------|---------|---------|
| Task 1 (routing) | `from sglang_omni.models.qwen3_omni.pipeline.next_stage import TALKER_STAGE` | 不报 ImportError |
| Task 2 (PipelineState) | `state = PipelineState(); state.talker_out = {...}; state.to_dict()` | dict 包含 talker_out |
| Task 3 (engine_io) | `build_talker_request(state)` 给一个有 thinker_out 的 state | 返回合法的 request data |
| Task 4+5 (talker engine) | 单独启动 talker engine，喂入 mock input embeddings | 产出 codec codes tensor |
| Task 6+7 (pipeline wiring) | `Qwen3OmniPipelineConfig` 的 stages 列表包含 talker | stage names 包含 talker |
| Task 9 (hidden states) | 运行 thinker，检查 request.data 上有 hidden_states_buffer | tensor shape 正确 |
| **Task 12 (E2E)** | **运行上面的 smoke test** | **文字 + codec codes 都有输出** |

### 补充测试

#### Unit Tests
- [ ] `TalkerModelRunner.execute()` produces correct codec codes shape `[batch, 16, seq_len]`
- [ ] `TalkerIterationController` correctly tracks position completion
- [ ] `build_talker_request()` correctly projects thinker outputs
- [ ] `apply_talker_result()` correctly stores codec codes in PipelineState
- [ ] `thinker_next()` correctly routes to TALKER_STAGE

#### Integration Tests
- [ ] Relay transfers thinker hidden states tensor without corruption
- [ ] Pipeline correctly routes: preprocessing -> encoders -> aggregate -> thinker -> talker -> decode
- [ ] Scheduler handles talker batch results (multi-layer codec codes)
- [ ] Stream events include both `text_delta` (from thinker) and `audio_chunk` (from talker)

#### Stress / Batch Tests
- [ ] Batch of 2+ concurrent requests: verify batching works correctly
- [ ] Chunked prefill: verify long prompts are chunked correctly for talker
- [ ] Thinker text output starts streaming immediately (not blocked by talker)
- [ ] Memory usage reasonable: thinker hidden states don't OOM
- [ ] KV cache properly released after request completion

---

## 7. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `pipeline/next_stage.py` | Modify | Add TALKER_STAGE, talker_next(), modify thinker_next() |
| `io.py` | Modify | Add TalkerOutput, talker_inputs/talker_out fields |
| `pipeline/engine_io.py` | Modify | Add build_talker_request(), apply_talker_result() |
| `pipeline/merge.py` | Modify | Add merge_for_talker() or talker input assembly |
| `pipeline/stages.py` | Modify | Add create_sglang_talker_executor[_from_config]() |
| `config.py` | Modify | Add TALKER_STAGE StageConfig |
| `engines/omni/factory.py` | Modify | Add create_sglang_talker_engine() |
| `engines/omni/runtime/sglang_talker.py` | **New** | TalkerModelRunner, TalkerIterationController, TalkerOutputProcessor |
| `engines/omni/runtime/sglang_ar.py` | Modify | Capture hidden states in SGLangModelRunner |
| `engines/ar/sglang_backend/model_worker.py` | Modify | Add TalkerModelWorker or extend ModelWorker |
| `serve/openai_api.py` | Modify | Handle audio output in API responses |
| `tests/test_thinker_talker_e2e.py` | **New** | End-to-end integration tests |

**Estimated new code**: ~800-1200 lines
**Estimated modified code**: ~200-300 lines across existing files
