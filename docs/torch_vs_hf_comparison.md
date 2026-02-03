# Torch vs HF Backend: Operator Comparison Record

## Verified Identical

| Component | Location | Method | Result |
|-----------|----------|--------|--------|
| **Weights** | All layers | Bit-for-bit comparison (gate_up_proj, down_proj, gate.weight at L0) | Exact match |
| **Input embeddings** | `embed_tokens` | Precision comparison script | cosine=1.0 |
| **MoE experts (grouped_mm)** | `modules.py:MoeExperts` | Rewrote to use `torch._grouped_mm` matching HF's default `grouped_mm_experts_forward` | Bit-for-bit identical at all 48 layers |
| **TopKRouter dtype** | `modules.py:TopKRouter` | Softmax in float32, routing weights kept in float32 | Matches HF (both keep float32 until after multiplication) |
| **TopKRouter sorted** | `modules.py:TopKRouter` | Changed `sorted=False` → `sorted=True` | Matches HF default |
| **Attention mask (SDPA)** | `modeling.py:_maybe_apply_causal_mask` + `modules.py:Attention` | Returns None for 2D/None masks, uses `is_causal=True` | Matches HF |
| **GQA handling** | `modules.py:Attention` + HF SDPA integration | Both use `enable_gqa=True` when `attn_mask=None` + PyTorch≥2.5 | Matches HF |
| **RoPE (`apply_rotary_pos_emb`)** | `modules.py:99-111` vs HF `modeling_qwen3_omni_moe.py:1493-1515` | Code comparison | Identical |
| **Audio data flow** | Pipeline merge → torch_thinker | Diagnostic prints | audio_embeds matched |
| **shared_expert** | `SparseMoeBlock` | Config check | `shared_expert_intermediate_size=0`, not used |
| **Deepstack operation** | `torch_thinker.py:_apply_deepstack` vs HF `_deepstack_process` | Code comparison | Identical |
| **Deepstack mask** | `visual_pos_masks` | Mask creation comparison | Identical |
| **SDPA kernel selection** | `F.scaled_dot_product_attention` | Code comparison | Both: `is_causal=True`, `enable_gqa=True` |
| **SDPA scale parameter** | `Attention.forward` | HF `scale=head_dim**-0.5` = PyTorch default | Numerically identical |
| **Decode loop** | `torch_thinker.py` vs HF | Code comparison | Identical position_ids computation |
| **SDPA determinism** | `F.scaled_dot_product_attention` | Run torch twice | max_diff=0 on H200 |
| **MoE determinism** | `MoeExperts` | Run torch twice | max_diff=0 |

## Issues Found & Fixed

| # | Component | Issue | Impact | Status |
|---|-----------|-------|--------|--------|
| 1 | **MoE experts (gate_up_proj fusion)** | Separate gate_proj + up_proj (two matmuls) vs fused gate_up_proj (one matmul + chunk) | Different FP reduction order | **FIXED** — fused to match HF |
| 2 | **TopKRouter cast** | `top_k_weights.to(hidden_states.dtype)` cast to bf16 | HF keeps in float32 | **FIXED** — removed cast |
| 3 | **nn.RMSNorm** | PyTorch `nn.RMSNorm` fused CUDA kernel differs from HF's Python implementation | max_diff=9.77e-4 per norm, compounds through 48 layers | **FIXED** — replaced with HF-compatible Python `RMSNorm` |
| 4 | **QK norm order** | Applied QK norm AFTER transpose `(B,H,S,D)`, HF applies BEFORE `(B,S,H,D)` | Different CUDA kernel paths | **FIXED** — reordered to match HF |
| 5 | **TopKRouter sorted** | `sorted=False` vs HF `sorted=True` | Different positional ordering of routing weights | **FIXED** — changed to `sorted=True` |
| 6 | **MoE expert computation kernel** | Per-expert `F.linear` loop vs HF's `torch._grouped_mm` | max_diff=7.8e-3 per layer, compounds through 48 layers, causes generation divergence | **FIXED** — switched to `torch._grouped_mm` matching HF's default `grouped_mm` implementation |
| 7 | **is_causal with use_cache** | `is_causal = attn_mask is None and past_key_value is None` evaluated AFTER `past_key_value = (k, v)` assignment | During prefill with `use_cache=True`, `is_causal=False` → bidirectional attention → corrupted KV cache → wrong decode output | **FIXED** — changed to `is_causal = attn_mask is None and q_len > 1` matching HF's SDPA logic |

## Root Cause Analysis

### Issue #6: The final and most impactful fix

**Discovery**: After fixing issues #1-5, layer 0 showed all sub-components identical (max_diff=0) up through moe_input and routing, but moe_output had max_diff=7.8e-3 despite identical inputs, weights, routing, and deterministic computation.

**Investigation**:
1. HF's `@use_experts_implementation` decorator (in `transformers/integrations/moe.py`) replaces the default forward method
2. `config._experts_implementation` defaults to `"grouped_mm"` (not `"eager"`) in transformers 5.0
3. `grouped_mm_experts_forward` uses `torch._grouped_mm` — a completely different CUDA kernel from per-expert `F.linear`

**Verification** (`scripts/verify_grouped_mm_hypothesis.py`):
```
HF(grouped_mm) vs Torch(F.linear): moe_output max_diff = 7.8e-3
HF(eager)      vs Torch(F.linear): moe_output max_diff = 0.0000e+00  ← CONFIRMED
```

**Fix**: Replaced `MoeExperts.forward` with `torch._grouped_mm` implementation matching HF's `grouped_mm_experts_forward` exactly:
- Sort tokens by expert ID
- Use `torch._grouped_mm` with offsets for gate_up_proj and down_proj
- Accumulate via `.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)` (deterministic)

## Final Bisection Results (after all 6 fixes)

### Layer 0 Sub-Component Comparison

| Sub-component | max_diff | identical? |
|---------------|----------|------------|
| layer_input | 0 | YES |
| attn_output | 0 | YES |
| moe_input | 0 | YES |
| moe_output | 0 | YES |
| layer_output | 0 | YES |
| Expert overlap | 6103/6103 = 100% | YES |

### Full 48-Layer Scan

All 48 layers: **per_tok_min = 1.0000** (bit-for-bit identical at every layer, every token)

| Layer | Input per_tok_min | Attn per_tok_min | Output per_tok_min |
|-------|------------------|------------------|-------------------|
| 0 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 |
| ... | 1.000 | 1.000 | 1.000 |
| 47 | 1.000 | 1.000 | 1.000 |

## E2E Test Results

| Test Input | Torch Backend | HF Backend | Match? |
|------------|--------------|------------|--------|
| cars.jpg + cough.wav | Describes all 4 cars + "person coughing" | Describes both | **YES** |
| screenshot + daisy.wav | Image: dog description + Audio: "female voice... ASR... data scaling" | Same | **YES** (word-for-word identical) |

## Files Modified

| File | Changes |
|------|---------|
| `modules.py` | `RMSNorm` class (HF-compatible), `MoeExperts.forward` (grouped_mm), QK norm order, TopKRouter sorted, `is_causal` fix |
| `modeling.py` | Import `RMSNorm`, replace `nn.RMSNorm` |
| `torch_thinker.py` | `torch.cuda.empty_cache()` after weight loading to release peak memory |

## Environment

- PyTorch: 2.9.1+cu129
- transformers: 5.0.0
- dtype: bfloat16
- Model: Qwen3-Omni-30B-A3B-Instruct (128 experts, top-8, 48 MoE layers)
- Hardware: NVIDIA H200
