# FishAudio OpenAudio-S2-Pro

Text-to-speech via the FishQwen3Omni (MoE slow head + 10-codebook fast head) architecture with DAC codec vocoding.

## Architecture

### Fish-style 2-D `input_ids`

The text model receives **2-D integer tensors** `[N, K+1]` where column 0 holds
semantic token IDs and columns 1..K hold raw codebook indices. VQ codebook
embeddings are computed **inline** in the model's forward pass
(`_embed_with_codebooks`), eliminating the old `input_embeds` float-injection
path.

This design:
- Keeps all model inputs as integer tensors (CUDA-graph friendly)
- Enables future text-model CUDA graph capture (no dynamic float embedding injection)
- Maintains full backward compatibility — when `input_ids` is 1-D the model
  falls back to standard `embed_tokens`

### Two-stage decode

Each text model step produces a **semantic token**. A separate audio decoder
then generates 10 codebook tokens conditioned on the text model's hidden state.
The codebook loop runs as a batched CUDA-graph-captured function for efficiency.

## Quick Start

```bash
# Basic TTS
python examples/run_fishaudio_s2pro_e2e.py \
    --text "Hello, how are you?" \
    --output output.wav

# Voice cloning
python examples/run_fishaudio_s2pro_e2e.py \
    --text "Hello, how are you?" \
    --reference-audio ref.wav --reference-text "Transcript of ref audio." \
    --output output.wav
```

## Benchmark Results (seed-tts-eval EN, 50 samples)

### Performance (text model CUDA graph, best config)

| Metric | BS=1 | BS=2 | BS=4 | BS=8 |
|---|---|---|---|---|
| tok/s (per-req) | **79.5** | 35.1 | 33.0 | 31.0 |
| tok/s (agg) | 73.4 | 35.4 | 33.4 | 31.2 |
| RTF | **0.301** | 0.624 | 0.659 | 0.696 |
| Latency (mean) | 1.16s | 2.37s | 2.55s | 2.74s |
| TTFB (mean) | 258ms | 336ms | 338ms | 342ms |

### Quality

| Metric | Value |
|---|---|
| Samples evaluated | 50 |
| WER (mean, text CUDA graph) | 1.61% |
| WER (mean, compile) | 1.10% |
| WER (mean, no-compile) | 0.99% |
| WER (median) | 0.0% |
| Samples >50% WER | 0 (0.0%) |

### vs FishAudio Official (H200)

| Metric | FishAudio Official (H200) | Ours (H100) | Gap |
|---|---|---|---|
| RTF (BS=1) | **0.195** | 0.301 | 1.54x |
| TTFA | **~100 ms** | 258 ms | 2.6x |
| Throughput (agg) | **3,000+ tok/s** | ~248 (BS=8) | 12x |
| WER (EN) | **0.99%** | 1.61% | OK |

### Optimizations Applied

| Optimization | Status | Impact |
|---|---|---|
| **Text model CUDA graph** | **Done** | **BS=1: 52 → 79.5 tok/s (+51%)** |
| Fish-style 2-D `input_ids` | Done | Integer-only model inputs, CUDA graph ready |
| topk + Gumbel-max sampling | Done | O(V log V) -> O(V + k log k), no CUDA sync |
| CUDA graph for codebook loop | Done | Eliminates kernel launch overhead at BS>1 |
| Continuous batching support | Done | Realistic load testing with Poisson arrivals |
| RadixAttention prefix cache | Built-in | Shared ref audio -> TTFT speedup on cache hit |
| torch.compile (max-autotune-no-cudagraphs) | Done | Avoids internal CUDA graph WER corruption |

### TODO

| Priority | Task | Expected Impact |
|---|---|---|
| P0 | torch.compile inside text CUDA graph | BS=1: 79 → ~100+ tok/s |
| P1 | Combined text+audio CUDA graph | TTFB: 258 → ~200ms |
| P2 | Audio decoder batch parallelism | Batch scaling: BS=8 agg 248 → ~500+ tok/s |
| P3 | Text/audio overlap scheduling | Hide codebook latency behind next text step |
| P5 | Benchmark on H200 | Fair comparison (expected RTF ~0.21) |

## Running Benchmarks

```bash
export S2PRO_CKPT=/root/.cache/huggingface/s2-pro/s2-pro
tar xzf /root/.cache/huggingface/seed_tts_eval_testset.tar.gz -C /tmp
export SEED_TTS=/tmp

# Single request + WER (compile mode)
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_sglang \
    --max-samples 50 --batch-sizes 1 --save-audio

python benchmarks/eval_wer.py \
    --meta $SEED_TTS/seedtts_testset/en/meta.lst \
    --audio-dir results/s2pro_sglang/audio --lang en

# Batched performance
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_sglang_batched \
    --max-samples 50 --batch-sizes 1,2,4,8

# Continuous batching (Poisson arrival at 10 req/s)
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_continuous \
    --max-samples 50 --batch-sizes 4 --request-rate 10

# Shared-prefix cache test (same ref audio, different texts)
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_prefix \
    --max-samples 20 --batch-sizes 1 --test-shared-prefix

# With CUDA graph for codebook loop
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_cudagraph \
    --max-samples 50 --batch-sizes 1,2,4,8 --enable-cuda-graph

# Text model CUDA graph (best performance)
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_text_cuda_graph \
    --max-samples 50 --batch-sizes 1,2,4,8 \
    --enable-text-cuda-graph --max-batch-size 16 --save-audio

# No-compile mode (for comparison)
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_nocompile \
    --max-samples 50 --batch-sizes 1 --no-compile --save-audio
```

## Benchmark Flags Reference

| Flag | Description |
|---|---|
| `--batch-sizes 1,2,4,8` | Comma-separated batch sizes to test |
| `--request-rate <float>` | Poisson arrival rate (req/s). Default `inf` = burst mode |
| `--test-shared-prefix` | Test RadixAttention prefix cache with shared ref audio |
| `--enable-cuda-graph` | Enable CUDA graph capture for codebook loop |
| `--enable-text-cuda-graph` | Enable CUDA graph capture for text model decode |
| `--max-batch-size <int>` | Max batch size for CUDA graph capture (default 64) |
| `--no-compile` | Disable torch.compile |
| `--save-audio` | Save generated audio files for WER evaluation |
