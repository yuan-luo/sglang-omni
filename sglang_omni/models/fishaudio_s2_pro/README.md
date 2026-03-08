# FishAudio OpenAudio-S2-Pro

Text-to-speech via the FishQwen3Omni (MoE slow head + 10-codebook fast head) architecture with DAC codec vocoding.

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

## Benchmark Results (seed-tts-eval EN, 50 samples, no-compile)

### Performance

| Metric | BS=1 | BS=2 | BS=4 | BS=8 |
|---|---|---|---|---|
| tok/s (per-req) | 25.7 | 24.9 | 24.6 | 24.1 |
| RTF | 0.84 | 0.87 | 0.88 | 0.89 |
| Latency (mean) | 3.32s | 3.40s | 3.46s | 3.53s |
| TTFT (mean) | 45.6ms | 44.1ms | 45.2ms | 46.2ms |
| TTFB (mean) | 407.8ms | 429.3ms | 419.7ms | 422.3ms |

Per-request tok/s degrades only 6% from BS=1 to BS=8, meaning the audio
decoder codebook loop is efficiently batched across concurrent requests.

### Quality

| Metric | Value |
|---|---|
| Samples evaluated | 50 |
| WER (mean) | 1.85% |
| WER (median) | 0.0% |
| Samples >50% WER | 0 (0.0%) |

### Optimizations Applied

| Optimization | Status | Impact |
|---|---|---|
| topk + Gumbel-max sampling | Done | O(V log V) → O(V + k log k), no CUDA sync |
| CUDA graph for codebook loop | Done | Eliminates kernel launch overhead at BS>1 |
| Continuous batching support | Done | Realistic load testing with Poisson arrivals |
| RadixAttention prefix cache | Built-in | Shared ref audio → TTFT speedup on cache hit |
| torch.compile (max-autotune-no-cudagraphs) | Done | Avoids internal CUDA graph WER corruption |
| Text model CUDA graph | Not yet | Requires `input_ids`-only forward (hard) |

### Known Issues

- `torch.compile` produces 99% WER (pre-existing bug, unrelated to batch changes). Use `--no-compile` for correct output. With compile enabled, single-request tok/s is ~60 but audio quality is broken.

### Running Benchmarks

```bash
export S2PRO_CKPT=/root/.cache/huggingface/s2-pro/s2-pro
tar xzf /root/.cache/huggingface/seed_tts_eval_testset.tar.gz -C /tmp
export SEED_TTS=/tmp

# Single request + WER
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_sglang \
    --max-samples 50 --batch-sizes 1 --save-audio --no-compile

python benchmarks/eval_wer.py \
    --meta $SEED_TTS/seedtts_testset/en/meta.lst \
    --audio-dir results/s2pro_sglang/audio --lang en

# Batched performance
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_sglang_batched \
    --max-samples 50 --batch-sizes 1,2,4,8 --no-compile

# Continuous batching (Poisson arrival at 10 req/s)
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_continuous \
    --max-samples 50 --batch-sizes 4 --request-rate 10 --no-compile

# Shared-prefix cache test (same ref audio, different texts)
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_prefix \
    --max-samples 20 --batch-sizes 1 --test-shared-prefix --no-compile

# With CUDA graph enabled
CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_s2pro_sglang.py \
    --checkpoint $S2PRO_CKPT \
    --testset $SEED_TTS/seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_cudagraph \
    --max-samples 50 --batch-sizes 1,2,4,8 --enable-cuda-graph --no-compile
```

### Benchmark Flags Reference

| Flag | Description |
|---|---|
| `--batch-sizes 1,2,4,8` | Comma-separated batch sizes to test |
| `--request-rate <float>` | Poisson arrival rate (req/s). Default `inf` = burst mode |
| `--test-shared-prefix` | Test RadixAttention prefix cache with shared ref audio |
| `--enable-cuda-graph` | Enable CUDA graph capture for codebook loop |
| `--max-batch-size <int>` | Max batch size for CUDA graph capture (default 64) |
| `--no-compile` | Disable torch.compile (recommended until compile WER bug is fixed) |
| `--save-audio` | Save generated audio files for WER evaluation |
