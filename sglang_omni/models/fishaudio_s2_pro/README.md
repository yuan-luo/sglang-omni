# FishAudio OpenAudio-S2-Pro

Text-to-speech via the DualAR (slow+fast transformer) architecture with DAC codec vocoding.

Same model architecture as S1-Mini (`fishaudio_s1`), but with a larger model
(4.5B params, dim=2560, 36 slow layers, 4 fast layers, 10 codebooks).

## Checkpoint Preparation

The S2-Pro checkpoint ships in HuggingFace format. Two extra files are needed
for the `fish_speech` runtime:

```bash
# 1. Generate tokenizer.tiktoken + special_tokens.json from tokenizer.json
python scripts/convert_hf_tokenizer.py /path/to/s2-pro

# 2. Symlink the DAC codec from S1-Mini (same codec)
ln -s /path/to/openaudio-s1-mini/codec.pth /path/to/s2-pro/codec.pth
```

## Quick Start

`torch.compile` and radix cache are **on by default**.

```bash
# Basic TTS
python examples/run_fishaudio_e2e.py \
    --checkpoint /path/to/s2-pro \
    --text "Hello, how are you?" \
    --output output.wav

# Voice cloning
python examples/run_fishaudio_e2e.py \
    --checkpoint /path/to/s2-pro \
    --text "Hello, how are you?" \
    --reference-audio ref.wav --reference-text "Transcript of ref audio." \
    --output output.wav

# Disable compile / radix cache if needed
python examples/run_fishaudio_e2e.py \
    --checkpoint /path/to/s2-pro \
    --text "Hello" --no-compile --no-radix-cache --output output.wav
```

## Architecture

Identical to S1-Mini — 3-stage linear pipeline:
`preprocessing` (CPU) → `tts_engine` (GPU) → `vocoder` (GPU).

All runtime components (DualAR engine, batch planner, input preparer, output
processor, radix cache) are reused from `fishaudio_s1`. Only the pipeline
configuration (default model ID, pipeline name) differs.

| Parameter | S1-Mini | S2-Pro |
|-----------|---------|--------|
| Total params | ~500M | ~4.5B |
| dim | 1024 | 2560 |
| Slow layers | 28 | 36 |
| Fast layers | 4 | 4 |
| Heads (slow) | 16 | 32 |
| KV heads | 8 | 8 |
| Codebooks | 10 | 10 |
| Codebook size | 4096 | 4096 |
| Config format | flat `dual_ar` | nested `fish_qwen3_omni` |
| Weight format | `model.pth` | sharded safetensors |
