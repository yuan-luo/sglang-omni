# FishAudio OpenAudio-S1

Text-to-speech via the DualAR (slow+fast transformer) architecture with DAC codec vocoding.

## Quick Start

```bash
# Basic TTS
python examples/run_fishaudio_e2e.py \
    --checkpoint fishaudio/openaudio-s1-mini \
    --text "Hello, how are you?" \
    --output output.wav

# With torch.compile (~9x decode speedup)
python examples/run_fishaudio_e2e.py \
    --checkpoint fishaudio/openaudio-s1-mini \
    --text "Hello, how are you?" \
    --compile --output output.wav

# Voice cloning
python examples/run_fishaudio_e2e.py \
    --checkpoint fishaudio/openaudio-s1-mini \
    --text "Hello, how are you?" \
    --reference-audio ref.wav --reference-text "Transcript of ref audio." \
    --output output.wav

# Voice cloning + radix cache (reuses KV for shared voice prefix)
python examples/run_fishaudio_e2e.py \
    --checkpoint fishaudio/openaudio-s1-mini \
    --text "Hello, how are you?" \
    --reference-audio ref.wav --reference-text "Transcript of ref audio." \
    --use-radix-cache --output output.wav
```

## Pipeline

3-stage linear pipeline: `preprocessing` (CPU) &rarr; `tts_engine` (GPU) &rarr; `vocoder` (GPU).

| Stage | Executor | What it does |
|-------|----------|--------------|
| `preprocessing` | `PreprocessingExecutor` | Tokenize text, encode reference audio via DAC codec, build DualAR prompt |
| `tts_engine` | `EngineExecutor` wrapping `OmniEngine` | DualAR decode: slow transformer samples semantic token, fast transformer samples 4 codebook tokens per step |
| `vocoder` | `PreprocessingExecutor` | DAC codec decode: VQ codes &rarr; 44.1kHz audio waveform |

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `fishaudio/openaudio-s1-mini` | HF model ID or local path |
| `--text` | `"Hello, how are you today?"` | Text to synthesize |
| `--device` | `cuda:0` | GPU device |
| `--output` / `-o` | None | Save output as WAV |
| `--reference-audio` | None | Reference WAV for voice cloning |
| `--reference-text` | `""` | Transcript of reference audio |
| `--compile` | off | Enable `torch.compile` for decode steps |
| `--use-radix-cache` | off | Radix-tree prefix cache for voice ref reuse |
| `--max-new-tokens` | 1024 | Max decode steps |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-p` | 0.8 | Top-p sampling |
