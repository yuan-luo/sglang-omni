# FishAudio S2-Pro

Text-to-speech via the `FishQwen3OmniForCausalLM` architecture with DAC VQGAN
codec vocoding.

**This is a fundamentally different model from S1-Mini.** S2-Pro uses:

- `FishQwen3OmniForCausalLM` (HuggingFace `AutoModel`-compatible)
- Built-in audio decoder for codebook generation
- Qwen3 chat-format prompts (`<|im_start|>system/user/assistant<|im_end|>`)
- HuggingFace `PreTrainedTokenizerFast` tokenizer
- Repetition Aware Sampling (RAS)
- Constrained semantic decoding

## Checkpoint Preparation

The S2-Pro checkpoint ships in HuggingFace format. Only the DAC codec
symlink is needed:

```bash
# Symlink the DAC codec from S1-Mini (same codec)
ln -s /path/to/openaudio-s1-mini/codec.pth /path/to/s2-pro/codec.pth
```

## Quick Start

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
```

## Architecture

3-stage linear pipeline:
`preprocessing` (CPU) → `tts_engine` (GPU) → `vocoder` (GPU)

### Key differences from S1-Mini

| Aspect | S1-Mini | S2-Pro |
|--------|---------|--------|
| Model class | `DualARTransformer` | `FishQwen3OmniForCausalLM` |
| Loading | Custom `from_pretrained` | `AutoModel.from_pretrained` |
| Tokenizer | `FishTokenizer` (tiktoken) | `PreTrainedTokenizerFast` (HF) |
| Prompt format | `ContentSequence(modality="interleave")` | `Conversation` (Qwen3 chat) |
| Generation | Manual step loop (`inference.py`) | `qwen3.generate` (built-in) |
| Fast decoder | Manual `forward_generate_fast` | Built-in `audio_decoder` |
| Anti-repetition | Repetition penalty | RAS (Repetition Aware Sampling) |
| Constrained decode | Manual masking | `constrain_to_semantic=True` |

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

## Prompt Format

S2-Pro uses the Qwen3 chat format with `<|speaker:0|>` tags:

```
<|im_start|>system
convert the provided text to speech reference to the following:

Text:
<|speaker:0|>{reference_text}

Speech:
[VQ CODES]<|im_end|>
<|im_start|>user
<|speaker:0|>{target_text}<|im_end|>
<|im_start|>assistant
<|voice|>
```

## seed-tts-eval Results

With `temperature=0.7, top_p=0.7, top_k=30`:

| Metric | 10-sample test |
|--------|---------------|
| WER | 3.8% |
| Natural stop rate | 100% |
