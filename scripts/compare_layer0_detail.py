# SPDX-License-Identifier: Apache-2.0
"""Intra-layer comparison at layer 0: attention vs MoE."""
from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from sglang_omni.models.weight_loader import resolve_model_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--image-path", type=str, default="tests/data/cars.jpg")
    p.add_argument("--audio-path", type=str, default="tests/data/cough.wav")
    p.add_argument("--layer", type=int, default=0)
    return p.parse_args()


def compare(name, a, b):
    a_f, b_f = a.float(), b.float()
    diff = (a_f - b_f).abs()
    cos = F.cosine_similarity(a_f.reshape(1, -1), b_f.reshape(1, -1))
    a_sq = a_f.view(-1, a_f.shape[-1])
    b_sq = b_f.view(-1, b_f.shape[-1])
    per_tok = F.cosine_similarity(a_sq, b_sq, dim=-1)
    print(f"  {name}:")
    print(f"    cos={cos.item():.8f}  max_diff={diff.max().item():.4e}  "
          f"per_tok: min={per_tok.min().item():.6f} mean={per_tok.mean().item():.6f}")


@torch.no_grad()
def preprocess(model_path, image_path, audio_path):
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
    from sglang_omni.frontends import ensure_audio_list, ensure_image_list

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, local_files_only=True)
    content = []
    images = ensure_image_list([image_path] if image_path else [])
    audios = ensure_audio_list([audio_path] if audio_path else [], target_sr=16000)
    for _ in images:
        content.append({"type": "image"})
    for _ in audios:
        content.append({"type": "audio"})
    content.append({"type": "text", "text": "Describe both the image and the audio content in detail."})
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    hf_inputs = processor(text=prompt_text, images=images or None, audio=audios or None,
                          add_special_tokens=False, return_tensors="pt")
    return dict(hf_inputs)


@torch.no_grad()
def run_encoders(model_path, hf_inputs, device, dtype):
    results = {}
    pixel_values = hf_inputs.get("pixel_values")
    image_grid_thw = hf_inputs.get("image_grid_thw")
    if pixel_values is not None:
        from sglang_omni.models.qwen3_omni.components.torch_image_encoder import Qwen3OmniTorchImageEncoder
        enc = Qwen3OmniTorchImageEncoder(model_path, device=device, dtype=dtype)
        out = enc(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        results["image_embeds"] = out["image_embeds"].cpu()
        results["deepstack_visual_embeds"] = [e.cpu() for e in out["deepstack_visual_embeds"]]
        results["image_grid_thw"] = out["image_grid_thw"].cpu()
        del enc, out; gc.collect(); torch.cuda.empty_cache()

    input_features = hf_inputs.get("input_features")
    feature_attention_mask = hf_inputs.get("feature_attention_mask")
    if input_features is not None:
        from sglang_omni.models.qwen3_omni.components.torch_audio_encoder import Qwen3OmniTorchAudioEncoder
        enc = Qwen3OmniTorchAudioEncoder(model_path, device=device, dtype=dtype)
        out = enc(input_features=input_features, feature_attention_mask=feature_attention_mask)
        results["audio_embeds"] = out["audio_embeds"].cpu()
        results["audio_feature_lengths"] = out["audio_feature_lengths"].cpu()
        del enc, out; gc.collect(); torch.cuda.empty_cache()
    return results


def _build_thinker_kwargs(hf_inputs, encoder_out):
    kwargs = {"output_hidden_states": True}
    if "image_embeds" in encoder_out:
        kwargs["image_embeds"] = encoder_out["image_embeds"]
        kwargs["deepstack_visual_embeds"] = encoder_out["deepstack_visual_embeds"]
    if "image_grid_thw" in encoder_out:
        kwargs["image_grid_thw"] = encoder_out["image_grid_thw"]
    if "audio_embeds" in encoder_out:
        kwargs["audio_embeds"] = encoder_out["audio_embeds"]
    fam = hf_inputs.get("feature_attention_mask")
    if fam is not None:
        kwargs["feature_attention_mask"] = fam
    return kwargs


@torch.no_grad()
def run_hf_with_hooks(model_path, hf_inputs, encoder_out, device, dtype, layer_idx):
    from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
    print("[HF] Loading ...")
    model = Qwen3OmniSplitThinker(model_path, device=device, dtype=dtype)

    # Find the target layer in HF model
    layers = model.thinker.model.layers

    captured = {}

    def make_attn_hook(name):
        def hook(module, args, output):
            # output is (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple):
                captured[name] = output[0].detach().cpu()
            else:
                captured[name] = output.detach().cpu()
        return hook

    def make_pre_hook(name):
        def hook(module, args, kwargs=None):
            if args:
                captured[name] = args[0].detach().cpu()
            elif kwargs and 'hidden_states' in kwargs:
                captured[name] = kwargs['hidden_states'].detach().cpu()
        return hook

    def make_output_hook(name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach().cpu()
            else:
                captured[name] = output.detach().cpu()
        return hook

    target = layers[layer_idx]
    handles = []
    handles.append(target.register_forward_pre_hook(make_pre_hook("layer_input")))
    handles.append(target.self_attn.register_forward_hook(make_attn_hook("attn_output")))
    handles.append(target.register_forward_hook(make_output_hook("layer_output")))

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[HF] Forward ...")
    out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    print(f"[HF] Done. Captured: {list(captured.keys())}")

    for h in handles:
        h.remove()
    del handles, model, out, target, layers
    gc.collect(); torch.cuda.empty_cache()
    import time; time.sleep(1)
    gc.collect(); torch.cuda.empty_cache()
    return captured


@torch.no_grad()
def run_torch_with_hooks(model_path, hf_inputs, encoder_out, device, dtype, layer_idx):
    from sglang_omni.models.qwen3_omni.components.torch_thinker import Qwen3OmniTorchThinker
    print("[Torch] Loading ...")
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)

    captured = {}

    def make_pre_hook(name):
        def hook(module, args, kwargs=None):
            if args:
                captured[name] = args[0].detach().cpu()
        return hook

    def make_output_hook(name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach().cpu()
            else:
                captured[name] = output.detach().cpu()
        return hook

    target = model.thinker.layers[layer_idx]
    target.register_forward_pre_hook(make_pre_hook("layer_input"))
    target.self_attn.register_forward_hook(make_output_hook("attn_output"))
    target.register_forward_hook(make_output_hook("layer_output"))

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[Torch] Forward ...")
    out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    print(f"[Torch] Done. Captured: {list(captured.keys())}")

    del model, out; gc.collect(); torch.cuda.empty_cache()
    return captured


def main():
    args = parse_args()
    model_path = str(resolve_model_path(args.model_path))

    print("Step 1: Preprocess")
    hf_inputs = preprocess(model_path, args.image_path, args.audio_path)
    print(f"  input_ids: {hf_inputs['input_ids'].shape}")

    print("\nStep 2: Encoders")
    encoder_out = run_encoders(model_path, hf_inputs, args.device, args.dtype)

    print(f"\nStep 3: HF thinker with hooks at layer {args.layer}")
    hf_cap = run_hf_with_hooks(model_path, hf_inputs, encoder_out, args.device, args.dtype, args.layer)

    print(f"\nStep 4: Torch thinker with hooks at layer {args.layer}")
    torch_cap = run_torch_with_hooks(model_path, hf_inputs, encoder_out, args.device, args.dtype, args.layer)

    print(f"\n{'='*60}")
    print(f"Layer {args.layer} intra-layer comparison:")
    print(f"{'='*60}")

    for key in ["layer_input", "attn_output", "layer_output"]:
        if key in hf_cap and key in torch_cap:
            compare(key, hf_cap[key], torch_cap[key])
        else:
            missing = []
            if key not in hf_cap:
                missing.append("HF")
            if key not in torch_cap:
                missing.append("Torch")
            print(f"  {key}: MISSING from {', '.join(missing)}")


if __name__ == "__main__":
    main()
