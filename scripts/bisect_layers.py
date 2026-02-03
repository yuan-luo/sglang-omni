# SPDX-License-Identifier: Apache-2.0
"""Bisect layers: capture every layer's input/attn/output from both backends in ONE pass each."""
from __future__ import annotations

import argparse
import gc
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
    return p.parse_args()


def compare(a, b):
    a_f, b_f = a.float(), b.float()
    cos = F.cosine_similarity(a_f.reshape(1, -1), b_f.reshape(1, -1)).item()
    a_sq = a_f.view(-1, a_f.shape[-1])
    b_sq = b_f.view(-1, b_f.shape[-1])
    per_tok = F.cosine_similarity(a_sq, b_sq, dim=-1)
    return cos, per_tok.min().item(), per_tok.mean().item()


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
def run_hf_all_layers(model_path, hf_inputs, encoder_out, device, dtype):
    from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
    print("[HF] Loading ...")
    model = Qwen3OmniSplitThinker(model_path, device=device, dtype=dtype)
    layers = model.thinker.model.layers
    num_layers = len(layers)

    captured = {}

    def make_pre_hook(layer_idx):
        def hook(module, args, kwargs=None):
            t = args[0] if args else (kwargs.get('hidden_states') if kwargs else None)
            if t is not None:
                captured[f"L{layer_idx}_input"] = t.detach().cpu()
        return hook

    def make_attn_hook(layer_idx):
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            captured[f"L{layer_idx}_attn"] = t.detach().cpu()
        return hook

    def make_output_hook(layer_idx):
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            captured[f"L{layer_idx}_output"] = t.detach().cpu()
        return hook

    handles = []
    for i in range(num_layers):
        handles.append(layers[i].register_forward_pre_hook(make_pre_hook(i)))
        handles.append(layers[i].self_attn.register_forward_hook(make_attn_hook(i)))
        handles.append(layers[i].register_forward_hook(make_output_hook(i)))

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[HF] Forward ...")
    model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    print(f"[HF] Done. Captured {len(captured)} tensors across {num_layers} layers")

    for h in handles:
        h.remove()
    del handles, model, layers
    gc.collect(); torch.cuda.empty_cache()
    import time; time.sleep(1)
    gc.collect(); torch.cuda.empty_cache()
    return captured, num_layers


@torch.no_grad()
def run_torch_all_layers(model_path, hf_inputs, encoder_out, device, dtype):
    from sglang_omni.models.qwen3_omni.components.torch_thinker import Qwen3OmniTorchThinker
    print("[Torch] Loading ...")
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)
    layers = model.thinker.layers
    num_layers = len(layers)

    captured = {}

    def make_pre_hook(layer_idx):
        def hook(module, args, kwargs=None):
            t = args[0] if args else None
            if t is not None:
                captured[f"L{layer_idx}_input"] = t.detach().cpu()
        return hook

    def make_attn_hook(layer_idx):
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            captured[f"L{layer_idx}_attn"] = t.detach().cpu()
        return hook

    def make_output_hook(layer_idx):
        def hook(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            captured[f"L{layer_idx}_output"] = t.detach().cpu()
        return hook

    handles = []
    for i in range(num_layers):
        handles.append(layers[i].register_forward_pre_hook(make_pre_hook(i)))
        handles.append(layers[i].self_attn.register_forward_hook(make_attn_hook(i)))
        handles.append(layers[i].register_forward_hook(make_output_hook(i)))

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[Torch] Forward ...")
    model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    print(f"[Torch] Done. Captured {len(captured)} tensors across {num_layers} layers")

    del model
    gc.collect(); torch.cuda.empty_cache()
    return captured


def main():
    args = parse_args()
    model_path = str(resolve_model_path(args.model_path))

    print("Step 1: Preprocess")
    hf_inputs = preprocess(model_path, args.image_path, args.audio_path)
    print(f"  input_ids: {hf_inputs['input_ids'].shape}")

    print("\nStep 2: Encoders")
    encoder_out = run_encoders(model_path, hf_inputs, args.device, args.dtype)

    print("\nStep 3: HF thinker (all layers)")
    hf_cap, num_layers = run_hf_all_layers(model_path, hf_inputs, encoder_out, args.device, args.dtype)

    print("\nStep 4: Torch thinker (all layers)")
    torch_cap = run_torch_all_layers(model_path, hf_inputs, encoder_out, args.device, args.dtype)

    print(f"\n{'='*80}")
    print(f"Layer-by-layer bisection ({num_layers} layers)")
    print(f"{'='*80}")
    print(f"{'Layer':>5} | {'Input cos':>10} {'(min)':>8} | {'Attn cos':>10} {'(min)':>8} | {'Output cos':>10} {'(min)':>8}")
    print("-" * 80)

    for i in range(num_layers):
        cols = []
        for key in ["input", "attn", "output"]:
            hk = f"L{i}_{key}"
            tk = f"L{i}_{key}"
            if hk in hf_cap and tk in torch_cap:
                cos, ptmin, ptmean = compare(hf_cap[hk], torch_cap[tk])
                cols.append(f"{cos:10.6f} {ptmin:8.4f}")
            else:
                cols.append(f"{'MISSING':>10} {'':>8}")
        print(f"{i:5d} | {' | '.join(cols)}")

    # Find first layer where output per_tok_min < 0.9
    print(f"\n{'='*80}")
    print("First layer where output per_tok_min < 0.9:")
    for i in range(num_layers):
        hk = f"L{i}_output"
        tk = f"L{i}_output"
        if hk in hf_cap and tk in torch_cap:
            cos, ptmin, ptmean = compare(hf_cap[hk], torch_cap[tk])
            if ptmin < 0.9:
                print(f"  Layer {i}: output per_tok_min={ptmin:.6f}")
                # Also print attn detail
                ak = f"L{i}_attn"
                if ak in hf_cap and ak in torch_cap:
                    acos, aptmin, aptmean = compare(hf_cap[ak], torch_cap[ak])
                    print(f"  Layer {i}: attn   per_tok_min={aptmin:.6f}")
                ik = f"L{i}_input"
                if ik in hf_cap and ik in torch_cap:
                    icos, iptmin, iptmean = compare(hf_cap[ik], torch_cap[ik])
                    print(f"  Layer {i}: input  per_tok_min={iptmin:.6f}")
                break
    else:
        print("  None found - all layers have per_tok_min >= 0.9")


if __name__ == "__main__":
    main()
