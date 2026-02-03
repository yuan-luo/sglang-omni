# SPDX-License-Identifier: Apache-2.0
"""Drill into layer 0 MoE: find exactly where max_diff > 1e-4."""
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
    p.add_argument("--layer", type=int, default=0)
    return p.parse_args()


def compare(name, a, b):
    a_f, b_f = a.float(), b.float()
    diff = (a_f - b_f).abs()
    cos = F.cosine_similarity(a_f.reshape(1, -1), b_f.reshape(1, -1)).item()
    a_sq = a_f.view(-1, a_f.shape[-1])
    b_sq = b_f.view(-1, b_f.shape[-1])
    per_tok = F.cosine_similarity(a_sq, b_sq, dim=-1)
    print(f"  {name}:")
    print(f"    cos={cos:.8f}  max_diff={diff.max().item():.4e}  mean_diff={diff.mean().item():.4e}")
    print(f"    per_tok: min={per_tok.min().item():.6f} mean={per_tok.mean().item():.6f}")
    return diff.max().item()


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
def run_hf(model_path, hf_inputs, encoder_out, device, dtype, layer_idx):
    from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
    print("[HF] Loading ...")
    model = Qwen3OmniSplitThinker(model_path, device=device, dtype=dtype)
    layer = model.thinker.model.layers[layer_idx]

    captured = {}
    handles = []

    # 1. Layer input (pre-hook on layer)
    def layer_pre(module, args, kwargs=None):
        captured["layer_input"] = args[0].detach().cpu()
    handles.append(layer.register_forward_pre_hook(layer_pre))

    # 2. Attention output
    def attn_out(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        captured["attn_output"] = t.detach().cpu()
    handles.append(layer.self_attn.register_forward_hook(attn_out))

    # 3. Post-attn residual + post_attention_layernorm input (pre-hook on mlp)
    def mlp_pre(module, args, kwargs=None):
        captured["moe_input"] = args[0].detach().cpu()
    handles.append(layer.mlp.register_forward_pre_hook(mlp_pre))

    # 4. Router output (gate logits and top-k indices)
    if hasattr(layer.mlp, 'gate'):
        def gate_out(module, args, output):
            if isinstance(output, tuple):
                captured["gate_logits"] = output[0].detach().cpu()
                if len(output) > 1 and output[1] is not None:
                    captured["gate_routing_weights"] = output[1].detach().cpu()
                if len(output) > 2 and output[2] is not None:
                    captured["gate_selected_experts"] = output[2].detach().cpu()
            else:
                captured["gate_logits"] = output.detach().cpu()
        handles.append(layer.mlp.gate.register_forward_hook(gate_out))

    # 5. MoE output (SparseMoeBlock output)
    def mlp_out(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        captured["moe_output"] = t.detach().cpu()
    handles.append(layer.mlp.register_forward_hook(mlp_out))

    # 6. Full layer output
    def layer_out(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        captured["layer_output"] = t.detach().cpu()
    handles.append(layer.register_forward_hook(layer_out))

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[HF] Forward ...")
    model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    print(f"[HF] Captured: {list(captured.keys())}")

    for h in handles:
        h.remove()
    del handles, model, layer
    gc.collect(); torch.cuda.empty_cache()
    import time; time.sleep(1)
    gc.collect(); torch.cuda.empty_cache()
    return captured


@torch.no_grad()
def run_torch(model_path, hf_inputs, encoder_out, device, dtype, layer_idx):
    from sglang_omni.models.qwen3_omni.components.torch_thinker import Qwen3OmniTorchThinker
    print("[Torch] Loading ...")
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)
    layer = model.thinker.layers[layer_idx]

    captured = {}
    handles = []

    def layer_pre(module, args, kwargs=None):
        captured["layer_input"] = args[0].detach().cpu()
    handles.append(layer.register_forward_pre_hook(layer_pre))

    def attn_out(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        captured["attn_output"] = t.detach().cpu()
    handles.append(layer.self_attn.register_forward_hook(attn_out))

    def mlp_pre(module, args, kwargs=None):
        captured["moe_input"] = args[0].detach().cpu()
    handles.append(layer.mlp.register_forward_pre_hook(mlp_pre))

    if hasattr(layer.mlp, 'gate'):
        def gate_out(module, args, output):
            if isinstance(output, tuple):
                captured["gate_logits"] = output[0].detach().cpu()
                if len(output) > 1 and output[1] is not None:
                    captured["gate_routing_weights"] = output[1].detach().cpu()
                if len(output) > 2 and output[2] is not None:
                    captured["gate_selected_experts"] = output[2].detach().cpu()
            else:
                captured["gate_logits"] = output.detach().cpu()
        handles.append(layer.mlp.gate.register_forward_hook(gate_out))

    def mlp_out(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        captured["moe_output"] = t.detach().cpu()
    handles.append(layer.mlp.register_forward_hook(mlp_out))

    def layer_out(module, args, output):
        t = output[0] if isinstance(output, tuple) else output
        captured["layer_output"] = t.detach().cpu()
    handles.append(layer.register_forward_hook(layer_out))

    input_ids = hf_inputs["input_ids"]
    attention_mask = hf_inputs.get("attention_mask", torch.ones_like(input_ids))
    kwargs = _build_thinker_kwargs(hf_inputs, encoder_out)

    print("[Torch] Forward ...")
    model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    print(f"[Torch] Captured: {list(captured.keys())}")

    for h in handles:
        h.remove()
    del handles, model, layer
    gc.collect(); torch.cuda.empty_cache()
    return captured


def main():
    args = parse_args()
    model_path = str(resolve_model_path(args.model_path))
    L = args.layer

    print("Step 1: Preprocess")
    hf_inputs = preprocess(model_path, args.image_path, args.audio_path)

    print("\nStep 2: Encoders")
    encoder_out = run_encoders(model_path, hf_inputs, args.device, args.dtype)

    print(f"\nStep 3: HF thinker layer {L}")
    hf_cap = run_hf(model_path, hf_inputs, encoder_out, args.device, args.dtype, L)

    print(f"\nStep 4: Torch thinker layer {L}")
    torch_cap = run_torch(model_path, hf_inputs, encoder_out, args.device, args.dtype, L)

    print(f"\n{'='*70}")
    print(f"Layer {L} detailed MoE bisection:")
    print(f"{'='*70}")

    for key in ["layer_input", "attn_output", "moe_input", "gate_logits", "moe_output", "layer_output"]:
        if key in hf_cap and key in torch_cap:
            md = compare(key, hf_cap[key], torch_cap[key])
            if md > 1e-4:
                print(f"    >>> ABOVE 1e-4 THRESHOLD <<<")
        else:
            present = []
            if key in hf_cap: present.append("HF")
            if key in torch_cap: present.append("Torch")
            missing = []
            if key not in hf_cap: missing.append("HF")
            if key not in torch_cap: missing.append("Torch")
            print(f"  {key}: present={present}, MISSING={missing}")

    # If gate_logits exist, compare top-k expert selection
    if "gate_logits" in hf_cap and "gate_logits" in torch_cap:
        print(f"\n  Expert selection overlap (top-8):")
        hg = hf_cap["gate_logits"].float().squeeze(0)
        tg = torch_cap["gate_logits"].float().squeeze(0)
        _, hf_topk = torch.topk(hg, 8, dim=-1)
        _, torch_topk = torch.topk(tg, 8, dim=-1)
        overlaps = []
        for tok in range(hf_topk.shape[0]):
            s1 = set(hf_topk[tok].tolist())
            s2 = set(torch_topk[tok].tolist())
            overlaps.append(len(s1 & s2))
        overlaps_t = torch.tensor(overlaps, dtype=torch.float)
        print(f"    min={overlaps_t.min().item():.0f}/8  mean={overlaps_t.mean().item():.1f}/8  "
              f"exact_match={int((overlaps_t == 8).sum())}/{len(overlaps)}")


if __name__ == "__main__":
    main()
