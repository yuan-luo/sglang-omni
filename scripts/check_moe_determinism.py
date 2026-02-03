# SPDX-License-Identifier: Apache-2.0
"""Check MoE determinism: run torch thinker twice, compare MoE output at layer 0."""
from __future__ import annotations
import argparse, gc, torch, torch.nn.functional as F
from sglang_omni.models.weight_loader import resolve_model_path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--image-path", type=str, default="tests/data/cars.jpg")
    p.add_argument("--audio-path", type=str, default="tests/data/cough.wav")
    return p.parse_args()

def cmp(name, a, b):
    diff = (a.float()-b.float()).abs()
    print(f"  {name}: max_diff={diff.max().item():.4e}  identical={diff.max().item()==0}")

@torch.no_grad()
def preprocess(model_path, image_path, audio_path):
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
    from sglang_omni.frontends import ensure_audio_list, ensure_image_list
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, local_files_only=True)
    content = []
    images = ensure_image_list([image_path] if image_path else [])
    audios = ensure_audio_list([audio_path] if audio_path else [], target_sr=16000)
    for _ in images: content.append({"type": "image"})
    for _ in audios: content.append({"type": "audio"})
    content.append({"type": "text", "text": "Describe both the image and the audio content in detail."})
    messages = [{"role": "user", "content": content}]
    pt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    hi = processor(text=pt, images=images or None, audio=audios or None,
                   add_special_tokens=False, return_tensors="pt")
    return dict(hi)

@torch.no_grad()
def run_encoders(model_path, hf_inputs, device, dtype):
    results = {}
    pv = hf_inputs.get("pixel_values"); igt = hf_inputs.get("image_grid_thw")
    if pv is not None:
        from sglang_omni.models.qwen3_omni.components.torch_image_encoder import Qwen3OmniTorchImageEncoder
        enc = Qwen3OmniTorchImageEncoder(model_path, device=device, dtype=dtype)
        out = enc(pixel_values=pv, image_grid_thw=igt)
        results["image_embeds"] = out["image_embeds"].cpu()
        results["deepstack_visual_embeds"] = [e.cpu() for e in out["deepstack_visual_embeds"]]
        results["image_grid_thw"] = out["image_grid_thw"].cpu()
        del enc, out; gc.collect(); torch.cuda.empty_cache()
    inf = hf_inputs.get("input_features"); fam = hf_inputs.get("feature_attention_mask")
    if inf is not None:
        from sglang_omni.models.qwen3_omni.components.torch_audio_encoder import Qwen3OmniTorchAudioEncoder
        enc = Qwen3OmniTorchAudioEncoder(model_path, device=device, dtype=dtype)
        out = enc(input_features=inf, feature_attention_mask=fam)
        results["audio_embeds"] = out["audio_embeds"].cpu()
        results["audio_feature_lengths"] = out["audio_feature_lengths"].cpu()
        del enc, out; gc.collect(); torch.cuda.empty_cache()
    return results

def _kw(hf_inputs, enc_out):
    kw = {"output_hidden_states": True}
    for k in ("image_embeds","deepstack_visual_embeds","image_grid_thw","audio_embeds"):
        if k in enc_out: kw[k] = enc_out[k]
    fam = hf_inputs.get("feature_attention_mask")
    if fam is not None: kw["feature_attention_mask"] = fam
    return kw

@torch.no_grad()
def run_torch(model_path, hf_inputs, enc_out, device, dtype, run_id):
    from sglang_omni.models.qwen3_omni.components.torch_thinker import Qwen3OmniTorchThinker
    print(f"[Run {run_id}] Loading ...")
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)
    cap = {}; h = []
    layer = model.thinker.layers[0]
    h.append(layer.mlp.register_forward_pre_hook(lambda m,a,kw=None: cap.__setitem__("moe_input", a[0].detach().cpu())))
    h.append(layer.mlp.register_forward_hook(lambda m,a,o: cap.__setitem__("moe_output", (o[0] if isinstance(o,tuple) else o).detach().cpu())))
    h.append(layer.register_forward_hook(lambda m,a,o: cap.__setitem__("layer_output", (o[0] if isinstance(o,tuple) else o).detach().cpu())))
    ids = hf_inputs["input_ids"]
    mask = hf_inputs.get("attention_mask", torch.ones_like(ids))
    kw = _kw(hf_inputs, enc_out)
    print(f"[Run {run_id}] Forward ...")
    model(input_ids=ids, attention_mask=mask, **kw)
    for x in h: x.remove()
    del model; gc.collect(); torch.cuda.empty_cache()
    return cap

def main():
    args = parse_args()
    mp = str(resolve_model_path(args.model_path))
    hf_inputs = preprocess(mp, args.image_path, args.audio_path)
    enc_out = run_encoders(mp, hf_inputs, args.device, args.dtype)
    r1 = run_torch(mp, hf_inputs, enc_out, args.device, args.dtype, 1)
    r2 = run_torch(mp, hf_inputs, enc_out, args.device, args.dtype, 2)
    print(f"\n{'='*60}")
    print("Torch MoE self-comparison (layer 0):")
    for k in ["moe_input","moe_output","layer_output"]:
        if k in r1 and k in r2:
            cmp(k, r1[k], r2[k])

if __name__ == "__main__":
    main()
