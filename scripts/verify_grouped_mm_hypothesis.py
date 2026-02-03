# SPDX-License-Identifier: Apache-2.0
"""Verify hypothesis: MoE output diff comes from grouped_mm vs F.linear.

Runs HF thinker with experts_implementation="eager" (matches our F.linear loop)
and compares MoE output at layer 0 with torch backend.
If max_diff drops to ~0, the hypothesis is confirmed.
"""
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
    diff = (a.float() - b.float()).abs()
    cos = F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
    print(f"  {name}: cos={cos:.8f}  max_diff={diff.max().item():.4e}  identical={diff.max().item()==0}")


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
    for k in ("image_embeds", "deepstack_visual_embeds", "image_grid_thw", "audio_embeds"):
        if k in enc_out: kw[k] = enc_out[k]
    fam = hf_inputs.get("feature_attention_mask")
    if fam is not None: kw["feature_attention_mask"] = fam
    return kw


@torch.no_grad()
def run_hf(model_path, hf_inputs, enc_out, device, dtype, experts_impl):
    from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
    print(f"[HF experts_implementation={experts_impl}] Loading ...")
    model = Qwen3OmniSplitThinker(model_path, device=device, dtype=dtype)

    # Override experts implementation on the config
    hf_model = model.thinker
    actual_impl = getattr(hf_model.config, "_experts_implementation", "unknown")
    print(f"  Default _experts_implementation = {actual_impl}")
    if experts_impl is not None:
        hf_model.config._experts_implementation = experts_impl
        actual_impl = hf_model.config._experts_implementation
        print(f"  Overridden to: {actual_impl}")

    layer = hf_model.model.layers[0]
    cap = {}

    def moe_pre(module, args, kwargs=None):
        cap["moe_input"] = args[0].detach().cpu()
    layer.mlp.register_forward_pre_hook(moe_pre)

    def moe_post(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        cap["moe_output"] = o.detach().cpu()
    layer.mlp.register_forward_hook(moe_post)

    def layer_post(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        cap["layer_output"] = o.detach().cpu()
    layer.register_forward_hook(layer_post)

    ids = hf_inputs["input_ids"]
    mask = hf_inputs.get("attention_mask", torch.ones_like(ids))
    kw = _kw(hf_inputs, enc_out)
    print(f"[HF experts_implementation={experts_impl}] Forward ...")
    model(input_ids=ids, attention_mask=mask, **kw)
    del model; gc.collect(); torch.cuda.empty_cache()
    return cap


@torch.no_grad()
def run_torch(model_path, hf_inputs, enc_out, device, dtype):
    from sglang_omni.models.qwen3_omni.components.torch_thinker import Qwen3OmniTorchThinker
    print("[Torch] Loading ...")
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)
    layer = model.thinker.layers[0]
    cap = {}

    def moe_pre(module, args, kwargs=None):
        cap["moe_input"] = args[0].detach().cpu()
    layer.mlp.register_forward_pre_hook(moe_pre)

    def moe_post(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        cap["moe_output"] = o.detach().cpu()
    layer.mlp.register_forward_hook(moe_post)

    def layer_post(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        cap["layer_output"] = o.detach().cpu()
    layer.register_forward_hook(layer_post)

    ids = hf_inputs["input_ids"]
    mask = hf_inputs.get("attention_mask", torch.ones_like(ids))
    kw = _kw(hf_inputs, enc_out)
    print("[Torch] Forward ...")
    model(input_ids=ids, attention_mask=mask, **kw)
    del model; gc.collect(); torch.cuda.empty_cache()
    return cap


def _run_and_save(kind, mp, hf_inputs, enc_out, device, dtype, path, experts_impl=None):
    """Run one model in a subprocess to avoid OOM."""
    import pickle
    if kind == "hf":
        cap = run_hf(mp, hf_inputs, enc_out, device, dtype, experts_impl)
    else:
        cap = run_torch(mp, hf_inputs, enc_out, device, dtype)
    torch.save(cap, path)
    print(f"  Saved to {path}")


def main():
    import subprocess, sys, tempfile, os
    args = parse_args()
    mp = str(resolve_model_path(args.model_path))
    hf_inputs = preprocess(mp, args.image_path, args.audio_path)
    enc_out = run_encoders(mp, hf_inputs, args.device, args.dtype)

    # Save preprocessed data for subprocess use
    tmp = tempfile.mkdtemp()
    prep_path = os.path.join(tmp, "prep.pt")
    torch.save({"hf_inputs": hf_inputs, "enc_out": enc_out, "mp": mp}, prep_path)

    hf_out = os.path.join(tmp, "hf_eager.pt")
    torch_out = os.path.join(tmp, "torch.pt")

    # Run each model as a subprocess to avoid OOM
    device = args.device
    script = f"""
import torch, gc, sys
sys.path.insert(0, '.')
prep = torch.load('{prep_path}', weights_only=False)
mp, hf_inputs, enc_out = prep['mp'], prep['hf_inputs'], prep['enc_out']
"""
    hf_script = script + f"""
from scripts.verify_grouped_mm_hypothesis import run_hf
cap = run_hf(mp, hf_inputs, enc_out, '{device}', '{args.dtype}', 'eager')
torch.save(cap, '{hf_out}')
print('HF eager saved.')
"""
    torch_script = script + f"""
from scripts.verify_grouped_mm_hypothesis import run_torch
cap = run_torch(mp, hf_inputs, enc_out, '{device}', '{args.dtype}')
torch.save(cap, '{torch_out}')
print('Torch saved.')
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_dev:
        env["CUDA_VISIBLE_DEVICES"] = cuda_dev

    print("=== Running HF (eager) in subprocess ===")
    r = subprocess.run([sys.executable, "-c", hf_script], env=env, capture_output=False)
    if r.returncode != 0:
        print("HF subprocess failed!"); return

    print("\n=== Running Torch in subprocess ===")
    r = subprocess.run([sys.executable, "-c", torch_script], env=env, capture_output=False)
    if r.returncode != 0:
        print("Torch subprocess failed!"); return

    # Load and compare
    hf_eager = torch.load(hf_out, weights_only=False)
    torch_cap = torch.load(torch_out, weights_only=False)

    print(f"\n{'='*70}")
    print("HYPOTHESIS TEST: grouped_mm vs F.linear causes MoE diff")
    print(f"{'='*70}")

    print("\n--- HF(eager=F.linear loop) vs Torch(F.linear loop) ---")
    for k in ["moe_input", "moe_output", "layer_output"]:
        if k in hf_eager and k in torch_cap:
            cmp(k, hf_eager[k], torch_cap[k])

    print("\nConclusion:")
    moe_diff_eager = (hf_eager["moe_output"].float() - torch_cap["moe_output"].float()).abs().max().item()
    print(f"  Prior: HF(grouped_mm) vs Torch: moe_output max_diff = 7.8e-3")
    print(f"  Now:   HF(eager)      vs Torch: moe_output max_diff = {moe_diff_eager:.4e}")
    if moe_diff_eager < 1e-4:
        print("  >>> CONFIRMED: grouped_mm vs F.linear is the cause of 7.8e-3 diff")
        print("  >>> FIX: Switch torch backend to use grouped_mm")
    elif moe_diff_eager < 7.8e-3:
        print(f"  >>> PARTIAL: eager reduces diff by {7.8e-3/max(moe_diff_eager,1e-10):.1f}x")
    else:
        print("  >>> REJECTED: diff persists even with eager, another cause")

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
