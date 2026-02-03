# SPDX-License-Identifier: Apache-2.0
"""Compare prefill logits and first few decode steps between HF and Torch backends."""
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
    p.add_argument("--decode-steps", type=int, default=5)
    return p.parse_args()


def cmp(name, a, b):
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH a={a.shape} b={b.shape}")
        return
    diff = (a.float() - b.float()).abs()
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
    kw = {}
    for k in ("image_embeds", "deepstack_visual_embeds", "image_grid_thw", "audio_embeds"):
        if k in enc_out: kw[k] = enc_out[k]
    fam = hf_inputs.get("feature_attention_mask")
    if fam is not None: kw["feature_attention_mask"] = fam
    return kw


@torch.no_grad()
def run_hf_decode(model_path, hf_inputs, enc_out, device, dtype, num_decode_steps):
    from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
    print("[HF] Loading ...")
    model = Qwen3OmniSplitThinker(model_path, device=device, dtype=dtype)
    ids = hf_inputs["input_ids"]
    mask = hf_inputs.get("attention_mask", torch.ones_like(ids))
    kw = _kw(hf_inputs, enc_out)

    # Prefill
    print("[HF] Prefill ...")
    out = model(input_ids=ids, attention_mask=mask, use_cache=True, **kw)
    prefill_logits = out.logits[:, -1:, :].cpu()
    past_kv = out.past_key_values
    results = {"prefill_logits": prefill_logits}
    token = prefill_logits.argmax(dim=-1).item()
    results["token_0"] = token
    print(f"  prefill token: {token}")

    # Decode steps
    seq_len = ids.shape[1]
    for step in range(num_decode_steps):
        new_ids = torch.tensor([[token]], device=ids.device)
        mask = torch.cat([mask, torch.ones(1, 1, device=mask.device, dtype=mask.dtype)], dim=1)
        cache_pos = torch.tensor([seq_len + step], dtype=torch.long, device=ids.device)
        out = model(
            input_ids=new_ids, attention_mask=mask,
            past_key_values=past_kv, use_cache=True,
            cache_position=cache_pos,
        )
        step_logits = out.logits[:, -1:, :].cpu()
        past_kv = out.past_key_values
        token = step_logits.argmax(dim=-1).item()
        results[f"decode_{step}_logits"] = step_logits
        results[f"token_{step+1}"] = token
        print(f"  decode step {step}: token={token}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


@torch.no_grad()
def run_torch_decode(model_path, hf_inputs, enc_out, device, dtype, num_decode_steps):
    from sglang_omni.models.qwen3_omni.components.torch_thinker import Qwen3OmniTorchThinker
    print("[Torch] Loading ...")
    model = Qwen3OmniTorchThinker(model_path, device=device, dtype=dtype)
    ids = hf_inputs["input_ids"]
    mask = hf_inputs.get("attention_mask", torch.ones_like(ids))
    kw = _kw(hf_inputs, enc_out)

    # Prefill
    print("[Torch] Prefill ...")
    out = model(input_ids=ids, attention_mask=mask, use_cache=True, **kw)
    prefill_logits = out.logits[:, -1:, :].cpu()
    past_kv = out.past_key_values
    results = {"prefill_logits": prefill_logits}
    token = prefill_logits.argmax(dim=-1).item()
    results["token_0"] = token
    print(f"  prefill token: {token}")

    # Decode steps
    seq_len = ids.shape[1]
    for step in range(num_decode_steps):
        new_ids = torch.tensor([[token]], device=ids.device)
        mask = torch.cat([mask, torch.ones(1, 1, device=mask.device, dtype=mask.dtype)], dim=1)
        cache_pos = torch.tensor([seq_len + step], dtype=torch.long, device=ids.device)
        out = model(
            input_ids=new_ids, attention_mask=mask,
            past_key_values=past_kv, use_cache=True,
            cache_position=cache_pos,
        )
        step_logits = out.logits[:, -1:, :].cpu()
        past_kv = out.past_key_values
        token = step_logits.argmax(dim=-1).item()
        results[f"decode_{step}_logits"] = step_logits
        results[f"token_{step+1}"] = token
        print(f"  decode step {step}: token={token}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    mp = str(resolve_model_path(args.model_path))
    hf_inputs = preprocess(mp, args.image_path, args.audio_path)
    enc_out = run_encoders(mp, hf_inputs, args.device, args.dtype)

    hf_res = run_hf_decode(mp, hf_inputs, enc_out, args.device, args.dtype, args.decode_steps)
    torch_res = run_torch_decode(mp, hf_inputs, enc_out, args.device, args.dtype, args.decode_steps)

    print(f"\n{'='*70}")
    print("Logit comparison: HF vs Torch")
    print(f"{'='*70}")

    cmp("prefill_logits", hf_res["prefill_logits"], torch_res["prefill_logits"])
    print(f"  prefill token: HF={hf_res['token_0']}  Torch={torch_res['token_0']}  match={hf_res['token_0']==torch_res['token_0']}")

    for step in range(args.decode_steps):
        k = f"decode_{step}_logits"
        if k in hf_res and k in torch_res:
            cmp(k, hf_res[k], torch_res[k])
        tk = f"token_{step+1}"
        if tk in hf_res and tk in torch_res:
            print(f"  step {step} token: HF={hf_res[tk]}  Torch={torch_res[tk]}  match={hf_res[tk]==torch_res[tk]}")


if __name__ == "__main__":
    main()
