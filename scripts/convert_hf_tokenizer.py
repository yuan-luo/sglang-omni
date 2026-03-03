#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a HuggingFace tokenizer.json to tiktoken format for fish_speech.

Usage:
    python scripts/convert_hf_tokenizer.py /path/to/model_dir

This reads ``tokenizer.json`` from the model directory and generates:
  - ``tokenizer.tiktoken``  (BPE vocabulary in tiktoken format)
  - ``special_tokens.json`` (special token name → ID mapping)

These two files are required by ``fish_speech.tokenizer.FishTokenizer``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-to-unicode mapping (256 bytes → 256 printable chars)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def convert(model_dir: Path) -> None:
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        print(f"Error: {tokenizer_json} not found", file=sys.stderr)
        sys.exit(1)

    with open(tokenizer_json, encoding="utf-8") as f:
        tok = json.load(f)

    # --- BPE vocabulary ---
    hf_vocab: dict[str, int] = tok["model"]["vocab"]
    unicode_to_byte = {v: k for k, v in _bytes_to_unicode().items()}

    lines: list[str] = []
    for token_str, rank in sorted(hf_vocab.items(), key=lambda x: x[1]):
        raw = bytes(unicode_to_byte[c] for c in token_str)
        b64 = base64.b64encode(raw).decode("ascii")
        lines.append(f"{b64} {rank}")

    tiktoken_path = model_dir / "tokenizer.tiktoken"
    tiktoken_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} BPE tokens to {tiktoken_path}")

    # --- Special tokens ---
    added_tokens = tok.get("added_tokens", [])
    added_tokens.sort(key=lambda x: x["id"])
    special_tokens = {t["content"]: t["id"] for t in added_tokens}

    special_tokens_path = model_dir / "special_tokens.json"
    with open(special_tokens_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(special_tokens)} special tokens to {special_tokens_path}")

    # --- Verify ---
    semantic = {k: v for k, v in special_tokens.items() if "semantic" in k}
    if semantic:
        ids = sorted(semantic.values())
        print(f"Semantic tokens: {len(semantic)} (ID {ids[0]}..{ids[-1]})")

    expected_base = len(hf_vocab)
    for i, (k, v) in enumerate(special_tokens.items()):
        if expected_base + i != v:
            print(f"Warning: {k} expected ID {expected_base + i} but got {v}")
            break
    else:
        print("All special token IDs are sequential — OK")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace tokenizer.json to tiktoken format"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model directory containing tokenizer.json",
    )
    args = parser.parse_args()
    convert(args.model_dir)


if __name__ == "__main__":
    main()
