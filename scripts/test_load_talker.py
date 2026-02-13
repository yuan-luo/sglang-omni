#!/usr/bin/env python3
"""Minimal test script for Talker weight loading."""

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def init_sglang_minimal():
    """Initialize minimal SGLang distributed environment (single process)."""
    # Set environment variables for single-GPU setup
    import os
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    # Initialize torch distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


def load_config(checkpoint_path: Path):
    """Load model config from checkpoint."""
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    
    # Import config class
    from sglang_omni.config.qwen3_omni import Qwen3OmniMoeTalkerConfig
    
    # Extract talker config
    if "talker" not in config_dict:
        raise KeyError("'talker' key not found in config.json")
    
    talker_config_dict = config_dict["talker"]
    return Qwen3OmniMoeTalkerConfig(**talker_config_dict)


def iter_talker_weights(checkpoint_path: Path):
    """Iterate over talker weights from safetensors shards."""
    from safetensors import safe_open
    
    # Load weight index
    index_file = checkpoint_path / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Group weights by shard
    shards = {}
    for key, shard_name in weight_map.items():
        if key.startswith("talker."):
            shards.setdefault(shard_name, []).append(key)
    
    # Yield weights from each shard
    for shard_name, keys in shards.items():
        shard_path = checkpoint_path / shard_name
        print(f"Loading shard: {shard_path.name} ({len(keys)} talker params)")
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                yield key, f.get_tensor(key)


def main():
    # Checkpoint path
    checkpoint_path = Path("/home/menyu/workspace/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct")
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Talker Weight Loading Test")
    print("=" * 80)
    
    # 1. Initialize SGLang
    print("\n[1/5] Initializing SGLang distributed environment...")
    try:
        init_sglang_minimal()
        print("✓ SGLang initialized")
    except Exception as e:
        print(f"✗ Failed to initialize SGLang: {e}")
        sys.exit(1)
    
    # 2. Load config
    print("\n[2/5] Loading model config...")
    try:
        config = load_config(checkpoint_path)
        print(f"✓ Config loaded")
        print(f"  - Text layers: {config.text_config.num_hidden_layers}")
        print(f"  - CodePredictor layers: {config.code_predictor_config.num_hidden_layers}")
        print(f"  - Num code groups: {config.num_code_groups}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 3. Instantiate model
    print("\n[3/5] Instantiating Talker model...")
    try:
        from sglang_omni.models.qwen3_omni.talker import Qwen3OmniTalker
        
        model = Qwen3OmniTalker(
            config=config,
            quant_config=None,
            prefix="",
        )
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        print(f"✓ Model instantiated")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {total_params:,}")
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Load weights
    print("\n[4/5] Loading weights...")
    try:
        weights = list(iter_talker_weights(checkpoint_path))
        print(f"✓ Found {len(weights)} talker weights")
        
        print("  Loading into model...")
        model.load_weights(weights)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 5. Verify a parameter
    print("\n[5/5] Verifying weights...")
    try:
        # Check codec_head weight
        codec_head_weight = model.codec_head.weight
        print(f"✓ codec_head.weight shape: {codec_head_weight.shape}")
        print(f"  - dtype: {codec_head_weight.dtype}")
        print(f"  - device: {codec_head_weight.device}")
        print(f"  - mean: {codec_head_weight.mean().item():.6f}")
        print(f"  - std: {codec_head_weight.std().item():.6f}")
        
        # Check against raw checkpoint tensor
        print("\n  Comparing with checkpoint...")
        checkpoint_codec_head = None
        for key, tensor in iter_talker_weights(checkpoint_path):
            if key == "talker.codec_head.weight":
                checkpoint_codec_head = tensor
                break
        
        if checkpoint_codec_head is not None:
            # Move to same device for comparison
            checkpoint_codec_head = checkpoint_codec_head.to(codec_head_weight.device, codec_head_weight.dtype)
            
            if torch.allclose(codec_head_weight, checkpoint_codec_head, atol=1e-5):
                print("✓ codec_head.weight matches checkpoint!")
            else:
                max_diff = (codec_head_weight - checkpoint_codec_head).abs().max().item()
                print(f"✗ codec_head.weight differs from checkpoint (max diff: {max_diff:.6e})")
        else:
            print("  Warning: Could not find talker.codec_head.weight in checkpoint for comparison")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
