# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 AR backbone on SGLang V1 primitives.

This is the AR (autoregressive text) half of the HunyuanImage-3.0 model. It
shares the 80B backbone with the DiT side — DiT reads AR's per-layer K/V via
the KVExporter helper.

Architecture parameters (from `config.json` of the Distil checkpoint):
  hidden_size=4096, intermediate_size=3072, num_hidden_layers=32,
  num_attention_heads=32, num_key_value_heads=8 (GQA 4:1),
  head_dim=128, vocab_size=133120, rms_norm_eps=1e-5,
  max_position_embeddings=22800, rope_theta=10000,
  rope_scaling={'alpha':1.0,'beta_fast':32,'beta_slow':1,'factor':1.0,'mscale':1.0,'mscale_all_dim':1.0,'type':'custom'},
  num_experts=64, moe_topk=[8]*32, num_shared_expert=[1]*32,
  moe_intermediate_size=[3072]*32, use_mixed_mlp_moe=True.

Weight-name remap (HF checkpoint conventions vs sglang naming):
  - QKV is already fused in the HF checkpoint as `self_attn.qkv_proj.weight`
    → maps 1:1 to sglang's QKVParallelLinear.qkv_proj.weight.
  - QK-norm uses HF names `self_attn.query_layernorm` / `self_attn.key_layernorm`
    → sglang convention `self_attn.q_norm` / `self_attn.k_norm`.
  - MoE router uses HF name `mlp.gate.wg.weight`
    → strip `.wg` → `mlp.gate.weight`.
  - MoE experts have gate+up fused already as
    `mlp.experts.N.gate_and_up_proj.weight`. Need a custom loader that splits
    the fused tensor into FusedMoE's w13_weight slots — distinct from the
    separate-gate/up case handled by existing helpers in this codebase.
  - Shared MLP keys: `mlp.shared_mlp.gate_and_up_proj.weight` /
    `mlp.shared_mlp.down_proj.weight`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Optional, Tuple

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from torch import nn

# Pre-compiled patterns for MoE expert weight remap.
_EXPERT_GATE_UP_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_and_up_proj\.weight$"
)
_EXPERT_DOWN_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$"
)


# ---------------------------------------------------------------------------
# Attention with QKV-fused + QK-norm + RoPE + RadixAttention
# ---------------------------------------------------------------------------
class HunyuanImage3Attention(nn.Module):
    """Self-attention block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: dict[str, Any],
        rms_norm_eps: float,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_heads_per_tp = num_heads // tp_size
        self.num_kv_heads_per_tp = max(1, num_kv_heads // tp_size)
        self.q_size = self.num_heads_per_tp * head_dim
        self.kv_size = self.num_kv_heads_per_tp * head_dim
        self.scaling = head_dim**-0.5

        # QKV: HF checkpoint stores fused. sglang's QKVParallelLinear accepts
        # the fused form natively.
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK-norm — HF names: query_layernorm / key_layernorm.
        # We use sglang convention q_norm / k_norm; load_weights remaps.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        # Custom RoPE handling. HunyuanImage-3's config sets
        # `rope_scaling.type='custom'` with alpha/factor/mscale/mscale_all_dim
        # fields. If all scaling factors are 1.0, the "custom" scaling is a
        # no-op and we can fall through to standard RoPE; sglang's get_rope
        # may not recognize 'custom' as a type.
        passable_rope_scaling = rope_scaling
        if isinstance(rope_scaling, dict) and rope_scaling.get("type") == "custom":
            identity_factors = all(
                float(rope_scaling.get(k, 1.0)) == 1.0
                for k in ("alpha", "factor", "mscale", "mscale_all_dim")
            )
            if identity_factors:
                passable_rope_scaling = None
            else:
                raise NotImplementedError(
                    "HunyuanImage3 custom rope_scaling with non-identity factors "
                    f"({rope_scaling}) requires yarn implementation; see "
                    "HunyuanImage3RotaryEmbedding."
                )
        self.rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=passable_rope_scaling,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads_per_tp,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads_per_tp,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # QK-norm — apply per-head (reshape, norm, flatten).
        q = self._apply_qk_norm(q, self.q_norm, self.num_heads_per_tp)
        k = self._apply_qk_norm(k, self.k_norm, self.num_kv_heads_per_tp)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch=forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    @staticmethod
    def _apply_qk_norm(x: torch.Tensor, norm: RMSNorm, num_heads: int) -> torch.Tensor:
        # x: [..., num_heads*head_dim] -> [..., num_heads, head_dim] -> norm -> flatten
        head_dim = x.shape[-1] // num_heads
        shape = x.shape
        x = x.view(*shape[:-1], num_heads, head_dim)
        x = norm(x)
        return x.view(*shape)


# ---------------------------------------------------------------------------
# Mixed MLP + MoE block (HunyuanImage-3 uses both a shared MLP and a sparse
# MoE per layer for the 80B configuration).
# ---------------------------------------------------------------------------
class HunyuanImage3MoEBlock(nn.Module):
    """Sparse MoE block with shared-expert mix.

    For HunyuanImage-3 every layer has:
      - 64 routed experts (top_k=8)
      - 1 shared expert (always activated)
      - Final output = shared_out + routed_out
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_expert: int,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router — replicated, never quantized (routing accuracy matters).
        # HF name is `gate.wg.weight` (extra `.wg.` level); load_weights strips it.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.topk = TopK(
            top_k=top_k,
            renormalize=True,
            use_grouped_topk=False,
            layer_id=layer_id,
        )

        # Routed experts via FusedMoE. HF checkpoint has gate+up fused per
        # expert as `gate_and_up_proj`; the custom loader in load_weights
        # splits the fused tensor across FusedMoE's w1/w3 shards.
        #
        # reduce_results=True so the expert output is all_reduced across TP
        # ranks before we combine it with the replicated shared_mlp output —
        # otherwise the shared contribution would be counted N times.
        moe_cls = get_moe_impl_class(quant_config)
        self.experts = moe_cls(
            num_experts=num_experts,
            top_k=top_k,
            layer_id=layer_id,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            quant_config=quant_config,
            reduce_results=True,
            routing_method_type=RoutingMethodType.Renormalize,
            prefix=f"{prefix}.experts",
        )

        # Shared MLP (1 per layer for HunyuanImage-3).
        # HF stores its gate+up fused as `shared_mlp.gate_and_up_proj`.
        if num_shared_expert > 0:
            self.shared_mlp = HunyuanImage3SharedMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size * num_shared_expert,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp",
            )
        else:
            self.shared_mlp = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # FusedMoE with inplace=True can stomp `hidden_states` — keep a copy
        # for the shared MLP path when one is present.
        identity = (
            hidden_states.clone() if self.shared_mlp is not None else hidden_states
        )
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed = self.experts(hidden_states, topk_output)
        if self.shared_mlp is not None:
            return routed + self.shared_mlp(identity)
        return routed


class HunyuanImage3SharedMLP(nn.Module):
    """Shared MLP for one layer (gate_and_up_proj fused in HF)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # HF: shared_mlp.gate_and_up_proj.weight has shape [2*intermediate, hidden].
        # Stored as ReplicatedLinear since the shared MLP is replicated, not TP.
        self.gate_up_proj = ReplicatedLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Decoder layer = Attention + MoE block with pre-norm.
# ---------------------------------------------------------------------------
class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HunyuanImage3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # MoE config differs per layer; pull from config arrays.
        moe_top_k = config.moe_topk[layer_id]
        num_shared = config.num_shared_expert[layer_id]
        moe_intermediate = config.moe_intermediate_size[layer_id]
        self.mlp = HunyuanImage3MoEBlock(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=moe_intermediate,
            num_experts=config.num_experts,
            top_k=moe_top_k,
            num_shared_expert=num_shared,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Stacked decoder = the AR backbone proper.
# ---------------------------------------------------------------------------
class HunyuanImage3Model(nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "model",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            [
                HunyuanImage3DecoderLayer(
                    config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, forward_batch
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level: ForCausalMM (matches HF `architectures` field name for registry parity).
# ---------------------------------------------------------------------------
class HunyuanImage3ForConditionalGeneration(nn.Module):
    """SGLang model class for HunyuanImage-3 AR backbone.

    The HF `config.architectures` field reads `HunyuanImage3ForCausalMM` —
    registered as an alias to this class in `registration.py`.
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = HunyuanImage3Model(
            config, quant_config=quant_config, prefix="model"
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix="lm_head",
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        """Embedding module — entry point for multimodal injection paths."""
        return self.model.get_input_embeddings()

    # ---------------- Weight loading (HF→sglang remap) ----------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load HF safetensor weights with name remap.

        Skips non-AR keys (vae.*, vision_model.*, patch_embed.*, time_embed*,
        timestep_emb*, guidance_emb.*, final_layer.*) — those are owned by
        DiT-side modules and load there.

        Remaps:
          - model.layers.{i}.self_attn.qkv_proj.weight                → as-is (sglang QKVParallelLinear accepts fused)
          - model.layers.{i}.self_attn.query_layernorm.weight         → self_attn.q_norm.weight
          - model.layers.{i}.self_attn.key_layernorm.weight           → self_attn.k_norm.weight
          - model.layers.{i}.mlp.gate.wg.weight                       → mlp.gate.weight (strip .wg.)
          - model.layers.{i}.mlp.experts.{N}.gate_and_up_proj.weight  → split into w13_weight (gate, up shards)
          - model.layers.{i}.mlp.experts.{N}.down_proj.weight         → w2_weight
          - model.layers.{i}.mlp.shared_mlp.gate_and_up_proj.weight   → shared_mlp.gate_up_proj.weight (as-is, no split)
          - model.layers.{i}.mlp.shared_mlp.down_proj.weight          → shared_mlp.down_proj.weight (as-is)
        """
        # AR-only top-level prefixes — everything else is DiT/VAE/vision and is loaded elsewhere.
        _SKIP_ROOTS = (
            "vae.",
            "vision_model.",
            "vision_aligner.",
            "patch_embed.",
            "time_embed.",
            "time_embed_2.",
            "timestep_emb.",
            "timestep_r_emb.",
            "guidance_emb.",
            "final_layer.",
        )

        params_dict = dict(self.named_parameters())
        loaded = set()

        for hf_name, loaded_weight in weights:
            if any(hf_name.startswith(p) for p in _SKIP_ROOTS):
                continue

            name = hf_name

            # QK-norm remap.
            name = name.replace(".self_attn.query_layernorm.", ".self_attn.q_norm.")
            name = name.replace(".self_attn.key_layernorm.", ".self_attn.k_norm.")

            # Router remap.
            name = name.replace(".mlp.gate.wg.", ".mlp.gate.")

            # MoE expert weights — gate_and_up_proj is gate+up fused as one [2*I, H] tensor.
            # ----- MoE experts: gate+up fused (gate_and_up_proj) -----
            m = _EXPERT_GATE_UP_RE.match(name)
            if m is not None:
                layer_id = int(m.group(1))
                expert_id = int(m.group(2))
                fused_param_name = f"model.layers.{layer_id}.mlp.experts.w13_weight"
                if fused_param_name not in params_dict:
                    raise KeyError(
                        f"Missing FusedMoE param {fused_param_name!r} "
                        f"while loading HF key {hf_name!r}"
                    )
                param = params_dict[fused_param_name]
                weight_loader = param.weight_loader

                # HF layout: [2*intermediate, hidden] = concat(gate, up).
                # FusedMoE shard_id="w1" (gate) and "w3" (up) load separately.
                intermediate = loaded_weight.shape[0] // 2
                gate_w = loaded_weight[:intermediate, :].contiguous()
                up_w = loaded_weight[intermediate:, :].contiguous()
                weight_loader(
                    param,
                    gate_w,
                    weight_name=f"experts.{expert_id}.gate_proj",
                    shard_id="w1",
                    expert_id=expert_id,
                )
                weight_loader(
                    param,
                    up_w,
                    weight_name=f"experts.{expert_id}.up_proj",
                    shard_id="w3",
                    expert_id=expert_id,
                )
                loaded.add(fused_param_name)
                continue

            # ----- MoE experts: down_proj -----
            m = _EXPERT_DOWN_RE.match(name)
            if m is not None:
                layer_id = int(m.group(1))
                expert_id = int(m.group(2))
                fused_param_name = f"model.layers.{layer_id}.mlp.experts.w2_weight"
                if fused_param_name not in params_dict:
                    raise KeyError(
                        f"Missing FusedMoE param {fused_param_name!r} "
                        f"while loading HF key {hf_name!r}"
                    )
                param = params_dict[fused_param_name]
                param.weight_loader(
                    param,
                    loaded_weight,
                    weight_name=f"experts.{expert_id}.down_proj",
                    shard_id="w2",
                    expert_id=expert_id,
                )
                loaded.add(fused_param_name)
                continue

            # Shared MLP is replicated (not TP) — direct load.
            # Standard layers (qkv_proj, o_proj, embeds, norm) — direct load.
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
            else:
                # Bubble up unknown keys for debugging — fail loud, not silent.
                raise KeyError(
                    f"HunyuanImage3ForConditionalGeneration.load_weights: "
                    f"unknown HF key {hf_name!r} (remapped to {name!r})"
                )

        # Sanity: report missing params (expected to be empty for a clean
        # HunyuanImage-3 Instruct/Distil checkpoint after the remap above).
        missing = set(params_dict.keys()) - loaded
        if missing:
            raise RuntimeError(
                f"HunyuanImage3 load_weights: {len(missing)} params not loaded. "
                f"Sample: {sorted(missing)[:8]}. Most likely cause: HF key "
                f"naming does not match the remap table — extend the regex "
                f"patterns / remap lines above."
            )


# Public exports.
__all__ = [
    "HunyuanImage3Attention",
    "HunyuanImage3DecoderLayer",
    "HunyuanImage3ForConditionalGeneration",
    "HunyuanImage3MoEBlock",
    "HunyuanImage3Model",
    "HunyuanImage3SharedMLP",
]
