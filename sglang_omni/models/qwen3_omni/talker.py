"""SGLang-native Talker model for Qwen3-Omni.

Simplified implementation:
- Reuse Thinker's components where possible
- Only define Talker-specific parts (Shared Expert MoE)
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.config.qwen3_omni import (
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
)

# Reuse Thinker's components
from sglang_omni.models.qwen3_omni.thinker import (
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import tensor_model_parallel_all_reduce
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QuantizationConfig,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
    top_k_top_p_sampling_from_probs,
)
from sglang_omni.vendor.sglang.utils import make_layers

# Note (chenyang): It is said that these constants are from the HF checkpoint.

CODE_PREDICTOR_TOP_K = 50
CODE_PREDICTOR_TOP_P = 0.8


class ResizeMLP(nn.Module):
    """Simple Linear-SiLU-Linear projection (used for text/hidden projection)."""

    def __init__(
        self,
        in_size: int,
        intermediate_size: int,
        out_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_fc1 = ReplicatedLinear(
            in_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc1", prefix),
        )
        self.act = nn.SiLU()
        self.linear_fc2 = ReplicatedLinear(
            intermediate_size,
            out_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_fc2", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.linear_fc1(x)
        out = self.act(out)
        out, _ = self.linear_fc2(out)
        return out


# ---------------------------------------------------------------------------
# Talker-specific MLP (Shared Expert MoE)
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerDenseMLP(nn.Module):
    """Standard SwiGLU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3OmniMoeTalkerSharedExpertMLP(nn.Module):
    """Shared expert MLP with reduce_results=False for unified all-reduce."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=False,  # Don't all-reduce here; unified with routed experts
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3OmniMoeTalkerSparseMoeBlock(Qwen3OmniMoeThinkerTextSparseMoeBlock):
    """MoE block with Shared Expert (Talker-specific).

    Inherits from Thinker's MoE for routed experts (topk, experts, gate).
    Adds shared expert with gated output.

    All-reduce is unified: both routed and shared expert outputs stay as
    per-rank partial sums until combined, then a single all-reduce is applied.
    """

    def __init__(
        self,
        layer_id: int,
        config: Qwen3OmniMoeTalkerTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # Initialize parent (Thinker's MoE: topk, experts, gate)
        super().__init__(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Shared expert (reduce_results=False to avoid double all-reduce)
        self.shared_expert = Qwen3OmniMoeTalkerSharedExpertMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("shared_expert", prefix),
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=add_prefix("shared_expert_gate", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # --- Routed experts (no all-reduce yet) ---
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_output = self.experts(hidden_states, topk_output)

        # --- Shared expert (no all-reduce, reduce_results=False) ---
        shared_output = self.shared_expert(hidden_states)
        shared_gate, _ = self.shared_expert_gate(hidden_states)
        shared_output = shared_output * torch.sigmoid(shared_gate)

        # --- Combine then unified all-reduce ---
        final_hidden_states = routed_output + shared_output

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


# ---------------------------------------------------------------------------
# Talker DecoderLayer (minimal override of Thinker's)
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerDecoderLayer(Qwen3OmniMoeThinkerTextDecoderLayer):
    """Talker decoder layer: inherit from Thinker, only replace MLP with Shared Expert MoE."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # Call parent's __init__ (Thinker's DecoderLayer)
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )

        # Replace MLP with Talker's Shared Expert MoE
        self.mlp = Qwen3OmniMoeTalkerSparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )


# ---------------------------------------------------------------------------
# Talker Text Model
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerTextModel(nn.Module):
    """Talker's MoE text backbone (20-layer, with shared expert).

    Uses codec_embedding instead of embed_tokens.
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Codec embedding (standard nn.Embedding, not VocabParallel - vocab is small)
        self.codec_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Decoder layers
        alt_stream = torch.cuda.Stream()
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Qwen3OmniMoeTalkerDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        if input_embeds is None:
            hidden_states = self.codec_embedding(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


# ---------------------------------------------------------------------------
# Code Predictor (single class, matches HF checkpoint structure)
# ---------------------------------------------------------------------------


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary position embedding cos/sin tables."""
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_cached = freqs.cos()
    sin_cached = freqs.sin()
    return cos_cached, sin_cached


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to Q or K tensor."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _CodePredictorMLP(nn.Module):
    """SwiGLU MLP for the code predictor."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = ReplicatedLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ReplicatedLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        return self.down_proj(torch.nn.functional.silu(gate) * up)[0]


class _CausalSelfAttention(nn.Module):
    """Lightweight causal self-attention with RoPE for the code predictor."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1000000.0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.rope_theta = rope_theta

        self.qkv_proj = nn.Linear(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q, k, v = self._split_qkv(hidden_states, batch_size, seq_len)
        q, k = self._apply_rope(q, k, seq_len)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    def _split_qkv(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV projection into Q, K, V and reshape to head layout."""
        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = (
            qkv[..., :q_size]
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            qkv[..., q_size : q_size + kv_size]
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            qkv[..., q_size + kv_size :]
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        return q, k, v

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K."""
        cos, sin = _build_rope_cache(seq_len, self.head_dim, self.rope_theta, q.device)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        return q, k


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """Code predictor for generating RVQ codes."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        cp_config = config.code_predictor_config

        self.model = nn.Module()

        self.model.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(cp_config.vocab_size, cp_config.hidden_size)
                for _ in range(config.num_code_groups - 1)
            ]
        )

        self.model.layers = nn.ModuleList()
        for idx in range(cp_config.num_hidden_layers):
            layer = nn.Module()
            layer.self_attn = _CausalSelfAttention(
                hidden_size=cp_config.hidden_size,
                num_heads=cp_config.num_attention_heads,
                num_kv_heads=cp_config.num_key_value_heads,
                head_dim=cp_config.head_dim,
                rope_theta=cp_config.rope_theta,
                quant_config=quant_config,
                prefix=add_prefix(f"model.layers.{idx}.self_attn", prefix),
            )
            layer.mlp = _CodePredictorMLP(
                hidden_size=cp_config.hidden_size,
                intermediate_size=cp_config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix(f"model.layers.{idx}.mlp", prefix),
            )
            layer.input_layernorm = RMSNorm(
                cp_config.hidden_size, eps=cp_config.rms_norm_eps
            )
            layer.post_attention_layernorm = RMSNorm(
                cp_config.hidden_size, eps=cp_config.rms_norm_eps
            )
            self.model.layers.append(layer)

        self.model.norm = RMSNorm(cp_config.hidden_size, eps=cp_config.rms_norm_eps)

        self.lm_head = nn.ModuleList(
            [
                ReplicatedLinear(
                    cp_config.hidden_size,
                    cp_config.vocab_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix(f"lm_head.{i}", prefix),
                )
                for i in range(config.num_code_groups - 1)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through the code predictor."""
        has_batch_dim = inputs_embeds.dim() == 3
        if has_batch_dim:
            batch_size, seq_len, hidden_size = inputs_embeds.shape
            hidden_states = inputs_embeds.reshape(-1, hidden_size)
        else:
            hidden_states = inputs_embeds

        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            if has_batch_dim:
                hidden_states = hidden_states.view(batch_size, seq_len, -1)
            hidden_states = layer.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            if has_batch_dim:
                hidden_states = hidden_states.reshape(-1, hidden_size)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.model.norm(hidden_states)

        if has_batch_dim:
            hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)
        return hidden_states


class Qwen3OmniTalker(nn.Module):
    """Talker: Text-to-Audio generation model."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.text_projection = ResizeMLP(
            config.thinker_hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("text_projection", prefix),
        )
        self.hidden_projection = ResizeMLP(
            config.thinker_hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("hidden_projection", prefix),
        )

        self.model = Qwen3OmniMoeTalkerTextModel(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.codec_head = ReplicatedLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("codec_head", prefix),
        )
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictor(
            config,
            quant_config=quant_config,
            prefix=add_prefix("code_predictor", prefix),
        )
        self._cached_params_dict: dict[str, torch.nn.Parameter] | None = None

    def prepare_input_embeds(
        self,
        thinker_embeds: Optional[torch.Tensor] = None,
        thinker_hidden_states: Optional[torch.Tensor] = None,
        is_multimodal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project thinker outputs to talker's hidden dimension."""
        if thinker_hidden_states is None or is_multimodal_mask is None:
            return self.text_projection(thinker_embeds)
        if thinker_embeds is None:
            return self.hidden_projection(thinker_hidden_states)

        output = torch.empty(
            (*thinker_embeds.shape[:-1], self.config.text_config.hidden_size),
            device=thinker_embeds.device,
            dtype=thinker_embeds.dtype,
        )
        if is_multimodal_mask.any():
            output[is_multimodal_mask] = self.hidden_projection(
                thinker_hidden_states[is_multimodal_mask]
            )
        text_mask = ~is_multimodal_mask
        if text_mask.any():
            output[text_mask] = self.text_projection(thinker_embeds[text_mask])
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the talker MoE backbone."""
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute layer-0 codec logits."""
        logits, _ = self.codec_head(hidden_states)
        return logits

    def code_predictor_forward(
        self,
        layer0_codes: torch.Tensor,
        talker_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate residual RVQ codes (layers 1 to N-1) for each position."""
        batch_size, seq_len = layer0_codes.shape
        num_groups = self.config.num_code_groups
        all_codes_per_pos = []
        all_summed_per_pos = []

        for pos in range(seq_len):
            layer0_code = layer0_codes[:, pos : pos + 1]
            layer0_embed = self.model.codec_embedding(layer0_code)
            last_hidden = talker_hidden[:, pos : pos + 1, :]

            current_input = torch.cat([last_hidden, layer0_embed], dim=1)
            pos_codes = [layer0_code]

            for layer_idx in range(num_groups - 1):
                predictor_hidden = self.code_predictor(
                    inputs_embeds=current_input,
                    positions=torch.arange(
                        current_input.shape[1], device=current_input.device
                    ),
                )

                logits, _ = self.code_predictor.lm_head[layer_idx](
                    predictor_hidden[:, -1:, :]
                )
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                code = top_k_top_p_sampling_from_probs(
                    probs.float(),
                    top_k=CODE_PREDICTOR_TOP_K,
                    top_p=CODE_PREDICTOR_TOP_P,
                )
                if code.dim() == 1:
                    code = code.unsqueeze(1)
                code = code.long()
                pos_codes.append(code)

                new_embed = self.code_predictor.model.codec_embedding[layer_idx](code)
                current_input = torch.cat([current_input, new_embed], dim=1)

            all_codes_per_pos.append(torch.stack(pos_codes, dim=1))

            codec_embeds = current_input[:, 1:, :]
            pos_summed = codec_embeds.sum(dim=1, keepdim=True)
            all_summed_per_pos.append(pos_summed)

        result_codes = torch.cat(all_codes_per_pos, dim=2)
        summed_embeddings = torch.cat(all_summed_per_pos, dim=1)

        return result_codes, summed_embeddings

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load weights from safetensors files."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if self._cached_params_dict is None:
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict

        stacked_params = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.text_config.num_experts,
        )

        cp_qkv_buffer: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            if not name.startswith("talker."):
                continue
            name = name[len("talker.") :]

            is_cp_qkv = (
                "code_predictor.model.layers." in name
                and ".self_attn." in name
                and any(p in name for p in (".q_proj.", ".k_proj.", ".v_proj."))
            )
            if is_cp_qkv:
                qkv_key = name.replace(".q_proj.", ".qkv_proj.")
                qkv_key = qkv_key.replace(".k_proj.", ".qkv_proj.")
                qkv_key = qkv_key.replace(".v_proj.", ".qkv_proj.")
                buf = cp_qkv_buffer.setdefault(qkv_key, {})
                if ".q_proj." in name:
                    buf["q"] = loaded_weight
                elif ".k_proj." in name:
                    buf["k"] = loaded_weight
                elif ".v_proj." in name:
                    buf["v"] = loaded_weight
                if len(buf) == 3:
                    param = params_dict.get(qkv_key)
                    if param is not None:
                        fused = torch.cat([buf["q"], buf["k"], buf["v"]], dim=0)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, fused)
                    del cp_qkv_buffer[qkv_key]
                continue

            is_handled = False
            for param_name, weight_name, shard_id in stacked_params:
                if weight_name in name and "mlp.experts" not in name:
                    param = params_dict.get(name.replace(weight_name, param_name))
                    if param is not None:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight, shard_id)
                        is_handled = True
                        break
            if is_handled:
                continue

            for param_name, weight_name, expert_id, shard_id in expert_params:
                if weight_name in name:
                    mapped = name.replace(weight_name, param_name)
                    param = params_dict.get(mapped)
                    if param is not None:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        is_handled = True
                        break
            if is_handled:
                continue

            param = params_dict.get(name)
            if param is not None:
                default_weight_loader(param, loaded_weight)
