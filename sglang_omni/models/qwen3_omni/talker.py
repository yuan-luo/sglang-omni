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
    QKVParallelLinear,
    QuantizationConfig,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    get_rope,
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


class _CodePredictorAttention(nn.Module):
    """Causal self-attention for the Code Predictor (eager mode, no KV cache).

    Uses QKVParallelLinear for weight-loading compatibility (stacked q/k/v),
    get_rope() for RoPE, and plain torch SDPA for attention (max seq_len ~17).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. hidden_states: [B, seq_len, hidden]. positions: [seq_len]."""
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection: flatten to [B*seq_len, hidden] for QKVParallelLinear
        qkv, _ = self.qkv_proj(hidden_states.reshape(-1, hidden_states.shape[-1]))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # RoPE: expand positions to match flattened batch
        flat_positions = positions.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        q, k = self.rotary_emb.forward_native(flat_positions, q, k)

        # Reshape for SDPA: [B, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # GQA expansion
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        # [B, num_heads, seq_len, head_dim] -> [B*seq_len, num_heads * head_dim]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size * seq_len, -1)
        )
        output, _ = self.o_proj(attn_output)
        return output.view(batch_size, seq_len, -1)


class _CodePredictorDecoderLayer(nn.Module):
    """Single transformer layer for the Code Predictor.

    Pre-norm architecture with RMSNorm, _CodePredictorAttention, and DenseMLP.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = _CodePredictorAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3OmniMoeTalkerDenseMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)


class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """Code predictor for generating RVQ codes.

    A small 5-layer dense transformer that runs in eager mode (no KV cache,
    max seq_len ~17). Uses proper SGLang layers for weight loading.
    """

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

        self.model.layers = nn.ModuleList(
            [
                _CodePredictorDecoderLayer(
                    hidden_size=cp_config.hidden_size,
                    intermediate_size=cp_config.intermediate_size,
                    num_attention_heads=cp_config.num_attention_heads,
                    num_key_value_heads=cp_config.num_key_value_heads,
                    head_dim=cp_config.head_dim,
                    max_position_embeddings=cp_config.max_position_embeddings,
                    rope_theta=cp_config.rope_theta,
                    rms_norm_eps=cp_config.rms_norm_eps,
                    attention_bias=cp_config.attention_bias,
                    quant_config=quant_config,
                    prefix=add_prefix(f"model.layers.{idx}", prefix),
                )
                for idx in range(cp_config.num_hidden_layers)
            ]
        )

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
        """Forward through the code predictor.

        Args:
            inputs_embeds: [B, seq_len, hidden] or [num_tokens, hidden].
            positions: 1D position indices of length seq_len.

        Returns:
            Hidden states with the same shape as inputs_embeds.
        """
        has_batch_dim = inputs_embeds.dim() == 3
        if has_batch_dim:
            batch_size, seq_len, hidden_size = inputs_embeds.shape
        else:
            # Treat flat [num_tokens, hidden] as single-batch [1, num_tokens, hidden]
            batch_size = 1
            seq_len, hidden_size = inputs_embeds.shape
            inputs_embeds = inputs_embeds.unsqueeze(0)

        hidden_states = inputs_embeds.reshape(-1, hidden_size)

        for layer in self.model.layers:
            # Pre-norm attention
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = hidden_states.view(batch_size, seq_len, -1)
            hidden_states = layer.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states = hidden_states.reshape(-1, hidden_size)
            hidden_states = residual + hidden_states

            # Pre-norm MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.model.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        if not has_batch_dim:
            hidden_states = hidden_states.squeeze(0)
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

        for name, loaded_weight in weights:
            if not name.startswith("talker."):
                continue
            name = name[len("talker.") :]

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


class Qwen3OmniTalkerForCausalLM(nn.Module):
    """SGLang-compatible wrapper for the Talker model.

    Adapts Qwen3OmniTalker to the interface expected by SGLang's ModelRunner:
    - forward(input_ids, positions, forward_batch) → LogitsProcessorOutput
    - load_weights(weights) → loads from checkpoint

    During prefill (extend), pre-computed thinker embeddings are injected
    via _prefill_embeds. During decode, codec_embedding is used normally.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        **kwargs,
    ):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        super().__init__()
        talker_cfg = getattr(config, "talker_config", None)
        if talker_cfg is None:
            talker_cfg = {}
        if isinstance(talker_cfg, dict):
            self.talker_config = Qwen3OmniMoeTalkerConfig(**talker_cfg)
        else:
            self.talker_config = talker_cfg

        self.talker = Qwen3OmniTalker(
            self.talker_config,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(self.talker_config.text_config)

        # Pre-computed thinker embeddings for prefill injection.
        # Set by TalkerSGLangModelRunner before prefill forward passes.
        self._prefill_embeds: Optional[torch.Tensor] = None

    def set_prefill_embeds(self, embeds: torch.Tensor) -> None:
        self._prefill_embeds = embeds

    def clear_prefill_embeds(self) -> None:
        self._prefill_embeds = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        inputs_embeds = None
        if self._prefill_embeds is not None and forward_batch.forward_mode.is_extend():
            inputs_embeds = self._prefill_embeds
            self._prefill_embeds = None
        hidden_states = self.talker.forward(
            input_ids, positions, forward_batch, inputs_embeds
        )
        return self.logits_processor.forward(
            input_ids,
            hidden_states,
            self.talker.codec_head,
            forward_batch,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        self.talker.load_weights(weights)
