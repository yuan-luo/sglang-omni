import socket
from dataclasses import dataclass

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import broadcast_pyobj, set_random_seed

from .model_runner import SGLModelRunner, TalkerSGLModelRunner


def _find_free_port() -> int:
    # TODO (chenyang): we can port this
    # function from sglang but not right now
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class ModelWorkerConfig:
    pass


class ModelWorker:
    def __init__(
        self,
        config: ModelWorkerConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int = 0,
    ):
        self.server_args = server_args

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self._init_model_config()
        self._init_model_runner()

        self.device = self.model_runner.device
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    def _init_model_config(self):
        # TODO(ocss884) add support for mtp
        self.model_config = ModelConfig.from_server_args(
            server_args=self.server_args,
            model_path=self.server_args.model_path,
            model_revision=self.server_args.revision,
            is_draft_model=False,
        )

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def _init_model_runner(self):
        self.model_runner = SGLModelRunner(
            model_config=self.model_config,
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=_find_free_port(),
        )

    def forward_batch_generation(
        self,
        forward_batch,
    ):
        out = self.model_runner.forward(forward_batch=forward_batch)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        return batch_result


class TalkerModelWorker(ModelWorker):
    """ModelWorker variant for the Talker model.

    Overrides _init_model_config to patch the HF config so that:
    - Architecture resolves to Qwen3OmniTalkerForCausalLM
    - KV cache is sized for the Talker dimensions (hidden_size=1024, 20 layers)
    """

    def _init_model_config(self):
        super()._init_model_config()

        hf_config = self.model_config.hf_config
        talker_cfg = getattr(hf_config, "talker_config", {})
        if isinstance(talker_cfg, dict):
            text_cfg = talker_cfg.get("text_config", {})
        else:
            text_cfg = getattr(talker_cfg, "text_config", {})
            if not isinstance(text_cfg, dict):
                text_cfg = text_cfg.__dict__

        # Override architecture for model class resolution
        hf_config.architectures = ["Qwen3OmniTalkerForCausalLM"]

        # Patch hf_text_config for correct KV cache sizing
        hf_text_config = self.model_config.hf_text_config
        for key in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "intermediate_size",
            "max_position_embeddings",
        ):
            if key in text_cfg:
                setattr(hf_text_config, key, text_cfg[key])

        # Update ModelConfig's derived fields
        self.model_config.hidden_size = text_cfg.get("hidden_size", 1024)
        self.model_config.num_attention_heads = text_cfg.get("num_attention_heads", 16)
        self.model_config.num_key_value_heads = text_cfg.get("num_key_value_heads", 2)
        self.model_config.head_dim = text_cfg.get("head_dim", 128)
        self.model_config.num_hidden_layers = text_cfg.get("num_hidden_layers", 20)

    def _init_model_runner(self):
        """Use TalkerSGLModelRunner to reuse the existing TP group."""
        self.model_runner = TalkerSGLModelRunner(
            model_config=self.model_config,
            server_args=self.server_args,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            moe_ep_rank=0,
            moe_ep_size=1,
            pp_rank=0,
            pp_size=1,
            nccl_port=_find_free_port(),
        )
