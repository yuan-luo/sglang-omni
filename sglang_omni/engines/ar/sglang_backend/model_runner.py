from __future__ import annotations

import logging
import os

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


class SGLModelRunner(ModelRunner):
    """Thin wrapper to bootstrap SGLang ModelRunner from backend args."""

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
    ) -> None:
        self.model_config = model_config
        self.server_args = server_args
        self.gpu_id = gpu_id

        self._register_omni_model()

        model_config = server_args.get_model_config()
        port_args = PortArgs.init_new(server_args)

        tp_size = server_args.tp_size

        self.nccl_port = port_args.nccl_port

        super().__init__(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def _register_omni_model(self):
        # Ensure sglang_omni.models` custom modeling classes
        # are injected into the model registry before runner initialization.
        os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang_omni.models")


class TalkerSGLModelRunner(SGLModelRunner):
    """SGLModelRunner variant that reuses an existing torch distributed group.

    When multiple SGLang model workers run in the same process (e.g., Thinker
    and Talker), the TP/PP parallel groups are already initialized by the first
    worker. This subclass skips re-initialization of torch distributed and
    model parallel groups, only querying the current available GPU memory.

    It also uses the pre-patched model_config from TalkerModelWorker instead
    of re-reading it from server_args (which would get the Thinker's config).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
    ) -> None:
        self.model_config = model_config
        self.server_args = server_args
        self.gpu_id = gpu_id

        self._register_omni_model()

        # Use the pre-patched model_config instead of re-reading from server_args.
        # TalkerModelWorker._init_model_config() patches architectures and dimensions.
        port_args = PortArgs.init_new(server_args)
        tp_size = server_args.tp_size
        self.nccl_port = port_args.nccl_port

        # Call ModelRunner.__init__ directly (skip SGLModelRunner's re-read)
        ModelRunner.__init__(
            self,
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def init_torch_distributed(self):
        """Skip distributed init — reuse the groups set up by the Thinker."""
        import torch
        from sglang.srt.distributed import get_tp_group

        torch.get_device_module(self.device).set_device(self.gpu_id)
        self.tp_group = get_tp_group()
        logger.info(
            "Talker reusing existing torch distributed (tp_rank=%d, tp_size=%d)",
            self.tp_rank,
            self.tp_size,
        )
        return get_available_gpu_memory(self.device, self.gpu_id)
