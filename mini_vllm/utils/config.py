from dataclasses import dataclass, field
import torch
from mini_vllm.utils.logger import logger


@dataclass
class EngineConfig:
    model_path: str
    device: str = "auto"
    dtype: str = "auto"
    block_size: int = 16
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    tp_size: int = 1
    compile: bool = False

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device string."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def get_torch_dtype(self) -> str:
        if self.dtype != "auto":
            return self.dtype
        return "float32" if self.resolve_device() == "cpu" else "float16"

    def get_num_gpu_blocks(self, head_dim: int, num_layers: int, num_heads: int,
                           model_memory_bytes: int = 0) -> int:
        """估算可用的 KV Cache block 数量

        Args:
            model_memory_bytes: 模型权重已占用的 GPU 显存，用于计算 KV cache 真正可用空间。
        """
        if self.resolve_device() == "cpu":
            return 256

        dtype_size = 2 if self.get_torch_dtype() in ("float16", "bfloat16") else 4
        # Each block: 2 (K+V) × block_size × num_heads × head_dim × num_layers × dtype
        block_memory = 2 * self.block_size * num_heads * head_dim * num_layers * dtype_size

        # mem_get_info()[0] reports driver-level free memory, but PyTorch's caching
        # allocator may hold fragmented blocks that make this report 0 even when
        # there's usable space. Fall back to total - model estimate.
        free_memory = torch.cuda.mem_get_info()[0]
        if model_memory_bytes > 0 and free_memory < model_memory_bytes * 0.3:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            cuda_overhead = 600 * 1024**2  # ~600 MB for CUDA context + allocator
            free_memory = max(total_memory - model_memory_bytes - cuda_overhead, 0)
            logger.info(
                f"KV cache budget: estimated from total ({free_memory / 1024**2:.0f} MB free) "
                f"instead of mem_get_info (fragmented allocator)"
            )

        usable_memory = int(free_memory * self.gpu_memory_utilization)
        return max(int(usable_memory / block_memory), 1)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    eos_token_id: int | list[int] = 2  # Overridden by engine from model config
