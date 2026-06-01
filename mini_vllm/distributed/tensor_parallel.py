# mini_vllm/distributed/tensor_parallel.py
"""
Tensor Parallel interface placeholder

Current implementation is a stub. Future expansion will add:
- ColumnParallelLinear: column-partitioned linear layer
- RowParallelLinear: row-partitioned linear layer
- TensorParallelEngine: wraps LLMEngine for transparent multi-GPU sharding
"""
from dataclasses import dataclass
from mini_vllm.utils.logger import logger


@dataclass
class TensorParallelConfig:
    tp_size: int = 1
    tp_rank: int = 0
    backend: str = "nccl"


class TensorParallelEngine:
    """Tensor Parallel Engine (placeholder)"""

    def __init__(self, config: TensorParallelConfig):
        self.config = config
        if config.tp_size > 1:
            logger.warning("Tensor parallel not yet implemented, falling back to single GPU")
        logger.info(f"TensorParallelEngine: tp_size={config.tp_size}, tp_rank={config.tp_rank}")


def init_distributed(tp_size: int = 1) -> TensorParallelConfig:
    """Initialize distributed environment (placeholder)"""
    if tp_size > 1:
        raise NotImplementedError("Tensor parallel will be implemented in a future version")
    return TensorParallelConfig(tp_size=1, tp_rank=0)
