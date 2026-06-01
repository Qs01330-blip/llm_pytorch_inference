import torch
from mini_vllm.ops._utils import is_ext_loaded


def _use_cuda(tensor: torch.Tensor) -> bool:
    return is_ext_loaded() and tensor.is_cuda


def softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along the specified dimension.

    Args:
        input: any shape tensor
        dim: dimension to softmax over (default -1)

    Returns:
        same shape, softmax applied along dim
    """
    if dim != -1 and dim != input.dim() - 1:
        # Only CUDA-accelerate last-dim softmax
        return torch.softmax(input, dim=dim)

    if _use_cuda(input):
        return torch.ops.mini_vllm_ops.softmax_forward(input)
    return torch.softmax(input, dim=dim)
