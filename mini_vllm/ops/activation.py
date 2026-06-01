import torch
import torch.nn.functional as F
from mini_vllm.ops._utils import is_ext_loaded


def _use_cuda(tensor: torch.Tensor) -> bool:
    """Check if we should use CUDA ops: extension loaded AND tensor on CUDA device."""
    return is_ext_loaded() and tensor.is_cuda


def silu(input: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)

    Args:
        input: [..., D] tensor

    Returns:
        [..., D] tensor with SiLU applied
    """
    if _use_cuda(input):
        return torch.ops.mini_vllm_ops.silu_forward(input)
    return F.silu(input)


def gelu(input: torch.Tensor) -> torch.Tensor:
    """GELU activation (tanh approximation)

    Args:
        input: [..., D] tensor

    Returns:
        [..., D] tensor with GELU applied
    """
    if _use_cuda(input):
        return torch.ops.mini_vllm_ops.gelu_forward(input)
    return F.gelu(input, approximate="tanh")


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: silu(gate) * up

    Args:
        gate: [..., D] tensor
        up: [..., D] tensor (same shape as gate)

    Returns:
        [..., D] tensor = silu(gate) * up
    """
    if _use_cuda(gate):
        return torch.ops.mini_vllm_ops.swiglu_forward(gate, up)
    return F.silu(gate) * up


def geglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused GeGLU: gelu(gate) * up

    Args:
        gate: [..., D] tensor
        up: [..., D] tensor (same shape as gate)

    Returns:
        [..., D] tensor = gelu(gate) * up
    """
    if _use_cuda(gate):
        return torch.ops.mini_vllm_ops.geglu_forward(gate, up)
    return F.gelu(gate, approximate="tanh") * up
