import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mini_vllm.ops._utils import is_ext_loaded


# ---------------------------------------------------------------------------
# Low-level functional ops (CUDA accelerated with PyTorch fallback)
# ---------------------------------------------------------------------------

def _use_cuda(tensor: torch.Tensor) -> bool:
    """Check if we should use CUDA ops: extension loaded AND tensor on CUDA device."""
    return is_ext_loaded() and tensor.is_cuda


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = weight if weight.dtype == input.dtype else weight.to(input.dtype)
    if _use_cuda(input):
        return torch.ops.mini_vllm_ops.rmsnorm_forward(input, w, eps)
    variance = input.float().pow(2).mean(-1, keepdim=True)
    return (w * input * torch.rsqrt(variance + eps)).to(input.dtype)


def rmsnorm_plus_one(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = weight if weight.dtype == input.dtype else weight.to(input.dtype)
    if _use_cuda(input):
        return torch.ops.mini_vllm_ops.rmsnorm_plus_one_forward(input, w, eps)
    variance = input.float().pow(2).mean(-1, keepdim=True)
    x = input * torch.rsqrt(variance + eps)
    return ((1.0 + w.float()) * x).to(input.dtype)


def fused_add_rmsnorm(residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> tuple:
    """Fused Add + RMSNorm: residual += input; output = weight * residual_norm. Returns (output, residual)."""
    w = weight if weight.dtype == input.dtype else weight.to(input.dtype)
    if _use_cuda(input):
        return torch.ops.mini_vllm_ops.fused_add_rmsnorm_forward(residual, input, w, eps)
    x = residual + input
    variance = x.float().pow(2).mean(-1, keepdim=True)
    out = (w * x * torch.rsqrt(variance + eps)).to(input.dtype)
    return out, x


def fused_add_rmsnorm_plus_one(residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> tuple:
    """Fused Add + RMSNorm + Plus One: residual += input; output = (1 + weight) * residual_norm. Returns (output, residual)."""
    w = weight if weight.dtype == input.dtype else weight.to(input.dtype)
    if _use_cuda(input):
        return torch.ops.mini_vllm_ops.fused_add_rmsnorm_plus_one_forward(residual, input, w, eps)
    x = residual + input
    variance = x.float().pow(2).mean(-1, keepdim=True)
    out = ((1.0 + w.float()) * x * torch.rsqrt(variance + eps)).to(input.dtype)
    return out, x


# ---------------------------------------------------------------------------
# nn.Module classes (drop-in replacements for model code)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm with optional plus_one mode.

    - plus_one=False (default): output = weight * normalize(x), weight init to ones
    - plus_one=True:            output = (1 + weight) * normalize(x), weight init to zeros
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, plus_one: bool = False):
        super().__init__()
        self.weight = Parameter(torch.zeros(hidden_size) if plus_one else torch.ones(hidden_size))
        self.eps = eps
        self.plus_one = plus_one

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.plus_one:
            return rmsnorm_plus_one(x, self.weight, self.eps)
        return rmsnorm(x, self.weight, self.eps)

