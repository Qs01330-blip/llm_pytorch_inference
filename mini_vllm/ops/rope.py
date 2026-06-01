import torch
import math
from mini_vllm.ops._utils import is_ext_loaded


def _use_cuda(tensor: torch.Tensor) -> bool:
    """Check if we should use CUDA ops: extension loaded AND tensor on CUDA device."""
    return is_ext_loaded() and tensor.is_cuda


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0,
                         device: str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin frequencies for rotary position embedding.

    Args:
        dim: Head dimension
        seq_len: Maximum sequence length
        theta: Base frequency for RoPE
        device: Device to store tensors
        dtype: Data type for cos/sin tensors

    Returns:
        cos: [seq_len, dim]
        sin: [seq_len, dim]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: [batch_size, num_heads, seq_len, head_dim]
        k: [batch_size, num_heads, seq_len, head_dim]
        cos: [batch_size, 1, seq_len, head_dim] (already broadcast-ready)
        sin: [batch_size, 1, seq_len, head_dim] (already broadcast-ready)

    Returns:
        q_embed: [batch_size, num_heads, seq_len, head_dim]
        k_embed: [batch_size, num_heads, seq_len, head_dim]
    """
    if _use_cuda(q):
        # CUDA path: fuse rotate_half + apply_rotary
        head_dim = q.size(-1)
        # Expand cos/sin: [batch, 1, seq_len, dim] -> [batch, heads, seq_len, dim]
        # Then flatten all to [-1, dim] for the CUDA kernel
        cos_exp = cos.expand_as(q)
        sin_exp = sin.expand_as(q)

        # .contiguous() is needed because expand creates a non-contiguous view
        # The CUDA kernel requires contiguous input
        q_flat = q.reshape(-1, head_dim).contiguous()
        k_flat = k.reshape(-1, head_dim).contiguous()
        cos_flat = cos_exp.reshape(-1, head_dim).contiguous()
        sin_flat = sin_exp.reshape(-1, head_dim).contiguous()

        q_embed = torch.ops.mini_vllm_ops.rotary_pos_emb_forward(q_flat, cos_flat, sin_flat)
        k_embed = torch.ops.mini_vllm_ops.rotary_pos_emb_forward(k_flat, cos_flat, sin_flat)

        return q_embed.reshape(q.shape), k_embed.reshape(k.shape)
    else:
        # PyTorch fallback — cos/sin already broadcastable with q/k
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
