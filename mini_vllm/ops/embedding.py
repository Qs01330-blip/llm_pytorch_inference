import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from mini_vllm.ops._utils import is_ext_loaded


def _use_cuda(tensor: torch.Tensor) -> bool:
    return is_ext_loaded() and tensor.is_cuda


def embedding(input_ids: torch.Tensor, weight: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Fused embedding lookup + optional scale.

    Args:
        input_ids: [num_tokens] or [batch, seq_len] (int32/int64)
        weight: [vocab_size, hidden_dim]
        scale: multiplicative scale (1.0 = no scale)

    Returns:
        [num_tokens, hidden_dim] or [batch, seq_len, hidden_dim]
    """
    orig_shape = input_ids.shape
    ids_flat = input_ids.reshape(-1)

    if _use_cuda(ids_flat) and _use_cuda(weight):
        out = torch.ops.mini_vllm_ops.embedding_forward(ids_flat, weight, scale)
    else:
        out = nn.functional.embedding(ids_flat, weight)
        if scale != 1.0:
            out = out * scale

    if len(orig_shape) > 1:
        return out.reshape(*orig_shape, -1)
    return out


class Embedding(nn.Module):
    """Drop-in replacement for nn.Embedding with optional scale and CUDA acceleration."""

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 1.0):
        super().__init__()
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return embedding(input_ids, self.weight, self.scale)
