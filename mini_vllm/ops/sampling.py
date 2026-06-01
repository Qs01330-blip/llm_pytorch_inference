import torch
from mini_vllm.ops._utils import is_ext_loaded


def _use_cuda(tensor: torch.Tensor) -> bool:
    """Check if we should use CUDA ops: extension loaded AND tensor on CUDA device."""
    return is_ext_loaded() and tensor.is_cuda


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling: argmax across vocab dimension.

    Args:
        logits: [batch_size, vocab_size]

    Returns:
        token_ids: [batch_size] (int64)
    """
    if _use_cuda(logits):
        return torch.ops.mini_vllm_ops.greedy_sample(logits)
    return logits.argmax(dim=-1).to(torch.int64)


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Fused top-k + top-p filter. Sets filtered positions to -inf.

    Args:
        logits: [batch_size, vocab_size]
        top_k: number of top candidates to keep (0 = disabled)
        top_p: cumulative probability threshold (1.0 = disabled)

    Returns:
        filtered logits: [batch_size, vocab_size]
    """
    logits = logits.clone()
    if _use_cuda(logits):
        torch.ops.mini_vllm_ops.top_k_top_p_filter(logits, top_k, top_p)
        return logits

    # PyTorch fallback
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    return logits


def softmax_multinomial_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Fused temperature scaling + softmax + multinomial sampling.

    Args:
        logits: [batch_size, vocab_size]
        temperature: sampling temperature

    Returns:
        token_ids: [batch_size] (int64)
    """
    if _use_cuda(logits):
        return torch.ops.mini_vllm_ops.softmax_multinomial_sample(logits, temperature)

    # PyTorch fallback
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    token_ids = torch.multinomial(probs, num_samples=1)
    return token_ids.squeeze(-1).to(torch.int64)
