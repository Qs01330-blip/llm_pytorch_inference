import torch
from mini_vllm.utils.config import SamplingParams


class Sampler:
    def __init__(self):
        # Lazy-load CUDA ops on first use
        self._cuda_sampling = None

    def _try_load_cuda(self):
        if self._cuda_sampling is not None:
            return self._cuda_sampling
        try:
            from mini_vllm.ops.sampling import greedy_sample, top_k_top_p_filter, softmax_multinomial_sample
            self._cuda_sampling = (greedy_sample, top_k_top_p_filter, softmax_multinomial_sample)
            return self._cuda_sampling
        except Exception:
            self._cuda_sampling = False
            return False

    def sample(self, logits: torch.Tensor, sampling_params: SamplingParams) -> list[int]:
        """
        Args:
            logits: [batch_size, vocab_size]
            sampling_params: sampling parameters
        Returns:
            sampled token_id for each sequence
        """
        if sampling_params.temperature == 0.0:
            return self._greedy(logits)
        return self._sample_with_temperature(logits, sampling_params)

    def _greedy(self, logits: torch.Tensor) -> list[int]:
        cuda_ops = self._try_load_cuda()
        if cuda_ops and logits.is_cuda:
            greedy_sample_fn = cuda_ops[0]
            return greedy_sample_fn(logits).tolist()
        return logits.argmax(dim=-1).tolist()

    def _sample_with_temperature(self, logits: torch.Tensor, params: SamplingParams) -> list[int]:
        cuda_ops = self._try_load_cuda()
        if cuda_ops and logits.is_cuda:
            _, top_k_top_p_fn, softmax_multinomial_fn = cuda_ops
            # Fused top-k + top-p filter
            if params.top_k > 0 or params.top_p < 1.0:
                logits = top_k_top_p_fn(logits, params.top_k, params.top_p)
            # Fused softmax + multinomial
            return softmax_multinomial_fn(logits, params.temperature).tolist()

        # PyTorch fallback
        logits = logits / params.temperature

        if params.top_k > 0:
            logits = self._top_k_filter(logits, params.top_k)

        if params.top_p < 1.0:
            logits = self._top_p_filter(logits, params.top_p)

        # Fallback: if all logits are -inf (over-filtered), use greedy
        if (logits == float("-inf")).all(dim=-1).any():
            probs = torch.softmax(logits, dim=-1)
            nan_mask = probs.isnan().any(dim=-1)
            if nan_mask.any():
                # Over-filtered: fall back to greedy (logits already temperature-scaled)
                return logits.argmax(dim=-1).tolist()

        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1)
        return token_ids.squeeze(-1).tolist()

    def _top_k_filter(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep only top-k logits, mask rest to -inf."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")
        return logits

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus sampling: keep smallest set with cumulative prob >= top_p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.clone()
        sorted_logits[sorted_indices_to_remove] = float("-inf")

        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
        return logits
