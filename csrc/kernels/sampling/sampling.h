#pragma once
#include <torch/types.h>

namespace mini_vllm {

// Greedy sampling: argmax across vocab dimension
// logits: [batch_size, vocab_size] -> output: [batch_size] (int64)
torch::Tensor greedy_sample(torch::Tensor logits);

// Fused top-k + top-p filter (in-place on logits)
// logits: [batch_size, vocab_size]
// Modifies logits: sets filtered positions to -inf
void top_k_top_p_filter(torch::Tensor logits, int64_t top_k, double top_p);

// Fused softmax + multinomial sampling
// logits: [batch_size, vocab_size] -> output: [batch_size] (int64)
torch::Tensor softmax_multinomial_sample(torch::Tensor logits, double temperature);

}  // namespace mini_vllm
