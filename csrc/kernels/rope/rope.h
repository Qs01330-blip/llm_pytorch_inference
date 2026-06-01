#pragma once
#include <torch/types.h>

namespace mini_vllm {

// Apply rotary position embedding to q and k tensors
// q, k: [batch_size, num_heads, seq_len, head_dim]
// cos, sin: [seq_len, head_dim] (precomputed frequencies)
torch::Tensor rotary_pos_emb_forward(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);

}  // namespace mini_vllm
