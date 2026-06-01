#pragma once
#include <torch/types.h>

namespace mini_vllm {

// Fused embedding lookup + optional scale
// input_ids: [num_tokens] (int32 or int64)
// weight: [vocab_size, hidden_dim]
// scale: scalar (1.0 = no scale)
// output: [num_tokens, hidden_dim]
torch::Tensor embedding_forward(torch::Tensor input_ids, torch::Tensor weight, double scale);

}  // namespace mini_vllm
