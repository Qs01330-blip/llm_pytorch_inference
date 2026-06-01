#pragma once
#include <torch/types.h>

namespace mini_vllm {

// Softmax on the last dimension.
// input: any shape [..., N]
// output: same shape, softmax applied along last dim
torch::Tensor softmax_forward(torch::Tensor input);

}  // namespace mini_vllm
