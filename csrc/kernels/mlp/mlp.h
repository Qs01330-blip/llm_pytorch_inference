#pragma once
#include <torch/types.h>

namespace mini_vllm {

torch::Tensor fused_mlp_forward(torch::Tensor gate_up, int64_t intermediate_size, int64_t activation);

}  // namespace mini_vllm
