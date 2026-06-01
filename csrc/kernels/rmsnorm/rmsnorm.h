#pragma once
#include <torch/types.h>

namespace mini_vllm {

torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps);
torch::Tensor rmsnorm_plus_one_forward(torch::Tensor input, torch::Tensor weight, double eps);
std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_forward(
    torch::Tensor residual, torch::Tensor input, torch::Tensor weight, double eps);
std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_plus_one_forward(
    torch::Tensor residual, torch::Tensor input, torch::Tensor weight, double eps);

}  // namespace mini_vllm
