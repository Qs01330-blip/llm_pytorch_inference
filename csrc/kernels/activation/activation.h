#pragma once
#include <torch/types.h>

namespace mini_vllm {

// SiLU activation: out = x * sigmoid(x)
torch::Tensor silu_forward(torch::Tensor input);

// GELU activation (tanh approximation): out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
torch::Tensor gelu_forward(torch::Tensor input);

// Fused SwiGLU: out = silu(gate) * up = gate * sigmoid(gate) * up
torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up);

// Fused GeGLU: out = gelu(gate) * up
torch::Tensor geglu_forward(torch::Tensor gate, torch::Tensor up);

}  // namespace mini_vllm
