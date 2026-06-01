#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "mini_vllm/utils.cuh"
#include "mini_vllm/dispatch.h"
#include "activation.h"

namespace mini_vllm {

// GELU tanh approximation constants
constexpr float GELU_COEF_A = 0.044715f;
constexpr float GELU_COEF_B = 0.7978845608028654f;  // sqrt(2/pi)

// SiLU kernel: out = x * sigmoid(x)
template <typename scalar_t>
__global__ void silu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = static_cast<float>(input[idx]);
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float sig = 1.0f / (1.0f + expf(-x));
    output[idx] = static_cast<scalar_t>(x * sig);
  }
}

// GELU kernel (tanh approximation): out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template <typename scalar_t>
__global__ void gelu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float x = static_cast<float>(input[idx]);
    float x3 = x * x * x;
    float inner = GELU_COEF_B * (x + GELU_COEF_A * x3);
    output[idx] = static_cast<scalar_t>(0.5f * x * (1.0f + tanhf(inner)));
  }
}

// Fused SwiGLU kernel: out = silu(gate) * up
template <typename scalar_t>
__global__ void swiglu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = static_cast<float>(gate[idx]);
    float u = static_cast<float>(up[idx]);
    // SiLU(gate) = gate * sigmoid(gate)
    float sig = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sig;
    output[idx] = static_cast<scalar_t>(silu_g * u);
  }
}

// Fused GeGLU kernel: out = gelu(gate) * up
template <typename scalar_t>
__global__ void geglu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = static_cast<float>(gate[idx]);
    float u = static_cast<float>(up[idx]);
    // GELU(gate) = 0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))
    float g3 = g * g * g;
    float inner = GELU_COEF_B * (g + GELU_COEF_A * g3);
    float gelu_g = 0.5f * g * (1.0f + tanhf(inner));
    output[idx] = static_cast<scalar_t>(gelu_g * u);
  }
}

// Host wrappers
torch::Tensor silu_forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int n = input.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_kernel", [&] {
    silu_kernel<scalar_t><<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        n);
  });
  return output;
}

torch::Tensor gelu_forward(torch::Tensor input) {
  auto output = torch::empty_like(input);
  int n = input.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "gelu_kernel", [&] {
    gelu_kernel<scalar_t><<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        n);
  });
  return output;
}

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
  auto output = torch::empty_like(gate);
  int n = gate.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  DISPATCH_FLOATING_TYPES(gate.scalar_type(), "swiglu_kernel", [&] {
    swiglu_kernel<scalar_t><<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        up.data_ptr<scalar_t>(),
        n);
  });
  return output;
}

torch::Tensor geglu_forward(torch::Tensor gate, torch::Tensor up) {
  auto output = torch::empty_like(gate);
  int n = gate.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  DISPATCH_FLOATING_TYPES(gate.scalar_type(), "geglu_kernel", [&] {
    geglu_kernel<scalar_t><<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        gate.data_ptr<scalar_t>(),
        up.data_ptr<scalar_t>(),
        n);
  });
  return output;
}

}  // namespace mini_vllm
