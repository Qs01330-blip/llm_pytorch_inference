#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "mini_vllm/utils.cuh"
#include "mini_vllm/dispatch.h"
#include "mlp.h"

namespace mini_vllm {

// Fused MLP kernel: split + activation + multiply in one pass
// gate_up [..., 2*intermediate] → output [..., intermediate] = act(gate) * up
constexpr int ACT_SILU = 0;
constexpr int ACT_GELU = 1;
constexpr float GELU_COEF_A = 0.044715f;
constexpr float GELU_COEF_B = 0.7978845608028654f;

template <typename scalar_t>
__global__ void fused_mlp_silu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gate_up,
    int intermediate_size,
    int64_t total_elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total_elements;
       idx += blockDim.x * gridDim.x) {
    int64_t row = idx / intermediate_size;
    int64_t col = idx % intermediate_size;
    int64_t gate_idx = row * 2 * intermediate_size + col;
    int64_t up_idx = gate_idx + intermediate_size;

    float g = static_cast<float>(gate_up[gate_idx]);
    float u = static_cast<float>(gate_up[up_idx]);
    float act_g = g / (1.0f + expf(-g));
    output[idx] = static_cast<scalar_t>(act_g * u);
  }
}

template <typename scalar_t>
__global__ void fused_mlp_gelu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gate_up,
    int intermediate_size,
    int64_t total_elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total_elements;
       idx += blockDim.x * gridDim.x) {
    int64_t row = idx / intermediate_size;
    int64_t col = idx % intermediate_size;
    int64_t gate_idx = row * 2 * intermediate_size + col;
    int64_t up_idx = gate_idx + intermediate_size;

    float g = static_cast<float>(gate_up[gate_idx]);
    float u = static_cast<float>(gate_up[up_idx]);
    float g3 = g * g * g;
    float inner = GELU_COEF_B * (g + GELU_COEF_A * g3);
    float act_g = 0.5f * g * (1.0f + tanhf(inner));
    output[idx] = static_cast<scalar_t>(act_g * u);
  }
}

torch::Tensor fused_mlp_forward(torch::Tensor gate_up, int64_t intermediate_size, int64_t activation) {
  TORCH_CHECK(gate_up.is_cuda(), "gate_up must be on CUDA");
  TORCH_CHECK(gate_up.dim() >= 1, "gate_up must be at least 1D");
  TORCH_CHECK(gate_up.size(-1) == 2 * intermediate_size,
      "gate_up last dim must be 2*intermediate_size, got ", gate_up.size(-1));

  auto out_sizes = gate_up.sizes().vec();
  out_sizes.back() = intermediate_size;
  auto output = torch::empty(out_sizes, gate_up.options());

  int64_t total_elements = output.numel();
  int threads = 256;
  int blocks = std::min((total_elements + threads - 1) / threads, (int64_t)65535);

  DISPATCH_FLOATING_TYPES(gate_up.scalar_type(), "fused_mlp_kernel", [&] {
    if (activation == 0) {
      fused_mlp_silu_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          output.data_ptr<scalar_t>(),
          gate_up.data_ptr<scalar_t>(),
          (int)intermediate_size,
          total_elements);
    } else {
      fused_mlp_gelu_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          output.data_ptr<scalar_t>(),
          gate_up.data_ptr<scalar_t>(),
          (int)intermediate_size,
          total_elements);
    }
  });
  return output;
}

}  // namespace mini_vllm
