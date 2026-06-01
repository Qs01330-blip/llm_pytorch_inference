#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "mini_vllm/utils.cuh"
#include "mini_vllm/dispatch.h"
#include "rope.h"

namespace mini_vllm {

// RoPE kernel: fused rotate_half + apply_rotary_pos_emb
// input and cos/sin must have the same number of rows: [N, head_dim]
// For each element:
//   if i < dim/2: out[i] = x[i] * cos[i] - x[i + dim/2] * sin[i]
//   if i >= dim/2: out[i] = x[i] * cos[i] + x[i - dim/2] * sin[i]
template <typename scalar_t>
__global__ void rotary_pos_emb_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    int head_dim) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int half_dim = head_dim / 2;

  const scalar_t* row_in = input + row * head_dim;
  scalar_t* row_out = output + row * head_dim;
  const scalar_t* cos_row = cos + row * head_dim;
  const scalar_t* sin_row = sin + row * head_dim;

  for (int i = tid; i < half_dim; i += blockDim.x) {
    float x1 = static_cast<float>(row_in[i]);
    float x2 = static_cast<float>(row_in[i + half_dim]);
    float c = static_cast<float>(cos_row[i]);
    float s = static_cast<float>(sin_row[i]);

    row_out[i] = static_cast<scalar_t>(x1 * c - x2 * s);
    row_out[i + half_dim] = static_cast<scalar_t>(x2 * c + x1 * s);
  }
}

// Host wrapper
// input: [N, head_dim], cos: [N, head_dim], sin: [N, head_dim]
torch::Tensor rotary_pos_emb_forward(torch::Tensor input, torch::Tensor cos, torch::Tensor sin) {
  auto output = torch::empty_like(input);
  int num_rows = input.size(0);
  int head_dim = input.size(-1);
  int half_dim = head_dim / 2;
  int threads = std::min(((half_dim + 31) / 32) * 32, 1024);

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rotary_pos_emb_kernel", [&] {
    rotary_pos_emb_kernel<scalar_t><<<num_rows, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        head_dim);
  });
  return output;
}

}  // namespace mini_vllm
