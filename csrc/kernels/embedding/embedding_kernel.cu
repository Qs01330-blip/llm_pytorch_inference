#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "mini_vllm/utils.cuh"
#include "mini_vllm/dispatch.h"
#include "embedding.h"

namespace mini_vllm {

// Embedding lookup kernel: output[row] = weight[token_id] * scale
// One block per token, threads cooperatively copy hidden_dim elements.
template <typename scalar_t>
__global__ void embedding_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ input_ids,
    int hidden_dim, float scale) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int token_id = input_ids[row];
  const scalar_t* emb_row = weight + token_id * hidden_dim;
  scalar_t* out_row = output + row * hidden_dim;

  if (scale == 1.0f) {
    for (int i = tid; i < hidden_dim; i += num_threads) {
      out_row[i] = emb_row[i];
    }
  } else {
    for (int i = tid; i < hidden_dim; i += num_threads) {
      out_row[i] = static_cast<scalar_t>(static_cast<float>(emb_row[i]) * scale);
    }
  }
}

// Host wrapper
// input_ids: [num_tokens] (int64)
// weight: [vocab_size, hidden_dim]
// scale: double
torch::Tensor embedding_forward(torch::Tensor input_ids, torch::Tensor weight, double scale) {
  TORCH_CHECK(input_ids.is_cuda(), "input_ids must be on CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
  TORCH_CHECK(input_ids.dim() == 1, "input_ids must be 1D [num_tokens]");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [vocab_size, hidden_dim]");

  int num_tokens = input_ids.size(0);
  int hidden_dim = weight.size(1);

  // Ensure input_ids is int64
  auto ids = input_ids.to(torch::kInt64);

  auto output = torch::empty({num_tokens, hidden_dim}, weight.options());
  int threads = std::min(((hidden_dim + 31) / 32) * 32, 1024);

  DISPATCH_FLOATING_TYPES(weight.scalar_type(), "embedding_kernel", [&] {
    embedding_kernel<scalar_t><<<num_tokens, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        ids.data_ptr<int64_t>(),
        hidden_dim, static_cast<float>(scale));
  });
  return output;
}

}  // namespace mini_vllm
