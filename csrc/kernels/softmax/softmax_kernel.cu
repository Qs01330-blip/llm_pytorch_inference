#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "mini_vllm/utils.cuh"
#include "mini_vllm/dispatch.h"
#include "softmax.h"

namespace mini_vllm {

// Softmax kernel: one block per row, 3-pass approach.
// Uses inline shared-memory reductions to avoid cross-function issues.
template <typename scalar_t>
__global__ void softmax_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int N) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const scalar_t* row_in = input + row * N;
  scalar_t* row_out = output + row * N;

  extern __shared__ float smem[];
  float* sdata = smem;  // [block_size]

  // Pass 1: find max using inline block reduction
  float max_val = -FLT_MAX;
  for (int i = tid; i < N; i += block_size) {
    float val = static_cast<float>(row_in[i]);
    if (val > max_val) max_val = val;
  }
  sdata[tid] = max_val;
  __syncthreads();

  for (int s = block_size / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  // Last warp reduction (no sync needed within a warp)
  if (tid < 32) {
    volatile float* vdata = sdata;
    if (block_size >= 64) vdata[tid] = fmaxf(vdata[tid], vdata[tid + 32]);
    vdata[tid] = fmaxf(vdata[tid], vdata[tid + 16]);
    vdata[tid] = fmaxf(vdata[tid], vdata[tid + 8]);
    vdata[tid] = fmaxf(vdata[tid], vdata[tid + 4]);
    vdata[tid] = fmaxf(vdata[tid], vdata[tid + 2]);
    vdata[tid] = fmaxf(vdata[tid], vdata[tid + 1]);
  }
  __syncthreads();
  max_val = sdata[0];

  // Pass 2: compute sum of exp(x - max) using inline block reduction
  float sum = 0.0f;
  for (int i = tid; i < N; i += block_size) {
    sum += expf(static_cast<float>(row_in[i]) - max_val);
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int s = block_size / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid < 32) {
    volatile float* vdata = sdata;
    if (block_size >= 64) vdata[tid] += vdata[tid + 32];
    vdata[tid] += vdata[tid + 16];
    vdata[tid] += vdata[tid + 8];
    vdata[tid] += vdata[tid + 4];
    vdata[tid] += vdata[tid + 2];
    vdata[tid] += vdata[tid + 1];
  }
  __syncthreads();
  float inv_sum = 1.0f / sdata[0];

  // Pass 3: write normalized output
  for (int i = tid; i < N; i += block_size) {
    float val = expf(static_cast<float>(row_in[i]) - max_val) * inv_sum;
    row_out[i] = static_cast<scalar_t>(val);
  }
}

torch::Tensor softmax_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be on CUDA");

  auto output = torch::empty_like(input);
  int N = input.size(-1);
  int num_rows = input.numel() / N;
  int threads = std::min(((N + 31) / 32) * 32, 1024);
  int smem_size = threads * sizeof(float);

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax_kernel", [&] {
    softmax_kernel<scalar_t><<<num_rows, threads, smem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        N);
  });
  return output;
}

}  // namespace mini_vllm
