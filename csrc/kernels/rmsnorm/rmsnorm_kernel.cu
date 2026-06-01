#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "mini_vllm/utils.cuh"
#include "mini_vllm/dispatch.h"
#include "rmsnorm.h"

namespace mini_vllm {

// Standard RMSNorm: y = weight * x / sqrt(mean(x^2) + eps)
template <typename scalar_t>
__global__ void rmsnorm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    float eps, int hidden_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* row_in = input + row * hidden_size;
  scalar_t* row_out = output + row * hidden_size;

  // Compute sum of squares in float
  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(row_in[i]);
    sum_sq += val * val;
  }
  sum_sq = block_reduce_sum(sum_sq);

  __shared__ float rms;
  if (tid == 0) {
    rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
  }
  __syncthreads();

  // Normalize and scale
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(row_in[i]);
    float w = static_cast<float>(weight[i]);
    row_out[i] = static_cast<scalar_t>(w * val * rms);
  }
}

// Plus-one RMSNorm: y = (1 + weight) * x / sqrt(mean(x^2) + eps)
template <typename scalar_t>
__global__ void rmsnorm_plus_one_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    float eps, int hidden_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* row_in = input + row * hidden_size;
  scalar_t* row_out = output + row * hidden_size;

  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(row_in[i]);
    sum_sq += val * val;
  }
  sum_sq = block_reduce_sum(sum_sq);

  __shared__ float rms;
  if (tid == 0) {
    rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
  }
  __syncthreads();

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(row_in[i]);
    float w = static_cast<float>(weight[i]);
    row_out[i] = static_cast<scalar_t>((1.0f + w) * val * rms);
  }
}

// Fused Add + RMSNorm: residual += input; output = weight * residual / sqrt(mean(residual^2) + eps)
template <typename scalar_t>
__global__ void fused_add_rmsnorm_kernel(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    float eps, int hidden_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  scalar_t* row_res = residual + row * hidden_size;
  const scalar_t* row_in = input + row * hidden_size;
  scalar_t* row_out = output + row * hidden_size;

  // Pass 1: add residual + input, compute sum of squares
  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float r = static_cast<float>(row_res[i]);
    float x = static_cast<float>(row_in[i]);
    float val = r + x;
    row_res[i] = static_cast<scalar_t>(val);  // in-place update residual
    sum_sq += val * val;
  }
  sum_sq = block_reduce_sum(sum_sq);

  __shared__ float rms;
  if (tid == 0) {
    rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
  }
  __syncthreads();

  // Pass 2: normalize and scale
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(row_res[i]);
    float w = static_cast<float>(weight[i]);
    row_out[i] = static_cast<scalar_t>(w * val * rms);
  }
}

// Fused Add + RMSNorm + Plus One: residual += input; output = (1 + weight) * residual / sqrt(mean(residual^2) + eps)
template <typename scalar_t>
__global__ void fused_add_rmsnorm_plus_one_kernel(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    float eps, int hidden_size) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  scalar_t* row_res = residual + row * hidden_size;
  const scalar_t* row_in = input + row * hidden_size;
  scalar_t* row_out = output + row * hidden_size;

  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float r = static_cast<float>(row_res[i]);
    float x = static_cast<float>(row_in[i]);
    float val = r + x;
    row_res[i] = static_cast<scalar_t>(val);
    sum_sq += val * val;
  }
  sum_sq = block_reduce_sum(sum_sq);

  __shared__ float rms;
  if (tid == 0) {
    rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
  }
  __syncthreads();

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(row_res[i]);
    float w = static_cast<float>(weight[i]);
    row_out[i] = static_cast<scalar_t>((1.0f + w) * val * rms);
  }
}

// Host wrappers
torch::Tensor rmsnorm_forward(torch::Tensor input, torch::Tensor weight, double eps) {
  auto output = torch::empty_like(input);
  int num_rows = input.numel() / input.size(-1);
  int hidden_size = input.size(-1);
  int threads = std::min(((hidden_size + 31) / 32) * 32, 1024);

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rmsnorm_kernel", [&] {
    rmsnorm_kernel<scalar_t><<<num_rows, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        eps, hidden_size);
  });
  return output;
}

torch::Tensor rmsnorm_plus_one_forward(torch::Tensor input, torch::Tensor weight, double eps) {
  auto output = torch::empty_like(input);
  int num_rows = input.numel() / input.size(-1);
  int hidden_size = input.size(-1);
  int threads = std::min(((hidden_size + 31) / 32) * 32, 1024);

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rmsnorm_plus_one_kernel", [&] {
    rmsnorm_plus_one_kernel<scalar_t><<<num_rows, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        eps, hidden_size);
  });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_forward(
    torch::Tensor residual, torch::Tensor input, torch::Tensor weight, double eps) {
  auto output = torch::empty_like(input);
  int num_rows = input.numel() / input.size(-1);
  int hidden_size = input.size(-1);
  int threads = std::min(((hidden_size + 31) / 32) * 32, 1024);

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_add_rmsnorm_kernel", [&] {
    fused_add_rmsnorm_kernel<scalar_t><<<num_rows, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        eps, hidden_size);
  });
  return {output, residual};
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm_plus_one_forward(
    torch::Tensor residual, torch::Tensor input, torch::Tensor weight, double eps) {
  auto output = torch::empty_like(input);
  int num_rows = input.numel() / input.size(-1);
  int hidden_size = input.size(-1);
  int threads = std::min(((hidden_size + 31) / 32) * 32, 1024);

  DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_add_rmsnorm_plus_one_kernel", [&] {
    fused_add_rmsnorm_plus_one_kernel<scalar_t><<<num_rows, threads, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        eps, hidden_size);
  });
  return {output, residual};
}

}  // namespace mini_vllm
