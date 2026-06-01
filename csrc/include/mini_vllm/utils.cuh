#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/types.h>

namespace mini_vllm {

// Warp-level reduction sum
__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  return val;
}

// Block-level reduction sum using shared memory
// blockDim.x must be a multiple of 32
__inline__ __device__ float block_reduce_sum(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  int num_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
  if (wid == 0) val = warp_reduce_sum(val);
  return val;
}

// Warp-level reduction max
__inline__ __device__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  return val;
}

// Block-level reduction max using shared memory
__inline__ __device__ float block_reduce_max(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warp_reduce_max(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  int num_warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
  if (wid == 0) val = warp_reduce_max(val);
  return val;
}

}  // namespace mini_vllm
